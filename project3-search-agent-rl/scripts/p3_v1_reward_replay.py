#!/usr/bin/env python
"""P3 Phase 4B/4B.1: historical-rollout replay of the frozen Search-aware GRPO v1 reward.

Replays the ENTIRE historical formal training rollouts (segments 0-50 / 50-100 /
100-300) through the single implementation source
(searchr1_repro/search_v1_reward.py, docs .../P3_PHASE4_SEARCH_CAUSAL_DIAG_RESULT_2026-08-19.md §9)
and enforces the hard replay gates:

  1. useful-search-correct mean  > direct-correct mean
  2. irrelevant-search-correct  <= direct-correct  (per-episode, max <= 1.0)
  3. answer-leak-search-correct <= direct-correct  (per-episode, max <= 1.0)
  4. redundant-search-correct   <= direct-correct  (per-episode, max <= 1.0)
  5. invalid < format-wrong < direct-correct (class means)

Evidence is rebuilt from the REAL corpus (document_ids -> wiki-18 text via
CorpusReader), ground truth from train.parquet env_kwargs, terminal score via
the same compute_score(format_score=0.1) the env uses.

Phase 4B.1 (trainer-exact, patch 0008) additionally simulates the EXACT training
advantage path of compute_grpo_trajectory_return_advantage: per-episode return
(== placed sum == component sum, exact cents) -> per-uid GRPO mean/std over the
TRAJECTORY returns (torch.std sample std -> numpy ddof=1; len==1 group -> mean
0/std 1) -> trajectory advantage broadcast to every record of the trajectory
(here each episode is one record, so the broadcast is the per-trajectory value).
Reported:

  - useful-search trajectory / search-record advantage DIRECTION (the Phase
    4B.1 fix: a useful search trajectory's search step no longer scores
    negative on its instant 0.15)
  - direct / irrelevant / invalid / leak class trajectory advantages
  - group completeness: exactly 5 trajectories per uid? missing/duplicate
    traj_uids (report only -- coverage loss is expected where gt/question
    cannot be resolved); traj_uid shared across uids (integrity, fail-closed
    mirror of the training function) is a HARD gate
  - shuffled-evidence false-positive rate ((i+17) mod n permutation over the
    searched episodes, same scheme as the diag2 counterfactual) vs real hit
    rate, vs the old substring rule's null rate and the diag2 oracle hit rate
    on confirm256 (context)
  - component sum / trajectory return / placed reward three-way consistency

Pass criteria (gates 6-11):
  - useful-search trajectory search records are NOT negative on average
  - useful class trajectory adv > direct class (T2 above T1)
  - T4/T7 not above T1 IN THE SAME QUESTION GROUP (matched in-group max
    comparison; cross-group class means are composition artifacts)
  - no traj_uid shared across uids (data integrity; the training function
    fails closed on this)
  - token-rule null hit rate < real hit rate (matcher discriminates)

Historical audit record_scores reflect the OLD reward, so they are intentionally
NOT compared with v1 totals; the sum-consistency check is internal to v1.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchr1_repro.search_v1_reward import (  # noqa: E402
    episode_totals,
    evidence_hit_in_docs,
    norm_text,
    search_step_components,
    terminal_step_components,
    valid_aliases,
)
from verl.utils.reward_score.search_r1_like_qa_em import compute_score  # noqa: E402

DATA_ROOT = Path("/media/imc/data/project3-search-agent-rl")
TRAIN_PARQUET = DATA_ROOT / "datasets/searchr1-upstream/train.parquet"
CORPUS = DATA_ROOT / "indexes/searchr1-wiki18-e5/prepared/wiki-18.jsonl"
OFFSETS = DATA_ROOT / "indexes/searchr1-wiki18-e5/prepared/wiki-18.offsets.npy"
TOKENIZER_PATH = DATA_ROOT / "models/Qwen2.5-3B"
OUT_PATH = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/p3_v1_reward_replay_20260819.json")

SEGMENT_RUNS = [
    ("seg0-50", "p3-formal-segment-0-50-fsdp6-b66-n5-s0-20260816a"),
    ("seg50-100", "p3-formal-segment-50-100-fsdp6-b66-n5-s0-20260817a"),
    ("seg100-300", "p3-formal-segment-100-300-fsdp6-b66-n5-s0-20260817b"),
]

Q_MARKER = "Now answer the following question:\n"
Q_SUFFIX = "\n\nNow it's your turn to respond"

# diag2 counterfactual oracle hit rate on confirm256 (gates/p3_diag2_counterfactual_analysis_20260819.json,
# /base/oracle/evidence_hit = 167 of 256) -- context for the null-hit comparison.
DIAG2_ORACLE_HIT = 167 / 256.0


class CorpusReader:
    def __init__(self, corpus_path: Path, offsets_path: Path):
        self._fd = os.open(corpus_path, os.O_RDONLY)
        self._offsets = np.load(offsets_path, mmap_mode="r")

    def get(self, index: int) -> dict:
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return json.loads(os.pread(self._fd, end - start, start))

    def close(self) -> None:
        os.close(self._fd)
        self._fd = -1


def decode_prompt_question(tok, ids, att, prompt_width: int) -> str | None:
    vp = [i for i in range(prompt_width) if att[i]]
    if not vp:
        return None
    text = tok.decode(ids[vp[0] : vp[-1] + 1], skip_special_tokens=False)
    if Q_MARKER not in text:
        return None
    return text.split(Q_MARKER, 1)[1].split(Q_SUFFIX, 1)[0].strip()


def decode_response_text(tok, ids, att, prompt_width: int, response_width: int) -> str:
    vr = [i for i in range(prompt_width, prompt_width + response_width) if att[i]]
    if not vr:
        return ""
    return tok.decode(ids[vr[0] : vr[-1] + 1], skip_special_tokens=False)


def load_gt_map() -> dict:
    df = pd.read_parquet(TRAIN_PARQUET, columns=["env_kwargs"])
    m: dict[str, list[str]] = {}
    for ek in df["env_kwargs"].tolist():
        q = str(ek["question"])
        if q not in m:
            m[q] = list(ek["ground_truth"]["target"])
    return m


def load_trajectories(segments, gt_map: dict, tokenizer_path: Path) -> tuple[list[dict], list[dict]]:
    """Load + dedup trajectories by traj_uid; returns (trajs, uid_crossing_violations).

    A traj_uid may legitimately repeat inside a file (adjust_batch duplicate
    rows of the same step) and across files of the same rollout; those are
    deduped by env_step. A traj_uid observed under TWO DIFFERENT uids is a data
    integrity violation (the training function compute_grpo_trajectory_return_
    advantage fails closed on exactly this) -- reported, first uid kept.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    trajs: dict[str, dict] = {}
    violations: list[dict] = []
    for seg, run in segments:
        run_dir = DATA_ROOT / "runs" / run / "rollouts"
        for aud_path in sorted(run_dir.glob("*.audit.jsonl")):
            for line in aud_path.open():
                a = json.loads(line)
                m = a["metadata"]
                traj_uid = m["traj_uid"]
                t = trajs.get(traj_uid)
                if t is None:
                    q = decode_prompt_question(tok, a["input_ids"], a["attention_mask"], a["prompt_width"])
                    t = {
                        "uid": m["uid"],
                        "traj_uid": traj_uid,
                        "question": q,
                        "gt": gt_map.get(q) if q is not None else None,
                        "steps": {},  # env_step -> output text (deduped)
                        "searches": [],  # in env order
                        "search_steps_seen": set(),  # adjust_batch duplicate rows
                    }
                    trajs[traj_uid] = t
                elif t["uid"] != m["uid"]:
                    violations.append({"traj_uid": traj_uid, "uid_first": t["uid"], "uid_seen": m["uid"],
                                       "file": str(aud_path)})
                    continue
                step = m["env_step"]
                if step not in t["steps"]:
                    t["steps"][step] = decode_response_text(
                        tok, a["input_ids"], a["attention_mask"], a["prompt_width"], a["response_width"]
                    )
                ret = m.get("retrieval")
                if ret is not None and step not in t["search_steps_seen"]:
                    t["search_steps_seen"].add(step)
                    t["searches"].append(
                        {
                            "env_step": step,
                            "query": ret.get("query"),
                            "status": ret.get("status"),
                            "document_ids": ret.get("document_ids") or [],
                        }
                    )
    out = []
    for t in trajs.values():
        if not t["steps"]:
            continue
        max_step = max(t["steps"])
        # the env concatenates the whole chat history (actions interleaved with
        # observations); only <answer> matters for EM extraction, so joining the
        # step responses in env order reproduces the terminal score.
        t["chat_history_str"] = "".join(t["steps"][s] for s in sorted(t["steps"]))
        t["final_output"] = t["steps"][max_step]
        out.append(t)
    return out, violations


def compute_episode_v1(t: dict, corpus: CorpusReader) -> dict | None:
    """v1 step-attributed reward for one episode; None when gt is missing."""
    if t["gt"] is None:
        return None
    gt_aliases = valid_aliases(t["gt"])
    step_comps = []
    had_effective = False
    doc_blobs = []
    for i, s in enumerate(t["searches"]):
        docs = []
        for did in s["document_ids"]:
            try:
                docs.append(corpus.get(int(did)))
            except (ValueError, IndexError):
                pass
        contents = [d.get("contents", "") for d in docs]
        blob = " ".join(contents)
        doc_blobs.append(blob)
        doc_text = "\n<information>" + blob + "</information>\n"
        sv = search_step_components(
            query=s["query"],
            status=s["status"],
            doc_text=doc_text,
            gt_aliases=gt_aliases,
            question=t["question"],
            prior_search_count=i,
        )
        sv["env_step"] = s["env_step"]
        step_comps.append(sv)
        if sv["evidence_effective"]:
            had_effective = True
    reward = float(compute_score(t["chat_history_str"], {"target": t["gt"]}, format_score=0.1))
    em = reward >= 1.0
    term = terminal_step_components(r_answer_total=reward, em=em, had_effective_evidence=had_effective)
    totals = episode_totals(step_comps, term)
    placed_c = sum(int(sv["step_shaping_c"]) for sv in step_comps) + int(term["answer_reward_c"]) \
        + int(term["format_reward_c"]) + int(term["sce_c"])
    return {
        "uid": t["uid"],
        "traj_uid": t["traj_uid"],
        "question": t["question"],
        "gt": t["gt"],
        "docs_blob": "\n<information>" + " ".join(doc_blobs) + "</information>\n" if doc_blobs else "",
        "n_searches": len(step_comps),
        "em": em,
        "r_answer_total": reward,
        "has_evidence_effective": any(sv["evidence_effective"] for sv in step_comps),
        "has_evidence_credit": any(sv["evidence_credit"] for sv in step_comps),
        "has_leak": any(sv["answer_leak"] for sv in step_comps),
        "has_invalid": any(sv["invalid_or_error"] for sv in step_comps),
        "total_reward_c": totals["total_reward_c"],
        "placed_c": placed_c,
        "components": totals,
        "step_comps": step_comps,
    }


def is_useful(e: dict) -> bool:
    return e["n_searches"] > 0 and e["em"] and e["has_evidence_effective"]


def is_direct(e: dict) -> bool:
    return e["n_searches"] == 0 and e["em"]


def is_irrelevant(e: dict) -> bool:
    return e["n_searches"] > 0 and e["em"] and not e["has_evidence_effective"]


def is_leak(e: dict) -> bool:
    return e["em"] and e["has_leak"]


def is_invalid(e: dict) -> bool:
    return e["r_answer_total"] == 0.0 and e["has_invalid"]


def trainer_exact_grpo(evals: list[dict]) -> dict:
    """Mirror compute_grpo_trajectory_return_advantage (patch 0008) offline.

    Per-uid GRPO over TRAJECTORY returns (not per-record sums): mean/std over
    the distinct trajectory returns of a uid (torch.std sample std -> numpy
    ddof=1; len==1 group -> mean 0/std 1), trajectory advantage broadcast to
    every record of the trajectory. Here each episode IS one trajectory, so the
    episode's advantage IS the broadcast trajectory advantage.
    """
    ordered = sorted(evals, key=lambda e: (str(e["uid"]), str(e["traj_uid"])))
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for e in ordered:
        by_uid[str(e["uid"])].append(e)
    uid_stats: dict[str, dict] = {}
    traj_uids_by_uid: dict[str, list[str]] = {}
    for uid, es in by_uid.items():
        rs = np.array([e["total_reward_c"] / 100.0 for e in es])
        uid_stats[uid] = {
            "mean": float(rs.mean()) if len(rs) > 1 else 0.0,
            "std": float(rs.std(ddof=1)) if len(rs) > 1 else 1.0,
            "n_trajs": len(es),
        }
        traj_uids_by_uid[uid] = sorted(str(e["traj_uid"]) for e in es)
    for e in ordered:
        s = uid_stats[str(e["uid"])]
        e["trajectory_return"] = e["total_reward_c"] / 100.0
        e["trajectory_adv"] = (e["trajectory_return"] - s["mean"]) / (s["std"] + 1e-6)

    group_sizes = Counter(s["n_trajs"] for s in uid_stats.values())
    not_five = [uid for uid, s in uid_stats.items() if s["n_trajs"] != 5]
    # duplicate traj_uid across uids (within-uid duplicates are impossible: the
    # loader keys trajectories by traj_uid)
    tid2uid = {}
    dup_tids = []
    for uid, tids in traj_uids_by_uid.items():
        for tid in tids:
            if tid in tid2uid and tid2uid[tid] != uid:
                dup_tids.append(tid)
            tid2uid[tid] = uid

    def class_stats(pred) -> dict:
        members = [e for e in ordered if pred(e)]
        traj_advs = [e["trajectory_adv"] for e in members]
        # a "search record" is any record of the trajectory that carries model
        # actions; in the replay the trajectory has one record, so the search
        # record advantage equals the trajectory advantage by construction. In
        # training (multi-record trajectories) the SAME value is broadcast to
        # every record, so this field documents the direction fix.
        search_rec_advs = [e["trajectory_adv"] for e in members if e["n_searches"] > 0]
        return {
            "n": len(members),
            "traj_adv_mean": float(np.mean(traj_advs)) if traj_advs else None,
            "search_record_adv_mean": float(np.mean(search_rec_advs)) if search_rec_advs else None,
        }

    return {
        "n_groups": len(uid_stats),
        "group_size_distribution": {str(k): v for k, v in sorted(group_sizes.items())},
        "n_groups_not_exactly_5": len(not_five),
        "groups_not_exactly_5_sample": sorted(not_five)[:5],
        "n_duplicate_traj_uid_across_uids": len(dup_tids),
        "duplicate_traj_uid_sample": dup_tids[:5],
        "class_trajectory_adv": {
            "useful_search_correct": class_stats(is_useful),
            "direct_correct": class_stats(is_direct),
            "irrelevant_search_correct": class_stats(is_irrelevant),
            "answer_leak_correct": class_stats(is_leak),
            "invalid_no_answer": class_stats(is_invalid),
        },
    }


def alias_null_hit_rates(evals: list[dict]) -> dict:
    """Shuffled-evidence null hit rate (diag2 scheme j=(i+17) mod n).

    The real hit rate counts the NEW token-boundary matcher's hits on the REAL
    retrieved docs of each search step; the null rates count hits on the docs
    of a PERMUTED (i+17 mod n) episode's search, both with the new token rule
    and with the old substring rule (norm_text contains), for contrast.
    """
    searched = [e for e in evals if e["n_searches"] > 0]
    searched = sorted(searched, key=lambda e: (str(e["uid"]), str(e["traj_uid"])))
    n = len(searched)
    real_hits = 0
    null_hits_new = 0
    null_hits_old = 0
    n_steps = 0
    for i, e in enumerate(searched):
        aliases = valid_aliases(e["gt"])
        old_aliases = [norm_text(a) for a in e["gt"]]
        for sv in e["step_comps"]:
            n_steps += 1
            real_hits += int(sv["evidence_hit"])
        j = (i + 17) % n
        other_text = searched[j]["docs_blob"]
        null_hits_new += int(evidence_hit_in_docs(other_text, aliases))
        blob_old = norm_text(other_text)
        null_hits_old += int(any(a in blob_old for a in old_aliases))
    return {
        "n_searched_episodes": n,
        "n_search_steps": n_steps,
        "real_hit_rate": real_hits / n_steps if n_steps else None,
        "null_hit_rate_token_rule": null_hits_new / n_steps if n_steps else None,
        "null_hit_rate_old_substring_rule": null_hits_old / n_steps if n_steps else None,
        "oracle_hit_rate_diag2_confirm256_context": DIAG2_ORACLE_HIT,
    }


def main() -> int:
    gt_map = load_gt_map()
    corpus = CorpusReader(CORPUS, OFFSETS)
    try:
        trajs, uid_crossing = load_trajectories(SEGMENT_RUNS, gt_map, TOKENIZER_PATH)
        print(f"[V1_REPLAY] loaded {len(trajs)} trajectories, {len(uid_crossing)} uid-crossing violations",
              flush=True)
        evals = [e for t in trajs if (e := compute_episode_v1(t, corpus)) is not None]
        print(f"[V1_REPLAY] scored {len(evals)} episodes (gt available)", flush=True)
    finally:
        corpus.close()

    DIRECT = 100  # no-search direct correct = 1.00

    def mean_cents(cond) -> float:
        vals = [e["total_reward_c"] for e in evals if cond(e)]
        return float(np.mean(vals)) if vals else None

    # class memberships
    direct_correct = [e for e in evals if is_direct(e)]
    useful_correct = [e for e in evals if is_useful(e)]
    irrelevant_correct = [e for e in evals if is_irrelevant(e)]
    leak_correct = [e for e in evals if is_leak(e)]
    redundant_correct = [e for e in evals if e["n_searches"] >= 2 and e["em"]]
    invalid_class = [e for e in evals if is_invalid(e)]
    format_wrong = [e for e in evals if e["r_answer_total"] == 0.1 and not e["has_invalid"]]

    m_direct = mean_cents(is_direct)
    m_useful = mean_cents(is_useful)
    m_invalid = mean_cents(is_invalid)
    m_format = mean_cents(lambda e: e["r_answer_total"] == 0.1 and not e["has_invalid"])

    # --- hard gates ---
    gates = {}

    # 1. useful-search-correct > direct-correct (class means; the incentive)
    g1 = m_useful is not None and m_direct is not None and m_useful > m_direct
    gates["1_useful_search_correct_gt_direct_correct"] = {
        "pass": bool(g1), "mean_useful_cents": m_useful, "mean_direct_cents": m_direct,
        "n_useful": len(useful_correct), "n_direct": len(direct_correct),
    }

    # 2-4. per-episode hard caps <= 1.0 (stronger than class means; structurally
    # guaranteed by alpha=0 / leak-zeroing / redundant-credit-withholding)
    for name, members in (("2_irrelevant", irrelevant_correct), ("3_leak", leak_correct),
                          ("4_redundant", redundant_correct)):
        excess = [e["total_reward_c"] for e in members if e["total_reward_c"] > DIRECT]
        gates[f"{name}_search_correct_le_direct"] = {
            "pass": not excess, "n_members": len(members),
            "max_cents": max((e["total_reward_c"] for e in members), default=None),
            "n_excess": len(excess),
        }

    # 5. invalid < format-wrong < direct-correct
    g5 = (m_invalid is not None and m_format is not None and m_direct is not None
          and m_invalid < m_format < m_direct)
    gates["5_invalid_lt_format_wrong_lt_direct"] = {
        "pass": bool(g5), "mean_invalid_cents": m_invalid, "mean_format_cents": m_format,
        "mean_direct_cents": m_direct, "n_invalid": len(invalid_class), "n_format": len(format_wrong),
    }

    # --- Phase 4B.1 trainer-exact section ---
    tr = trainer_exact_grpo(evals)
    c = tr["class_trajectory_adv"]
    m_useful_adv = c["useful_search_correct"]["search_record_adv_mean"]
    m_direct_adv = c["direct_correct"]["traj_adv_mean"]
    m_irrelevant_adv = c["irrelevant_search_correct"]["traj_adv_mean"]
    m_leak_adv = c["answer_leak_correct"]["traj_adv_mean"]

    # 6. useful-search search records are NOT negative (the Phase 4B.1 fix)
    g6 = m_useful_adv is not None and m_useful_adv > 0
    gates["6_useful_search_records_positive_adv"] = {
        "pass": bool(g6), "useful_search_record_adv_mean": m_useful_adv,
        "n_useful": c["useful_search_correct"]["n"],
    }
    # 7. useful class trajectory adv > direct (T2 above T1)
    g7 = m_useful_adv is not None and m_direct_adv is not None and m_useful_adv > m_direct_adv
    gates["7_useful_gt_direct_trajectory_adv"] = {
        "pass": bool(g7), "useful_traj_adv_mean": m_useful_adv, "direct_traj_adv_mean": m_direct_adv,
    }
    # 8/9. T4/T7 not above T1 (irrelevant/leak-correct trajectories do not
    # rank above direct-correct IN THE SAME QUESTION GROUP). Matched in-group
    # comparison: for every uid group containing both classes, max trajectory
    # adv of the search class <= max trajectory adv of direct class. This is
    # the construction-test invariant (T1 == T4: identical return 1.00 ->
    # identical in-group adv; leak members are 0.80/0.35, strictly below).
    # Cross-group class means are NOT comparable: the direct class is diluted
    # by all-direct groups (adv ~ 0) while search classes only exist in groups
    # where returns sit above a lower group mean -- a composition artifact,
    # reported below for transparency only. Empty classes pass vacuously
    # (mirrors gates 2-4 in the Phase 4B run).
    def in_group_max_adviance_violations(pred_search) -> list[dict]:
        by_uid: dict[str, list[dict]] = defaultdict(list)
        for e in evals:
            by_uid[str(e["uid"])].append(e)
        violations = []
        for uid, es in by_uid.items():
            search_members = [e for e in es if pred_search(e)]
            direct_members = [e for e in es if is_direct(e)]
            if not search_members or not direct_members:
                continue
            max_search = max(e["trajectory_adv"] for e in search_members)
            max_direct = max(e["trajectory_adv"] for e in direct_members)
            if max_search > max_direct + 1e-9:
                violations.append({"uid": uid, "max_search_adv": max_search,
                                   "max_direct_adv": max_direct})
        return violations

    v8 = in_group_max_adviance_violations(is_irrelevant)
    v9 = in_group_max_adviance_violations(is_leak)
    g8 = not v8
    gates["8_irrelevant_not_above_direct_in_group"] = {
        "pass": bool(g8), "n_violations": len(v8), "samples": v8[:5],
        "irrelevant_traj_adv_mean": m_irrelevant_adv, "direct_traj_adv_mean": m_direct_adv,
        "note": "class means are cross-group composition artifacts; the gate is the matched in-group max comparison",
    }
    g9 = not v9
    gates["9_leak_not_above_direct_in_group"] = {
        "pass": bool(g9), "n_violations": len(v9), "samples": v9[:5],
        "leak_traj_adv_mean": m_leak_adv, "direct_traj_adv_mean": m_direct_adv,
    }
    # 10. data integrity: no traj_uid observed under two uids, neither during
    # loading nor across computed groups. Group size (exactly 5) is REPORTED,
    # not gated: coverage loss is expected where gt/question cannot be resolved
    # from the audit data, and the trainer-exact GRPO mirrors whatever group the
    # trainer actually saw.
    g10 = tr["n_duplicate_traj_uid_across_uids"] == 0 and len(uid_crossing) == 0
    gates["10_no_traj_uid_across_uids"] = {
        "pass": bool(g10),
        "n_uid_crossing_in_load": len(uid_crossing),
        "n_duplicate_traj_uid_across_uids": tr["n_duplicate_traj_uid_across_uids"],
        "group_size_distribution": tr["group_size_distribution"],
        "n_groups_not_exactly_5": tr["n_groups_not_exactly_5"],
    }
    # 11. token-rule null hit rate < real hit rate (matcher discriminates)
    null_stats = alias_null_hit_rates(evals)
    g11 = (null_stats["real_hit_rate"] is not None and null_stats["null_hit_rate_token_rule"] is not None
           and null_stats["null_hit_rate_token_rule"] < null_stats["real_hit_rate"])
    gates["11_null_hit_rate_lt_real_hit_rate"] = {
        "pass": bool(g11), **null_stats,
    }

    # --- report ---
    # three-way consistency: component sum == placed sum (asserted per episode
    # below) == trajectory return (same cents by construction)
    dup_bad = [e for e in evals if e["placed_c"] != e["total_reward_c"]]
    comps = defaultdict(int)
    triggers = defaultdict(int)
    for e in evals:
        for k in ("answer_reward_c", "format_reward_c", "evidence_hit_reward_c",
                  "searched_correct_bonus_c", "invalid_penalty_c", "redundant_penalty_c",
                  "answer_leak_penalty_c"):
            comps[k] += e["components"][k]
        triggers["episodes_with_evidence_credit"] += int(e["has_evidence_credit"])
        triggers["episodes_with_evidence_effective"] += int(e["has_evidence_effective"])
        triggers["episodes_with_leak"] += int(e["has_leak"])
        triggers["episodes_with_invalid"] += int(e["has_invalid"])
        triggers["episodes_with_redundant_search"] += int(e["n_searches"] >= 2)
        triggers["search_steps_total"] += e["n_searches"]
        triggers["em_correct"] += int(e["em"])
        triggers["format_only"] += int(e["r_answer_total"] == 0.1)

    report = {
        "config_fp_note": "historical audits record the OLD reward; record_scores not compared",
        "gates": gates,
        "trainer_exact_grpo": tr,
        "alias_null_hit": null_stats,
        "uid_crossing_violations": uid_crossing[:10],
        "component_totals_cents": dict(comps),
        "component_trigger_counts": dict(triggers),
        "duplicate_computation_check": {
            "n_episodes": len(evals),
            "n_sum_mismatch": len(dup_bad),
            "samples": [{"traj": e["traj_uid"], "placed": e["placed_c"], "total": e["total_reward_c"]} for e in dup_bad[:5]],
        },
        "class_sizes": {
            "direct_correct": len(direct_correct),
            "useful_search_correct": len(useful_correct),
            "irrelevant_search_correct": len(irrelevant_correct),
            "answer_leak_search_correct": len(leak_correct),
            "redundant_search_correct": len(redundant_correct),
            "invalid_no_answer": len(invalid_class),
            "format_wrong": len(format_wrong),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = OUT_PATH.with_suffix(".json.partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(OUT_PATH)

    print(f"[V1_REPLAY] report written: {OUT_PATH}")
    print(json.dumps({"gates": gates,
                      "trainer_exact": {k: tr[k] for k in ("group_size_distribution",
                                      "n_groups_not_exactly_5", "n_duplicate_traj_uid_across_uids",
                                      "class_trajectory_adv")},
                      "alias_null_hit": null_stats, "class_sizes": report["class_sizes"],
                      "component_trigger_counts": triggers, "component_totals_cents": dict(comps),
                      "dup_check": report["duplicate_computation_check"]},
                     ensure_ascii=False, indent=2))

    ok = all(g["pass"] for g in gates.values()) and len(dup_bad) == 0
    if not ok:
        print("[V1_REPLAY] HARD GATE FAILURE", file=sys.stderr)
        return 1
    print("[V1_REPLAY] all hard gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
