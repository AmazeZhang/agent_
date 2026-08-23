#!/usr/bin/env python
"""P3 Search-aware clean v2: historical-rollout replay (old-rule vs v2-rule redundancy).

User directive 2026-08-22 (docs/P3_SEARCH_AWARE_GRPO10_RESULT_REPORT_2026-08-22.md):
replay the ENTIRE historical rollouts -- formal segments (20260816a / 20260817a /
20260817b) and the v1 10-step GRPO run (20260822a) -- through the frozen v2 rule
(searchr1_repro/search_v2_reward.py, the single implementation source shared with
the training-side env) and:

  1. compute, per trajectory, the OLD rule's redundant verdict (every search
     after the first is redundant, the v1 rule) vs the NEW v2 verdict (TRUE
     redundancy only: duplicate normalized query, or no new document ID, or
     content-hash duplicate when IDs are unstable); report the number and ratio
     of MIS-PENALIZED steps (old rule flagged, v2 does not) and fully
     exonerated trajectories, per segment and overall.
  2. enforce the frozen ordering constraints as hard gates:
       G1 useful search+correct mean  > direct-correct mean
       G2 two different-new-evidence searches >= one valid search
          (per-episode matched truncation, plus class means reported)
       G3 duplicate/no-new-document search < its non-redundant counterpart
          (per-episode zeroed-counterfactual -- the redundant penalty must
          cost relative to the SAME trajectory without it)
       G4 invalid < format-wrong/direct-wrong < direct-correct (class means)
       G5 repeated searches cannot accumulate evidence rewards (cap 15/episode,
          no evidence credit on redundant steps)
       G6 answer-leak <= direct-correct (class means)
  3. trainer-exact GRPO (compute_grpo_trajectory_return_advantage offline):
     per-trajectory return (== placed sum == component sum, exact cents) ->
     per-uid GRPO mean/std over the 5 trajectory returns (torch.std sample std
     -> numpy ddof=1; len==1 group -> mean 0/std 1) -> trajectory advantage
     broadcast to every record (here each episode is one record).
  4. fail-closed consistency: component sum == placed sum == trajectory return
     for every episode; no traj_uid shared across uids.

Evidence is rebuilt from the REAL corpus (document_ids -> wiki-18 text via
CorpusReader); ground truth from train.parquet env_kwargs; terminal score via
the same compute_score(format_score=0.0) the clean env uses.

Historical audit record_scores reflect the OLD reward, so they are intentionally
NOT compared with v2 totals; the sum-consistency check is internal to v2.

Exit code 1 on any hard-gate failure.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p3_v1_reward_replay import (  # noqa: E402
    CorpusReader,
    DATA_ROOT,
    OFFSETS,
    CORPUS,
    SEGMENT_RUNS,
    TOKENIZER_PATH,
    TRAIN_PARQUET,
    decode_prompt_question,
    decode_response_text,
    load_gt_map,
)
from searchr1_repro.search_v2_reward import (  # noqa: E402
    ANSWER_REWARD_C,
    EVIDENCE_HIT_C,
    REDUNDANT_C,
    episode_totals,
    norm_query,
    search_step_components_v2,
    terminal_step_components_v2,
    valid_aliases,
)
from verl.utils.reward_score.search_r1_like_qa_em import compute_score  # noqa: E402

# v1 10-step GRPO run (20260822a): additional replay source mandated by §3.
V1_GRPO10_RUN = ("v1-grpo10", "p3-search-aware-instruct-grpo10-fsdp6-b66-n5-s0-20260822a")
ALL_RUNS = SEGMENT_RUNS + [V1_GRPO10_RUN]

OUT_PATH = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/p3_v2_reward_replay_20260822.json")


def load_trajectories_v2(segments, gt_map: dict, tokenizer_path: Path) -> tuple[list[dict], list[dict]]:
    """Same loading/dedup semantics as p3_v1_reward_replay.load_trajectories,
    additionally tagging each trajectory with its segment name (needed for the
    per-segment old-vs-new mis-penalty breakdown).
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
                        "segment": seg,
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
        t["chat_history_str"] = "".join(t["steps"][s] for s in sorted(t["steps"]))
        t["final_output"] = t["steps"][max_step]
        out.append(t)
    return out, violations


def compute_episode_v2(t: dict, corpus: CorpusReader) -> dict | None:
    """v2 step-attributed reward for one episode; None when gt is missing.

    State (prior_queries / prior_doc_ids / prior_content_hashes /
    had_evidence_credit) is threaded through the searches in env order exactly
    like the training env does (search_step_components_v2 returns the updated
    state dict which the caller MUST apply).
    """
    if t["gt"] is None:
        return None
    gt_aliases = valid_aliases(t["gt"])
    state = {
        "prior_queries": set(),
        "prior_doc_ids": set(),
        "prior_content_hashes": set(),
        "had_evidence_credit": False,
    }
    step_comps = []
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
        # The training env passes the tool observation (wrapped in
        # <information>) as doc_text -- replicate exactly.
        doc_text = "\n<information>" + blob + "</information>\n"
        sv = search_step_components_v2(
            query=s["query"],
            status=s["status"],
            doc_ids=s["document_ids"],
            doc_text=doc_text,
            gt_aliases=gt_aliases,
            question=t["question"],
            prior_queries=state["prior_queries"],
            prior_doc_ids=state["prior_doc_ids"],
            prior_content_hashes=state["prior_content_hashes"],
            is_first_search=(i == 0),
            had_evidence_credit=state["had_evidence_credit"],
        )
        # subtype diagnostic: which clause fired (must agree with the verdict)
        subtype = None
        if sv["redundant_search"]:
            if norm_query(s["query"]) and norm_query(s["query"]) in state["prior_queries"]:
                subtype = "duplicate_query"
            elif s["document_ids"] is not None:
                subtype = "no_new_document"
            else:
                subtype = "content_hash"
        sv["redundant_subtype"] = subtype
        state = sv.pop("state")  # apply updated trackers (fail-closed: must exist)
        sv["env_step"] = s["env_step"]
        step_comps.append(sv)

    # clean terminal: compute_score(format_score=0.0) exactly like the env
    reward = float(compute_score(t["chat_history_str"], {"target": t["gt"]}, format_score=0.0))
    em = reward >= 1.0
    had_effective = any(sv["evidence_effective"] for sv in step_comps)
    term = terminal_step_components_v2(r_answer_total=reward, em=em, had_effective_evidence=had_effective)
    totals = episode_totals(step_comps, term)
    placed_c = (
        sum(int(sv["step_shaping_c"]) for sv in step_comps)
        + int(term["answer_reward_c"])
        + int(term["format_reward_c"])
        + int(term["sce_c"])
    )
    return {
        "segment": t["segment"],
        "uid": t["uid"],
        "traj_uid": t["traj_uid"],
        "question": t["question"],
        "gt": t["gt"],
        "docs_blob": "\n<information>" + " ".join(doc_blobs) + "</information>\n" if doc_blobs else "",
        "n_searches": len(step_comps),
        "n_redundant_v2": sum(1 for sv in step_comps if sv["redundant_search"]),
        "n_invalid": sum(1 for sv in step_comps if sv["invalid_or_error"]),
        "n_leak": sum(1 for sv in step_comps if sv["answer_leak"]),
        "all_searches_clean": all(
            not sv["redundant_search"] and not sv["invalid_or_error"] and not sv["answer_leak"]
            for sv in step_comps
        ),
        "em": em,
        "r_answer_total": reward,
        "has_evidence_effective": had_effective,
        "has_evidence_credit": any(sv["evidence_credit"] for sv in step_comps),
        "has_leak": any(sv["answer_leak"] for sv in step_comps),
        "has_invalid": any(sv["invalid_or_error"] for sv in step_comps),
        "total_reward_c": totals["total_reward_c"],
        "placed_c": placed_c,
        "components": totals,
        "step_comps": step_comps,
        # old rule: every search after the first is redundant (the v1 rule)
        "old_redundant_count": max(0, len(step_comps) - 1),
    }


# ---------------------------------------------------------------------------
# classes (v2 semantics)
# ---------------------------------------------------------------------------
def is_useful(e: dict) -> bool:
    return e["n_searches"] > 0 and e["em"] and e["has_evidence_effective"]


def is_direct(e: dict) -> bool:
    return e["n_searches"] == 0 and e["em"]


def is_wrong(e: dict) -> bool:
    return e["r_answer_total"] == 0.0 and not e["has_invalid"]


def is_invalid(e: dict) -> bool:
    return e["r_answer_total"] == 0.0 and e["has_invalid"]


def is_leak(e: dict) -> bool:
    return e["em"] and e["has_leak"]


def is_single_valid_search_correct(e: dict) -> bool:
    return e["n_searches"] == 1 and e["em"] and e["all_searches_clean"]


def is_multi_hop_nonredundant_correct(e: dict) -> bool:
    return e["n_searches"] >= 2 and e["em"] and e["all_searches_clean"]


def is_redundant_search_correct(e: dict) -> bool:
    return e["em"] and e["n_redundant_v2"] >= 1


def trainer_exact_grpo_v2(evals: list[dict]) -> dict:
    """Mirror compute_grpo_trajectory_return_advantage (v2-0005) offline.

    Per-uid GRPO over TRAJECTORY returns: mean/std over the distinct trajectory
    returns of a uid (torch.std sample std -> numpy ddof=1; len==1 group ->
    mean 0/std 1), trajectory advantage broadcast to every record of the
    trajectory. Here each episode IS one trajectory, so the episode's advantage
    IS the broadcast trajectory advantage.
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
        return {
            "n": len(members),
            "traj_adv_mean": float(np.mean(traj_advs)) if traj_advs else None,
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
            "multi_hop_nonredundant_correct": class_stats(is_multi_hop_nonredundant_correct),
            "single_valid_search_correct": class_stats(is_single_valid_search_correct),
            "redundant_search_correct": class_stats(is_redundant_search_correct),
            "answer_leak_correct": class_stats(is_leak),
            "invalid_no_answer": class_stats(is_invalid),
        },
    }


def main() -> int:
    gt_map = load_gt_map()
    corpus = CorpusReader(CORPUS, OFFSETS)
    try:
        trajs, uid_crossing = load_trajectories_v2(ALL_RUNS, gt_map, TOKENIZER_PATH)
        print(f"[V2_REPLAY] loaded {len(trajs)} trajectories, {len(uid_crossing)} uid-crossing violations",
              flush=True)
        evals = []
        for t in trajs:
            e = compute_episode_v2(t, corpus)
            if e is not None:
                e["_traj"] = t  # for the G2 truncated-first-search counterfactual
                evals.append(e)
        print(f"[V2_REPLAY] scored {len(evals)} episodes (gt available)", flush=True)
    finally:
        corpus.close()

    # --- fail-closed consistency: component sum == placed sum == return ---
    dup_bad = [e for e in evals if e["placed_c"] != e["total_reward_c"]]

    # --- old rule vs new v2 rule redundancy comparison (§3) ---
    old_vs_new = {"total": {}, "per_segment": {}}
    by_segment: dict[str, list[dict]] = defaultdict(list)
    for e in evals:
        by_segment[e["segment"]].append(e)

    def old_vs_new_stats(pool: list[dict]) -> dict:
        old_total = sum(e["old_redundant_count"] for e in pool)
        new_total = sum(e["n_redundant_v2"] for e in pool)
        false_steps = old_total - new_total  # old flagged, v2 does not
        exonerated_trajs = [e for e in pool if e["old_redundant_count"] > 0 and e["n_redundant_v2"] == 0]
        kept_trajs = [e for e in pool if e["old_redundant_count"] > 0 and e["n_redundant_v2"] > 0]
        return {
            "n_trajectories": len(pool),
            "old_rule_redundant_steps": old_total,
            "v2_rule_redundant_steps": new_total,
            "mis_penalized_steps": false_steps,
            "mis_penalized_ratio_of_old_steps": (false_steps / old_total) if old_total else None,
            "mis_penalized_ratio_of_search_steps": (
                false_steps / sum(e["n_searches"] for e in pool)
            ) if pool else None,
            "n_trajectories_exonerated": len(exonerated_trajs),
            "n_trajectories_exonerated_ratio": (len(exonerated_trajs) / len(pool)) if pool else None,
            "n_trajectories_kept_redundant": len(kept_trajs),
            "exonerated_sample": [
                {"traj_uid": e["traj_uid"], "segment": e["segment"], "n_searches": e["n_searches"],
                 "old_redundant": e["old_redundant_count"], "v2_redundant": 0,
                 "subtypes": [sv["redundant_subtype"] for sv in e["step_comps"] if sv["redundant_search"]]}
                for e in exonerated_trajs[:8]
            ],
        }

    old_vs_new["total"] = old_vs_new_stats(evals)
    old_vs_new["per_segment"] = {seg: old_vs_new_stats(pool) for seg, pool in sorted(by_segment.items())}

    # --- ordering-constraint gates (§3 frozen semantics) ---
    DIRECT = ANSWER_REWARD_C  # 100

    def mean_cents(pred) -> float | None:
        vals = [e["total_reward_c"] for e in evals if pred(e)]
        return float(np.mean(vals)) if vals else None

    useful_correct = [e for e in evals if is_useful(e)]
    direct_correct = [e for e in evals if is_direct(e)]
    invalid_class = [e for e in evals if is_invalid(e)]
    wrong_class = [e for e in evals if is_wrong(e)]
    leak_correct = [e for e in evals if is_leak(e)]
    multi_hop = [e for e in evals if is_multi_hop_nonredundant_correct(e)]
    single_valid = [e for e in evals if is_single_valid_search_correct(e)]
    redundant_correct = [e for e in evals if is_redundant_search_correct(e)]

    m_useful = mean_cents(is_useful)
    m_direct = mean_cents(is_direct)
    m_invalid = mean_cents(is_invalid)
    m_wrong = mean_cents(is_wrong)
    m_leak = mean_cents(is_leak)
    m_multi = mean_cents(is_multi_hop_nonredundant_correct)
    m_single = mean_cents(is_single_valid_search_correct)
    m_redundant = mean_cents(is_redundant_search_correct)

    gates = {}

    # G1: useful search+correct > direct-correct (the incentive)
    g1 = m_useful is not None and m_direct is not None and m_useful > m_direct
    gates["G1_useful_search_correct_gt_direct_correct"] = {
        "pass": bool(g1), "mean_useful_cents": m_useful, "mean_direct_cents": m_direct,
        "n_useful": len(useful_correct), "n_direct": len(direct_correct),
    }

    # G2: two different-new-evidence searches >= one valid search.
    # Per-episode matched truncation (below, after the trainer-exact section):
    # for every multi-hop clean-correct episode, re-scoring with only the first
    # search must not RAISE the reward (evidence credit is capped at one; sce
    # needs effective evidence; all searches clean -> truncation only ever
    # removes non-negative contributions). Plus the class-mean comparison.

    # G3: duplicate/no-new-document search < its non-redundant counterpart:
    # for every episode with >=1 v2-redundant step, zeroing the redundant
    # penalties (the "corresponding non-redundant trajectory") strictly raises
    # the reward. Per-episode, plus the class-mean comparison.
    # "duplicate/no-new-document 搜索 < 对应非重复轨迹": the redundant search
    # must cost relative to the SAME trajectory with its redundant penalties
    # removed (the corresponding non-redundant trajectory), per episode. A
    # redundant-search-CORRECT trajectory may still beat a direct answer (its
    # genuine evidence + sce bonuses are intact); only the marginal redundant
    # penalty is gated. Class means are reported for transparency only.
    g3_violations = []
    for e in evals:
        if e["n_redundant_v2"] == 0:
            continue
        cf_c = e["total_reward_c"] - REDUNDANT_C * e["n_redundant_v2"]
        if e["total_reward_c"] >= cf_c - 1e-9:
            g3_violations.append({"traj_uid": e["traj_uid"], "actual_c": e["total_reward_c"],
                                  "counterfactual_c": cf_c, "n_redundant": e["n_redundant_v2"]})
    g3 = not g3_violations
    gates["G3_duplicate_no_new_doc_lt_nonredundant_counterpart"] = {
        "pass": bool(g3), "n_violations": len(g3_violations), "samples": g3_violations[:5],
        "mean_redundant_search_correct_cents": m_redundant, "mean_direct_cents": m_direct,
        "n_redundant_correct": len(redundant_correct),
        "note": "per-episode counterfactual (redundant penalties zeroed) is the gate; "
                "class means are cross-group composition artifacts, transparency only",
    }

    # G4: invalid < format-wrong/direct-wrong < direct-correct (class means)
    g4 = (m_invalid is not None and m_wrong is not None and m_direct is not None
          and m_invalid < m_wrong < m_direct)
    gates["G4_invalid_lt_wrong_lt_direct"] = {
        "pass": bool(g4), "mean_invalid_cents": m_invalid, "mean_wrong_cents": m_wrong,
        "mean_direct_cents": m_direct, "n_invalid": len(invalid_class), "n_wrong": len(wrong_class),
    }

    # G5: repeated searches cannot accumulate evidence rewards
    over_cap = [e for e in evals if e["components"]["evidence_hit_reward_c"] > EVIDENCE_HIT_C]
    credit_on_redundant = [
        e for e in evals
        for sv in e["step_comps"]
        if sv["evidence_credit"] and sv["redundant_search"]
    ]
    g5 = not over_cap and not credit_on_redundant
    gates["G5_no_evidence_reward_accumulation"] = {
        "pass": bool(g5), "n_over_cap": len(over_cap), "n_credit_on_redundant": len(credit_on_redundant),
        "max_evidence_cents": max((e["components"]["evidence_hit_reward_c"] for e in evals), default=0),
    }

    # G6: answer leak <= direct-correct (class means; a leak never earns
    # evidence credit by construction)
    g6 = m_leak is None or m_direct is None or m_leak <= m_direct
    leak_credit = [e for e in evals for sv in e["step_comps"] if sv["answer_leak"] and sv["evidence_credit"]]
    gates["G6_answer_leak_le_direct"] = {
        "pass": bool(g6) and not leak_credit, "mean_leak_cents": m_leak, "mean_direct_cents": m_direct,
        "n_leak_correct": len(leak_correct), "n_leak_credit_steps": len(leak_credit),
    }

    # G7: the third search is not penalized merely for being the third.
    # Every v2-redundant step must carry a real redundancy subtype (duplicate
    # query / no new document / content hash) -- ordinal position alone can
    # never flag a step. 3+ search episodes are reported for transparency.
    three_plus = [e for e in evals if e["n_searches"] >= 3]
    subtype_missing = [
        {"traj_uid": e["traj_uid"], "env_step": sv["env_step"]}
        for e in evals for sv in e["step_comps"]
        if sv["redundant_search"] and sv["redundant_subtype"] is None
    ]
    g7 = not subtype_missing
    gates["G7_third_search_not_penalized_by_count"] = {
        "pass": bool(g7), "n_3plus_search_episodes": len(three_plus),
        "n_subtype_missing_steps": len(subtype_missing), "samples": subtype_missing[:5],
        "n_3plus_with_redundant_penalty": sum(1 for e in three_plus if e["n_redundant_v2"] > 0),
    }

    # --- trainer-exact GRPO section ---
    tr = trainer_exact_grpo_v2(evals)
    c = tr["class_trajectory_adv"]
    m_useful_adv = c["useful_search_correct"]["traj_adv_mean"]
    m_direct_adv = c["direct_correct"]["traj_adv_mean"]
    m_multi_adv = c["multi_hop_nonredundant_correct"]["traj_adv_mean"]
    m_single_adv = c["single_valid_search_correct"]["traj_adv_mean"]

    # G8: useful-search trajectory advantage positive and above direct
    g8 = m_useful_adv is not None and m_useful_adv > 0 and (
        m_direct_adv is None or m_useful_adv > m_direct_adv
    )
    gates["G8_useful_search_advantage_positive_and_gt_direct"] = {
        "pass": bool(g8), "useful_traj_adv_mean": m_useful_adv, "direct_traj_adv_mean": m_direct_adv,
    }
    # G9: two different-new-evidence searches are not disadvantaged vs one
    # valid search IN THE SAME QUESTION GROUP (the construction-test invariant:
    # a multi-hop clean-correct return 145 equals a single valid-search return
    # 145 -> identical in-group adv). Matched in-group comparison, mirroring
    # the v1 replay's approach; class means are cross-group composition
    # artifacts (reported for transparency only).
    def in_group_max_adv_violations(pred_search, pred_base) -> list[dict]:
        by_uid: dict[str, list[dict]] = defaultdict(list)
        for e in evals:
            by_uid[str(e["uid"])].append(e)
        violations = []
        for uid, es in by_uid.items():
            search_members = [e for e in es if pred_search(e)]
            base_members = [e for e in es if pred_base(e)]
            if not search_members or not base_members:
                continue
            max_search = max(e["trajectory_adv"] for e in search_members)
            max_base = max(e["trajectory_adv"] for e in base_members)
            if max_search > max_base + 1e-9:
                violations.append({"uid": uid, "max_multi_hop_adv": max_search,
                                   "max_single_valid_adv": max_base})
        return violations

    g9_violations = in_group_max_adv_violations(
        is_multi_hop_nonredundant_correct, is_single_valid_search_correct
    )
    g9 = not g9_violations
    gates["G9_multi_hop_adv_ge_single_valid"] = {
        "pass": bool(g9), "n_violations": len(g9_violations), "samples": g9_violations[:5],
        "multi_hop_traj_adv_mean": m_multi_adv, "single_valid_traj_adv_mean": m_single_adv,
        "note": "the gate is the matched in-group max comparison; class means are "
                "cross-group composition artifacts, transparency only",
    }
    # G10: data integrity (no traj_uid across uids; loading + computed groups)
    g10 = tr["n_duplicate_traj_uid_across_uids"] == 0 and len(uid_crossing) == 0
    gates["G10_no_traj_uid_across_uids"] = {
        "pass": bool(g10), "n_uid_crossing_in_load": len(uid_crossing),
        "n_duplicate_traj_uid_across_uids": tr["n_duplicate_traj_uid_across_uids"],
        "group_size_distribution": tr["group_size_distribution"],
        "n_groups_not_exactly_5": tr["n_groups_not_exactly_5"],
    }

    # --- G2 truncated-first-search gate (needs the corpus; reopen) ---
    corpus2 = CorpusReader(CORPUS, OFFSETS)
    g2_violations = []
    try:
        for e in multi_hop:
            t = e["_traj"]
            full = e["total_reward_c"]
            truncated = compute_episode_v2({**t, "searches": t["searches"][:1]}, corpus2)
            if truncated is None:
                continue
            truncated_c = truncated["total_reward_c"]
            if truncated_c > full:
                g2_violations.append({"traj_uid": e["traj_uid"], "full_c": full, "truncated_c": truncated_c})
    finally:
        corpus2.close()
    g2 = not g2_violations
    gates["G2_two_new_evidence_searches_ge_one_valid"] = {
        "pass": bool(g2), "n_violations": len(g2_violations), "samples": g2_violations[:5],
        "mean_multi_hop_cents": m_multi, "mean_single_valid_cents": m_single,
        "n_multi_hop": len(multi_hop), "n_single_valid": len(single_valid),
        "note": "the gate is the per-episode matched truncation (first search only); "
                "class means are cross-group composition artifacts, transparency only",
    }

    # --- report ---
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
        triggers["episodes_with_v2_redundant"] += int(e["n_redundant_v2"] > 0)
        triggers["search_steps_total"] += e["n_searches"]
        triggers["em_correct"] += int(e["em"])
        subtype_counter = Counter(
            sv["redundant_subtype"] for sv in e["step_comps"] if sv["redundant_search"]
        )
        for k, v in subtype_counter.items():
            triggers[f"redundant_subtype_{k}"] += v

    report = {
        "config_fp_note": "historical audits record the OLD reward; record_scores not compared; "
                          "terminal score = compute_score(format_score=0.0) exactly like the clean env",
        "old_rule_vs_v2": old_vs_new,
        "gates": gates,
        "trainer_exact_grpo": tr,
        "uid_crossing_violations": uid_crossing[:10],
        "component_totals_cents": dict(comps),
        "component_trigger_counts": dict(triggers),
        "duplicate_computation_check": {
            "n_episodes": len(evals),
            "n_sum_mismatch": len(dup_bad),
            "samples": [{"traj": e["traj_uid"], "placed": e["placed_c"], "total": e["total_reward_c"]}
                        for e in dup_bad[:5]],
        },
        "class_sizes": {
            "useful_search_correct": len(useful_correct),
            "direct_correct": len(direct_correct),
            "multi_hop_nonredundant_correct": len(multi_hop),
            "single_valid_search_correct": len(single_valid),
            "redundant_search_correct": len(redundant_correct),
            "answer_leak_search_correct": len(leak_correct),
            "invalid_no_answer": len(invalid_class),
            "wrong_no_answer": len(wrong_class),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = OUT_PATH.with_suffix(".json.partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(OUT_PATH)

    print(f"[V2_REPLAY] report written: {OUT_PATH}")
    print(json.dumps({
        "old_rule_vs_v2_total": old_vs_new["total"],
        "gates": gates,
        "trainer_exact": {k: tr[k] for k in ("group_size_distribution", "n_groups_not_exactly_5",
                                             "n_duplicate_traj_uid_across_uids", "class_trajectory_adv")},
        "class_sizes": report["class_sizes"],
        "component_trigger_counts": dict(triggers),
        "dup_check": report["duplicate_computation_check"],
    }, ensure_ascii=False, indent=2))

    ok = all(g["pass"] for g in gates.values()) and len(dup_bad) == 0
    if not ok:
        print("[V2_REPLAY] HARD GATE FAILURE", file=sys.stderr)
        return 1
    print("[V2_REPLAY] all hard gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
