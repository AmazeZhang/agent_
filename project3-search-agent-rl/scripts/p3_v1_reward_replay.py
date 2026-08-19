#!/usr/bin/env python
"""P3 Phase 4B: historical-rollout replay of the frozen Search-aware GRPO v1 reward.

Replays the ENTIRE historical formal training rollouts (segments 0-50 / 50-100 /
100-300) through the single implementation source
(searchr1_repro/search_v1_reward.py, docs .../P3_PHASE4_SEARCH_CAUSAL_DIAG_RESULT_2026-08-19.md §9)
and enforces the 5 hard replay gates:

  1. useful-search-correct mean  > direct-correct mean
  2. irrelevant-search-correct  <= direct-correct  (per-episode, max <= 1.0)
  3. answer-leak-search-correct <= direct-correct  (per-episode, max <= 1.0)
  4. redundant-search-correct   <= direct-correct  (per-episode, max <= 1.0)
  5. invalid < format-wrong < direct-correct (class means)

Evidence is rebuilt from the REAL corpus (document_ids -> wiki-18 text via
CorpusReader), ground truth from train.parquet env_kwargs, terminal score via
the same compute_score(format_score=0.1) the env uses. Reported per
authorization: intra-group reward variance, all-same-reward group ratio, search
action advantage direction, per-component trigger counts, and a duplicate
computation check (per-episode sum of placed step scores == component sum).

Historical audit record_scores reflect the OLD reward, so they are intentionally
NOT compared with v1 totals; the sum-consistency check is internal to v1.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchr1_repro.search_v1_reward import (  # noqa: E402
    episode_totals,
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


def load_trajectories(segments, gt_map: dict, tokenizer_path: Path) -> list[dict]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    trajs: dict[str, dict] = {}
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
    return out


def compute_episode_v1(t: dict, corpus: CorpusReader) -> dict | None:
    """v1 step-attributed reward for one episode; None when gt is missing."""
    if t["gt"] is None:
        return None
    gt_aliases = valid_aliases(t["gt"])
    step_comps = []
    had_effective = False
    for i, s in enumerate(t["searches"]):
        docs = []
        for did in s["document_ids"]:
            try:
                docs.append(corpus.get(int(did)))
            except (ValueError, IndexError):
                pass
        doc_text = "\n<information>" + " ".join(d.get("contents", "") for d in docs) + "</information>\n"
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
        "question": t["question"],
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


def group_stats(evals: list[dict]) -> dict:
    by_uid: dict[str, list[float]] = defaultdict(list)
    for e in evals:
        by_uid[e["uid"]].append(e["total_reward_c"] / 100.0)
    group_rewards = list(by_uid.values())
    intra_var = float(np.mean([float(np.var(g)) for g in group_rewards])) if group_rewards else 0.0
    all_same = sum(1 for g in group_rewards if max(g) - min(g) < 1e-9) / len(group_rewards) if group_rewards else 0.0
    adv = np.zeros(len(evals))
    for i, e in enumerate(evals):
        g = np.array(by_uid[e["uid"]], dtype=float)
        m = g.mean()
        s = g.std() if len(g) > 1 else 1.0
        adv[i] = (e["total_reward_c"] / 100.0 - m) / (s + 1e-6) if s > 1e-9 else 0.0
    searched = [a for a, e in zip(adv, evals) if e["n_searches"] > 0]
    nosearch = [a for a, e in zip(adv, evals) if e["n_searches"] == 0]
    return {
        "n_trajs": len(evals),
        "n_groups": len(group_rewards),
        "reward_mean": float(np.mean([e["total_reward_c"] / 100.0 for e in evals])),
        "reward_std": float(np.std([e["total_reward_c"] / 100.0 for e in evals])),
        "intra_group_variance_mean": intra_var,
        "all_same_reward_group_ratio": all_same,
        "search_frac": sum(e["n_searches"] > 0 for e in evals) / len(evals),
        "search_adv_mean": float(np.mean(searched)) if searched else None,
        "nosearch_adv_mean": float(np.mean(nosearch)) if nosearch else None,
        "n_search_trajs": len(searched),
        "n_nosearch_trajs": len(nosearch),
    }


def main() -> int:
    gt_map = load_gt_map()
    corpus = CorpusReader(CORPUS, OFFSETS)
    try:
        trajs = load_trajectories(SEGMENT_RUNS, gt_map, TOKENIZER_PATH)
        print(f"[V1_REPLAY] loaded {len(trajs)} trajectories", flush=True)
        evals = [e for t in trajs if (e := compute_episode_v1(t, corpus)) is not None]
        print(f"[V1_REPLAY] scored {len(evals)} episodes (gt available)", flush=True)
    finally:
        corpus.close()

    DIRECT = 100  # no-search direct correct = 1.00

    def mean_cents(cond) -> float:
        vals = [e["total_reward_c"] for e in evals if cond(e)]
        return float(np.mean(vals)) if vals else None

    # class memberships
    direct_correct = [e for e in evals if e["n_searches"] == 0 and e["em"]]
    useful_correct = [e for e in evals if e["n_searches"] > 0 and e["em"] and e["has_evidence_effective"]]
    irrelevant_correct = [e for e in evals if e["n_searches"] > 0 and e["em"] and not e["has_evidence_effective"]]
    leak_correct = [e for e in evals if e["em"] and e["has_leak"]]
    redundant_correct = [e for e in evals if e["n_searches"] >= 2 and e["em"]]
    invalid_class = [e for e in evals if e["r_answer_total"] == 0.0 and e["has_invalid"]]
    format_wrong = [e for e in evals if e["r_answer_total"] == 0.1 and not e["has_invalid"]]

    m_direct = mean_cents(lambda e: e["n_searches"] == 0 and e["em"])
    m_useful = mean_cents(lambda e: e["n_searches"] > 0 and e["em"] and e["has_evidence_effective"])
    m_invalid = mean_cents(lambda e: e["r_answer_total"] == 0.0 and e["has_invalid"])
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

    # --- report ---
    # duplicate computation check: every episode's placed step sum == component sum
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
        "group_stats": group_stats(evals),
        "component_totals_cents": dict(comps),
        "component_trigger_counts": dict(triggers),
        "duplicate_computation_check": {
            "n_episodes": len(evals),
            "n_sum_mismatch": len(dup_bad),
            "samples": [{"traj": e["uid"], "placed": e["placed_c"], "total": e["total_reward_c"]} for e in dup_bad[:5]],
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
    print(json.dumps({"gates": gates, "class_sizes": report["class_sizes"],
                      "group_stats": report["group_stats"],
                      "component_trigger_counts": triggers,
                      "component_totals_cents": dict(comps),
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
