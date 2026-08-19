#!/usr/bin/env python3
"""Phase 4A diagnostic 4: auditable reward simulator (pure CPU, no training).

Formula (prereg section 5):
    R = R_answer + a*valid_retrieval + b*evidence_hit + g*searched_and_correct_and_evidence_hit
        - l*invalid_or_error - m*redundant_search_count

R_answer uses the current semantics with a candidate format_score in {0.1, 0.05, 0.0}:
    EM correct (score>=0.5) -> 1.0 ; well-formed <answer> but wrong -> format_score ; else 0.0.

Part 1: anti-reward-hacking suite (8 fixed trajectories). Hard preregistered
        ordering (must hold for every candidate):
            T2 (search+evidence+correct) > T1 (no-search direct correct) > T3 (format, wrong)
            > T5 (invalid search)
        plus guards: T6 (redundant search spam, correct) <= T1 ; alpha <= 0.05
        (calling search alone must never be farmable).

Part 2: historical evaluation over formal training rollouts (group_n=5):
        runs/p3-formal-segment-*-*/rollouts/*.audit.jsonl (search semantics)
        aligned row-for-row with *.jsonl (prompt/output text). Ground truth joined
        from the training parquet by exact question; evidence_hit decided by
        reading the retrieved document_ids from the wiki-18 corpus by offset
        (no re-query, no faiss).

Per candidate coefficient set: reward distribution, intra-group variance,
all-same-reward group ratio, and GRPO group-normalized advantage direction
(search vs no-search trajectories). Only coefficient RANGES are recommended;
nothing is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verl.utils.reward_score.search_r1_like_qa_em import compute_score  # noqa: E402

DATA_ROOT = Path("/media/imc/data/project3-search-agent-rl")
TRAIN_PARQUET = DATA_ROOT / "datasets/searchr1-upstream/train.parquet"
CORPUS = DATA_ROOT / "indexes/searchr1-wiki18-e5/prepared/wiki-18.jsonl"
OFFSETS = DATA_ROOT / "indexes/searchr1-wiki18-e5/prepared/wiki-18.offsets.npy"
SEGMENT_RUNS = [
    ("seg0-50", "p3-formal-segment-0-50-fsdp6-b66-n5-s0-20260816a"),
    ("seg50-100", "p3-formal-segment-50-100-fsdp6-b66-n5-s0-20260817a"),
]
TOKENIZER_PATH = DATA_ROOT / "models/Qwen2.5-3B"
OUT_PATH = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/p3_diag4_reward_sim_20260819.json")

INVALID_STATUSES = {"invalid_query", "api_error", "no_results", "processing_error", "tool_exception"}


def norm_text(s: str) -> str:
    import re
    import unicodedata

    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return re.sub(r"[\s\W_]+", "", s)


def r_answer(score: float, format_score: float) -> float:
    if score >= 0.5:
        return 1.0
    return format_score


def sim_reward(
    r_ans: float,
    valid_retrieval: float,
    evidence_hit: float,
    s_and_c_and_e: float,
    invalid_or_error: float,
    redundant: float,
    a: float,
    b: float,
    g: float,
    l: float,
    m: float,
) -> float:
    return (
        r_ans
        + a * valid_retrieval
        + b * evidence_hit
        + g * s_and_c_and_e
        - l * invalid_or_error
        - m * redundant
    )


# --------------------------------------------------------------------------- #
# Part 1: anti-reward-hacking suite
# --------------------------------------------------------------------------- #

# Each trajectory: (name, r_ans, valid, evidence, s_and_c_and_e, invalid, redundant)
ANTI_HACK = [
    ("T1 no-search direct correct", 1.0, 0, 0, 0, 0, 0),
    ("T2 valid search + evidence + correct", 1.0, 1, 1, 1, 0, 0),
    ("T3 search relevant evidence but wrong (format)", 0.1, 1, 1, 0, 0, 0),
    ("T4 search irrelevant docs, correct from memory", 1.0, 1, 0, 0, 0, 0),
    ("T5 invalid query, no answer", 0.0, 0, 0, 0, 1, 0),
    ("T6 redundant search spam x2, evidence, correct", 1.0, 1, 1, 1, 0, 1),
    ("T7 ground-truth answer leaked into query, correct", 1.0, 1, 1, 1, 0, 0),
    ("T8 format-only correct, wrong answer, no search", 0.1, 0, 0, 0, 0, 0),
]

# Candidate coefficient sets: format, alpha, beta, gamma, lambda, mu.
# mu >= alpha+beta+gamma enforced so redundant spam never outranks direct memory
# answers; alpha <= 0.05 so "just calling search" adds at most a small nudge.
CANDIDATES = [
    {"name": "C0-current-baseline", "format": 0.1, "a": 0.0, "b": 0.0, "g": 0.0, "l": 0.0, "m": 0.0},
    {"name": "C1-conservative", "format": 0.1, "a": 0.02, "b": 0.05, "g": 0.10, "l": 0.10, "m": 0.20},
    {"name": "C2-moderate", "format": 0.1, "a": 0.02, "b": 0.10, "g": 0.20, "l": 0.20, "m": 0.35},
    {"name": "C3-format-0.05", "format": 0.05, "a": 0.02, "b": 0.05, "g": 0.10, "l": 0.10, "m": 0.20},
    {"name": "C4-format-0.0", "format": 0.0, "a": 0.02, "b": 0.05, "g": 0.10, "l": 0.10, "m": 0.20},
    {"name": "C5-evidence-driven-a0", "format": 0.1, "a": 0.0, "b": 0.15, "g": 0.30, "l": 0.20, "m": 0.45},
    {"name": "C6-spam-averse", "format": 0.1, "a": 0.05, "b": 0.10, "g": 0.20, "l": 0.30, "m": 0.40},
]


def anti_hack_report(c: dict) -> dict:
    f = c["format"]
    rewards = {}
    for name, r_ans, v, e, sce, inv, red in ANTI_HACK:
        r_ans_eff = r_ans if r_ans < 1.0 else 1.0
        if name.startswith("T3") or name.startswith("T8"):
            r_ans_eff = f  # format-score variants
        rewards[name] = sim_reward(r_ans_eff, v, e, sce, inv, red, c["a"], c["b"], c["g"], c["l"], c["m"])
    t1, t2, t3, t4, t5, t6, t7, t8 = (rewards[n] for n in ("T1 no-search direct correct", "T2 valid search + evidence + correct",
                                                           "T3 search relevant evidence but wrong (format)", "T4 search irrelevant docs, correct from memory",
                                                           "T5 invalid query, no answer", "T6 redundant search spam x2, evidence, correct",
                                                           "T7 ground-truth answer leaked into query, correct", "T8 format-only correct, wrong answer, no search"))
    checks = {
        "T2 > T1": t2 > t1,
        "T1 > T3": t1 > t3,
        "T3 > T5": t3 > t5,
        "T6 <= T1 (spam guard)": t6 <= t1 + 1e-9,
        "alpha <= 0.05 (search-call cap)": c["a"] <= 0.05,
        "T8 (format farming) < T1": t8 < t1,
        "mu >= a+b+g": c["m"] >= c["a"] + c["b"] + c["g"] - 1e-9,
    }
    return {
        "candidate": c["name"],
        "rewards": {n: round(v, 4) for n, v in rewards.items()},
        "checks": checks,
        "pass": all(checks.values()),
        "note_T7": "T7 == T2 by construction: answer-in-query is undetectable from reward alone; mitigation lives in a query-validation layer (count it in historical data instead)",
    }


# --------------------------------------------------------------------------- #
# Corpus reader by document id (offset seek; no faiss import)
# --------------------------------------------------------------------------- #

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


def alias_hit_in_docs(targets: list[str], docs: list[dict]) -> bool:
    ntargets = [norm_text(t) for t in targets if t]
    if not ntargets:
        return False
    nblob = norm_text(" ".join(d.get("contents", "") for d in docs))
    return any(t and t in nblob for t in ntargets)


# --------------------------------------------------------------------------- #
# Historical trajectory loading (audit jsonl only; prompt/response decoded from
# input_ids). Prompt template (all segments): system message + user content
# "…Your question: {SEARCH_PROMPT_PREFIX}{QUESTION}\n\nNow it's your turn to
# respond for the current step.\n…". Rows padded/duplicated by adjust_batch are
# deduped by (traj_uid, env_step).
# --------------------------------------------------------------------------- #

from agent_system.environments.env_package.search.envs import SEARCH_PROMPT_PREFIX as SEARCH_PREFIX  # noqa: E402

Q_MARKER = "Now answer the following question:\n"
Q_SUFFIX = "\n\nNow it's your turn to respond"


def decode_prompt_question(tok, ids, att, prompt_width: int) -> str | None:
    vp = [i for i in range(prompt_width) if att[i]]
    if not vp:
        return None
    text = tok.decode(ids[vp[0] : vp[-1] + 1], skip_special_tokens=False)
    if Q_MARKER not in text:
        return None
    q = text.split(Q_MARKER, 1)[1].split(Q_SUFFIX, 1)[0]
    return q.strip()


def decode_response_text(tok, ids, att, prompt_width: int, response_width: int) -> str:
    vr = [i for i in range(prompt_width, prompt_width + response_width) if att[i]]
    if not vr:
        return ""
    return tok.decode(ids[vr[0] : vr[-1] + 1], skip_special_tokens=False)


def load_trajectories(segments, corpus: CorpusReader, gt_map: dict, tokenizer_path: Path) -> list[dict]:
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
                        "steps": {},  # env_step -> output text
                        "searches": [],
                        "searches_by_step": {},
                        "invalid_or_error": False,
                        "retrieval_failed": False,
                    }
                    trajs[traj_uid] = t
                step = m["env_step"]
                if step not in t["steps"]:  # dedupe adjust_batch duplicate rows
                    t["steps"][step] = decode_response_text(tok, a["input_ids"], a["attention_mask"],
                                                            a["prompt_width"], a["response_width"])
                ret = m.get("retrieval")
                if ret is not None and step not in t["searches_by_step"]:
                    t["searches_by_step"][step] = True
                    t["searches"].append(
                        {
                            "query": ret.get("query"),
                            "status": ret.get("status"),
                            "document_ids": ret.get("document_ids") or [],
                            "total_results": ret.get("total_results"),
                        }
                    )
                    if ret.get("status") in INVALID_STATUSES:
                        t["invalid_or_error"] = True
                if m.get("retrieval_failed"):
                    t["retrieval_failed"] = True
    out = []
    for t in trajs.values():
        max_step = max(t["steps"])
        t["final_output"] = t["steps"][max_step]
        out.append(t)
    return out


def load_gt_map() -> dict:
    df = pd.read_parquet(TRAIN_PARQUET)
    m = {}
    for _, r in df.iterrows():
        ek = r["env_kwargs"]
        q = str(ek["question"])
        if q not in m:
            m[q] = list(ek["ground_truth"]["target"])
    return m


def evaluate_trajectory(t: dict, c: dict, corpus: CorpusReader) -> dict | None:
    """Terms per prereg section 5; None when ground truth is missing."""
    if t["gt"] is None:
        return None
    score = float(compute_score(t["final_output"], {"target": t["gt"]}, method="strict", format_score=0.0, score=1.0))
    r_ans = r_answer(score, c["format"])
    searched = len(t["searches"]) > 0
    valid = any(s["status"] == "success" for s in t["searches"])
    evidence = False
    for s in t["searches"]:
        docs = [corpus.get(int(d)) for d in s["document_ids"]]
        if alias_hit_in_docs(t["gt"], docs):
            evidence = True
            break
    correct = score >= 0.5
    sce = 1.0 if (searched and evidence and correct) else 0.0
    invalid = 1.0 if (t["invalid_or_error"] or t["retrieval_failed"]) else 0.0
    redundant = max(0, len(t["searches"]) - 1)
    r = sim_reward(r_ans, 1.0 if valid else 0.0, 1.0 if evidence else 0.0, sce, invalid, redundant,
                   c["a"], c["b"], c["g"], c["l"], c["m"])
    return {
        "uid": t["uid"],
        "question": t["question"],
        "searched": searched,
        "valid_retrieval": valid,
        "evidence_hit": evidence,
        "correct": correct,
        "r_answer": r_ans,
        "r": r,
        "invalid_or_error": invalid,
        "redundant": redundant,
    }


def group_stats(evals: list[dict], c: dict) -> dict:
    by_uid: dict[str, list[float]] = defaultdict(list)
    for e in evals:
        by_uid[e["uid"]].append(e["r"])
    group_rewards = list(by_uid.values())
    intra_var = float(np.mean([float(np.var(g)) for g in group_rewards]))
    all_same = sum(1 for g in group_rewards if max(g) - min(g) < 1e-9) / len(group_rewards)
    # GRPO-style group-normalized advantage: (r - mean)/std per uid group
    adv = np.zeros(len(evals))
    for i, e in enumerate(evals):
        g = np.array(by_uid[e["uid"]], dtype=float)
        m = g.mean()
        s = g.std() if len(g) > 1 else 1.0
        adv[i] = (e["r"] - m) / (s + 1e-6) if s > 1e-9 else 0.0
    search_adv = [adv[i] for i, e in enumerate(evals) if e["searched"]]
    nosearch_adv = [adv[i] for i, e in enumerate(evals) if not e["searched"]]
    return {
        "n_trajs": len(evals),
        "n_groups": len(group_rewards),
        "reward_mean": float(np.mean([e["r"] for e in evals])),
        "reward_std": float(np.std([e["r"] for e in evals])),
        "intra_group_variance_mean": intra_var,
        "all_same_reward_group_ratio": all_same,
        "search_frac": sum(e["searched"] for e in evals) / len(evals),
        "search_adv_mean": float(np.mean(search_adv)) if search_adv else None,
        "nosearch_adv_mean": float(np.mean(nosearch_adv)) if nosearch_adv else None,
        "n_search_trajs": len(search_adv),
        "n_nosearch_trajs": len(nosearch_adv),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="seg0-50,seg50-100")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    selected = [s for s in SEGMENT_RUNS if s[0] in args.segments.split(",")]

    # Part 1: anti-hack suite (C0 is the no-shaping reference: T2 == T1 by
    # construction, so its strict ordering check is informational only).
    suite = [anti_hack_report(c) for c in CANDIDATES]
    print("=== Part 1: anti-reward-hacking suite ===")
    for s in suite:
        print(f"  {s['candidate']:<24} pass={s['pass']}  " + " ".join(f"{k}:{v}" for k, v in s["checks"].items()))
    shaping = [s for s in suite if s["candidate"] != "C0-current-baseline"]
    assert all(s["pass"] for s in shaping), "anti-hack ordering violated for a shaping candidate"
    assert not suite[0]["checks"]["T2 > T1"], "baseline reference must have T2 == T1 (no shaping)"

    # Part 2: historical evaluation
    print("\n=== Part 2: historical rollouts (group_n=5) ===")
    gt_map = load_gt_map()
    corpus = CorpusReader(CORPUS, OFFSETS)
    trajs = load_trajectories(selected, corpus, gt_map, TOKENIZER_PATH)
    print(f"  trajectories loaded: {len(trajs)} (gt joined: {sum(1 for t in trajs if t['gt'] is not None)})")
    with_gt = [t for t in trajs if t["gt"] is not None]
    print(f"  searched: {sum(1 for t in with_gt if t['searches'])}  "
          f"search steps: {sum(len(t['searches']) for t in with_gt)}  "
          f"invalid/error trajs: {sum(1 for t in with_gt if t['invalid_or_error'] or t['retrieval_failed'])}")

    results = {}
    for c in CANDIDATES:
        evals = [e for e in (evaluate_trajectory(t, c, corpus) for t in with_gt) if e is not None]
        gs = group_stats(evals, c)
        results[c["name"]] = {
            "coefs": {"format": c["format"], "a": c["a"], "b": c["b"], "g": c["g"], "l": c["l"], "m": c["m"]},
            **gs,
        }
        print(f"  {c['name']:<24} n={gs['n_trajs']} mean={gs['reward_mean']:.4f} "
              f"intra_var={gs['intra_group_variance_mean']:.4f} all_same={gs['all_same_reward_group_ratio']:.3f} "
              f"search_adv={gs['search_adv_mean']} nosearch_adv={gs['nosearch_adv_mean']}")

    # historical search quality: how often shaping terms can fire at all
    quals = {"valid": 0, "evidence": 0, "correct": 0, "valid_and_evidence": 0, "correct_with_evidence_search": 0,
             "searched_total": 0, "search_steps_total": 0, "redundant_sum": 0}
    for t in with_gt:
        if not t["searches"]:
            continue
        quals["searched_total"] += 1
        quals["search_steps_total"] += len(t["searches"])
        quals["redundant_sum"] += max(0, len(t["searches"]) - 1)
        ev = evaluate_trajectory(t, CANDIDATES[0], corpus)  # terms are coeff-independent
        if ev is None:
            continue
        if ev["valid_retrieval"]:
            quals["valid"] += 1
        if ev["evidence_hit"]:
            quals["evidence"] += 1
        if ev["valid_retrieval"] and ev["evidence_hit"]:
            quals["valid_and_evidence"] += 1
        if ev["correct"]:
            quals["correct"] += 1
        if ev["correct"] and ev["evidence_hit"] and ev["valid_retrieval"]:
            quals["correct_with_evidence_search"] += 1
    corpus.close()

    out = {
        "prereg_ref": "docs/P3_PHASE4_SEARCH_CAUSAL_DIAG_PREREG_2026-08-19.md section 5",
        "anti_hack_suite": suite,
        "all_shaping_candidates_pass": all(s["pass"] for s in suite if s["candidate"] != "C0-current-baseline"),
        "baseline_reference_info": suite[0],
        "candidates": results,
        "historical_search_quality": quals,
        "note": "coefficient RANGES only, not frozen; see result doc. Advantage direction is a property of historical (off-policy) trajectories, not a prediction for a policy trained under the candidate reward.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
