#!/usr/bin/env python3
"""Minimal CPU-only funnel diagnosis for the final seed2026 Clean/Aware pair.

This script does not call the Retriever or a model. It reuses the frozen v2
evidence matcher over the document bodies already stored in episodes.jsonl and
partitions each question into:

    no search | searched without evidence hit | searched with evidence hit
        x correct | answered wrong | unanswered

The output is intentionally small: it answers whether the next bottleneck is
retrieval/query recall, evidence-to-answer conversion, or answer submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from searchr1_repro.search_v2_reward import evidence_hit_in_docs, valid_aliases


DEFAULT_CLEAN = "p3-eval-clean-grpo10-seed2026-gs10-confirm256-20260824a"
DEFAULT_AWARE = "p3-eval-aware-v2-seed2026-gs10-confirm256-20260824a"
ENV_MAX_STEPS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_search(step: dict[str, Any]) -> bool:
    return step.get("info", {}).get("tool_name") == "search"


def is_answer(step: dict[str, Any]) -> bool:
    return str(step.get("info", {}).get("postprocessed_action", "")).startswith("<answer>")


def analyze_episode(episode: dict[str, Any]) -> dict[str, Any]:
    aliases = valid_aliases(episode["answers"])
    searches = [step for step in episode["steps"] if is_search(step)]
    valid_searches = [
        step
        for step in searches
        if (step.get("info", {}).get("retrieval") or {}).get("status")
        in {"success", "no_results"}
    ]
    hit_steps = [
        step
        for step in valid_searches
        if evidence_hit_in_docs(step.get("observation") or "", aliases)
    ]
    answered_env = any(is_answer(step) for step in episode["steps"])
    answered_offline = bool(episode.get("offline", {}).get("has_answer"))
    correct = bool(episode.get("reward", 0) >= 1.0)
    if not valid_searches:
        stage = "no_search"
    elif hit_steps:
        stage = "searched_with_evidence"
    else:
        stage = "searched_without_evidence"
    if correct:
        outcome = "correct"
    elif answered_offline:
        outcome = "answered_wrong"
    else:
        outcome = "unanswered"
    return {
        "qid": int(episode["question_id"]),
        "source": episode["source"],
        "stage": stage,
        "outcome": outcome,
        "correct": correct,
        "answered_offline": answered_offline,
        "answered_env": answered_env,
        "evidence_hit": bool(hit_steps),
        "search_calls": len(valid_searches),
        "evidence_hit_calls": len(hit_steps),
        "max_steps_exhausted": not answered_env and len(episode["steps"]) >= ENV_MAX_STEPS,
    }


def load_run(runs_root: Path, run_id: str) -> tuple[dict[str, Any], dict[int, dict[str, Any]], str]:
    run_dir = runs_root / run_id
    episode_path = run_dir / "episodes.jsonl"
    result = json.loads((run_dir / "results.json").read_text())
    rows = [json.loads(line) for line in episode_path.open()]
    analyzed = [analyze_episode(row) for row in rows]
    by_qid = {row["qid"]: row for row in analyzed}
    assert len(rows) == len(by_qid) == 256
    assert set(by_qid) == set(range(256))
    assert sum(row["correct"] for row in analyzed) == result["metrics"]["overall"]["em"]
    return result, by_qid, sha256(episode_path)


def summarize(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    stages = ("no_search", "searched_without_evidence", "searched_with_evidence")
    outcomes = ("correct", "answered_wrong", "unanswered")
    matrix = {
        stage: {
            outcome: sum(row["stage"] == stage and row["outcome"] == outcome for row in rows.values())
            for outcome in outcomes
        }
        for stage in stages
    }
    searched = [row for row in rows.values() if row["stage"] != "no_search"]
    hit = [row for row in searched if row["evidence_hit"]]
    no_hit = [row for row in searched if not row["evidence_hit"]]
    incorrect = [row for row in rows.values() if not row["correct"]]
    return {
        "matrix": matrix,
        "searched_questions": len(searched),
        "evidence_hit_questions": len(hit),
        "question_level_evidence_hit_rate_given_search": len(hit) / max(len(searched), 1),
        "hit_to_answer": sum(row["answered_offline"] for row in hit) / max(len(hit), 1),
        "hit_to_correct": sum(row["correct"] for row in hit) / max(len(hit), 1),
        "no_hit_to_correct": sum(row["correct"] for row in no_hit) / max(len(no_hit), 1),
        "incorrect_partition": {
            "no_search": sum(row["stage"] == "no_search" for row in incorrect),
            "searched_without_evidence": sum(
                row["stage"] == "searched_without_evidence" for row in incorrect
            ),
            "searched_with_evidence": sum(
                row["stage"] == "searched_with_evidence" for row in incorrect
            ),
        },
        "max_steps_exhausted": sum(row["max_steps_exhausted"] for row in rows.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/media/imc/data")
    parser.add_argument("--clean-run", default=DEFAULT_CLEAN)
    parser.add_argument("--aware-run", default=DEFAULT_AWARE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs_root = Path(args.data_root) / "project3-search-agent-rl" / "runs"
    clean_result, clean, clean_sha = load_run(runs_root, args.clean_run)
    aware_result, aware, aware_sha = load_run(runs_root, args.aware_run)
    assert clean_result["data_files"]["sha256"] == aware_result["data_files"]["sha256"]
    for key in ("temperature", "seed", "max_steps", "history_length", "topk", "retrieval_condition"):
        assert clean_result["parameters"][key] == aware_result["parameters"][key]

    clean_summary = summarize(clean)
    aware_summary = summarize(aware)
    added_search = [qid for qid in clean if clean[qid]["stage"] == "no_search" and aware[qid]["stage"] != "no_search"]
    paired_added_search = {
        "questions": len(added_search),
        "aware_evidence_hit": sum(aware[qid]["evidence_hit"] for qid in added_search),
        "aware_correct": sum(aware[qid]["correct"] for qid in added_search),
        "clean_correct": sum(clean[qid]["correct"] for qid in added_search),
        "gained": sum(not clean[qid]["correct"] and aware[qid]["correct"] for qid in added_search),
        "lost": sum(clean[qid]["correct"] and not aware[qid]["correct"] for qid in added_search),
    }

    gaps = aware_summary["incorrect_partition"]
    largest_gap = max(gaps, key=gaps.get)
    recommendation = {
        "searched_without_evidence": "Prioritize query quality / retrieval recall with evaluation-only top-k or reranking checks.",
        "searched_with_evidence": "Prioritize evidence-to-answer extraction/synthesis; do not replace the Retriever first.",
        "no_search": "Prioritize remaining search triggering failures.",
    }[largest_gap]
    report = {
        "kind": "p3-seed2026-retrieval-answer-funnel",
        "n": 256,
        "runs": {"clean": args.clean_run, "aware": args.aware_run},
        "integrity": {
            "data_sha256": clean_result["data_files"]["sha256"],
            "episodes_sha256": {"clean": clean_sha, "aware": aware_sha},
            "same_protocol": True,
            "evidence_rule": "frozen v2 alias-in-returned-document-body matcher",
        },
        "clean": clean_summary,
        "aware": aware_summary,
        "paired_clean_no_search_to_aware_search": paired_added_search,
        "decision": {
            "largest_incorrect_partition": largest_gap,
            "counts": gaps,
            "recommendation": recommendation,
            "claim_boundary": "Evidence hit is a deterministic answer-alias proxy, not a semantic sufficiency judge.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    partial.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
