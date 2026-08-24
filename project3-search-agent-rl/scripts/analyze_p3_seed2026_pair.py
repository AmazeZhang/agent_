#!/usr/bin/env python3
"""Strict paired analysis for the preregistered P3 seed2026 clean/aware runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_CLEAN = "p3-eval-clean-grpo10-seed2026-gs10-confirm256-20260824a"
DEFAULT_AWARE = "p3-eval-aware-v2-seed2026-gs10-confirm256-20260824a"
ENV_MAX_STEPS = 4


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 6), round(center + half, 6)]


def mcnemar_exact(lost: int, gained: int) -> float:
    n = lost + gained
    if not n:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(math.comb(n, i) * 0.5**n for i in range(k + 1)))


def is_search(step: dict) -> bool:
    return step.get("info", {}).get("tool_name") == "search"


def is_answer(step: dict) -> bool:
    return str(step.get("info", {}).get("postprocessed_action", "")).startswith("<answer>")


def analyze_episode(ep: dict) -> dict:
    steps = ep["steps"]
    search_actions = [step for step in steps if is_search(step)]
    valid_searches = [
        step for step in search_actions
        if (step.get("info", {}).get("retrieval") or {}).get("status") == "success"
    ]
    searched = bool(valid_searches)
    answered_env = any(is_answer(step) for step in steps)
    answered_offline = bool(ep.get("offline", {}).get("has_answer", False))
    invalid = any(
        is_search(step)
        and (
            step.get("info", {}).get("retrieval_failed", False)
            or bool((step.get("info", {}).get("retrieval") or {}).get("api_request_error"))
        )
        for step in steps
    )
    seen_docs: set[str] = set()
    redundant = 0
    for step in valid_searches:
        doc_ids = set((step.get("info", {}).get("retrieval") or {}).get("document_ids") or [])
        if doc_ids and doc_ids <= seen_docs:
            redundant += 1
        seen_docs.update(doc_ids)
    return {
        "question_id": ep["question_id"],
        "source": ep["source"],
        "won": bool(ep["won"]),
        "searched": searched,
        "search_action_attempted": bool(search_actions),
        "answered_env": answered_env,
        "answered_offline": answered_offline,
        "invalid": invalid,
        "max_steps_exhausted": not answered_env and len(steps) >= ENV_MAX_STEPS,
        "steps": len(steps),
        "search_action_calls": len(search_actions),
        "valid_search_calls": len(valid_searches),
        "redundant_searches": redundant,
    }


def load_run(root: Path, run_id: str) -> tuple[dict, dict[int, dict]]:
    run = root / run_id
    results = json.loads((run / "results.json").read_text())
    rows = [json.loads(line) for line in (run / "episodes.jsonl").open()]
    analyzed = [analyze_episode(row) for row in rows]
    by_qid = {row["question_id"]: row for row in analyzed}
    assert len(rows) == len(by_qid) == 256, f"{run_id}: expected 256 unique episodes"
    assert [row["question_id"] for row in analyzed] == sorted(by_qid), f"{run_id}: qids not sorted"
    assert sum(row["won"] for row in analyzed) == results["metrics"]["overall"]["em"]
    return results, by_qid


def behavior(rows: dict[int, dict]) -> dict:
    eps = list(rows.values())
    searched = [ep for ep in eps if ep["searched"]]
    searched_answered = [ep for ep in searched if ep["answered_offline"]]
    search_action_calls = sum(ep["search_action_calls"] for ep in eps)
    valid_search_calls = sum(ep["valid_search_calls"] for ep in eps)
    redundant = sum(ep["redundant_searches"] for ep in eps)
    return {
        "em": sum(ep["won"] for ep in eps),
        "wilson95": wilson(sum(ep["won"] for ep in eps), len(eps)),
        "searched": len(searched),
        "search_rate": len(searched) / len(eps),
        "search_action_questions_including_invalid": sum(ep["search_action_attempted"] for ep in eps),
        "answered_offline": sum(ep["answered_offline"] for ep in eps),
        "answered_env": sum(ep["answered_env"] for ep in eps),
        "search_to_answer": len(searched_answered) / len(searched),
        "searched_and_correct": sum(ep["searched"] and ep["won"] for ep in eps),
        "search_to_correct": sum(ep["searched"] and ep["won"] for ep in eps) / len(searched),
        "no_search_and_correct": sum((not ep["searched"]) and ep["won"] for ep in eps),
        "invalid_questions": sum(ep["invalid"] for ep in eps),
        "max_steps_exhausted": sum(ep["max_steps_exhausted"] for ep in eps),
        "mean_steps": sum(ep["steps"] for ep in eps) / len(eps),
        "search_action_calls_including_invalid": search_action_calls,
        "valid_search_calls": valid_search_calls,
        "true_redundant_searches": redundant,
        "true_redundant_rate": redundant / valid_search_calls if valid_search_calls else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/media/imc/data")
    parser.add_argument("--clean-run", default=DEFAULT_CLEAN)
    parser.add_argument("--aware-run", default=DEFAULT_AWARE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs_root = Path(args.data_root) / "project3-search-agent-rl" / "runs"
    clean_results, clean = load_run(runs_root, args.clean_run)
    aware_results, aware = load_run(runs_root, args.aware_run)

    assert set(clean) == set(aware) == set(range(256)), "qid sets differ from 0..255"
    clean_sha = clean_results["data_files"]["sha256"]
    aware_sha = aware_results["data_files"]["sha256"]
    assert clean_sha == aware_sha, "evaluation data SHA differs"
    for key in ("temperature", "num_rollouts", "seed", "max_steps", "history_length", "topk", "retrieval_condition"):
        assert clean_results["parameters"][key] == aware_results["parameters"][key], f"parameter mismatch: {key}"

    cells = {
        "both_correct": 0,
        "both_wrong": 0,
        "clean_correct_aware_wrong": 0,
        "clean_wrong_aware_correct": 0,
    }
    per_source: dict[str, dict[str, int]] = {}
    discordant: list[dict] = []
    for qid in sorted(clean):
        c, a = clean[qid], aware[qid]
        assert c["source"] == a["source"]
        if c["won"] and a["won"]:
            cell = "both_correct"
        elif not c["won"] and not a["won"]:
            cell = "both_wrong"
        elif c["won"]:
            cell = "clean_correct_aware_wrong"
        else:
            cell = "clean_wrong_aware_correct"
        cells[cell] += 1
        per_source.setdefault(c["source"], {key: 0 for key in cells})[cell] += 1
        if c["won"] != a["won"]:
            discordant.append({
                "question_id": qid,
                "source": c["source"],
                "direction": "lost" if c["won"] else "gained",
                "clean_searched": c["searched"],
                "aware_searched": a["searched"],
            })

    lost = cells["clean_correct_aware_wrong"]
    gained = cells["clean_wrong_aware_correct"]
    clean_behavior = behavior(clean)
    aware_behavior = behavior(aware)
    report = {
        "kind": "p3-seed2026-clean-aware-paired-confirm256",
        "data_sha256": clean_sha,
        "n": 256,
        "runs": {"clean": args.clean_run, "aware": args.aware_run},
        "integrity": {
            "qid_space": "0..255 sorted and unique",
            "data_sha_equal": True,
            "decoding_and_environment_parameters_equal": True,
            "temperature": clean_results["parameters"]["temperature"],
            "seed": clean_results["parameters"]["seed"],
            "max_steps": clean_results["parameters"]["max_steps"],
            "retrieval_condition": clean_results["parameters"]["retrieval_condition"],
        },
        "behavior": {"clean": clean_behavior, "aware": aware_behavior},
        "behavior_delta_aware_minus_clean": {
            key: aware_behavior[key] - clean_behavior[key]
            for key in (
                "em", "search_rate", "search_to_answer", "searched_and_correct",
                "search_to_correct", "max_steps_exhausted", "mean_steps",
                "true_redundant_searches", "true_redundant_rate",
            )
        },
        "paired": {
            "cells": cells,
            "gained": gained,
            "lost": lost,
            "net": gained - lost,
            "discordant_n": gained + lost,
            "mcnemar_exact_two_sided_p": mcnemar_exact(lost, gained),
            "per_source": per_source,
            "discordant_questions": discordant,
        },
        "claim_boundary": (
            "The seed2026 pair supports a large search/completion behavior shift, but not an accuracy gain; "
            "EM is 78 vs 77 with exact paired p=1.0."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    partial.replace(args.out)
    print(json.dumps({"behavior": report["behavior"], "paired": report["paired"] | {"discordant_questions": "omitted"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
