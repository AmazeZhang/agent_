#!/usr/bin/env python3
"""Per-question comparison of the HF-transformers and vLLM-native eval runs.

CPU-only analysis over two episodes.jsonl files (from
run_p3_eval_heldout.py and run_p3_eval_vllm.py). Aligns episodes by
normalized question and reports, per question and in aggregate:

  - EM (environment reward >= 1.0) per backend
  - search rate (number of steps that executed a search)
  - invalid actions (steps whose projected action was invalid)
  - raw action text: byte-identical / first-step identical, normalized
    Levenshtein distance over the concatenated action text

Running speed is NOT compared as a quality metric (per project decision).

Usage:
  python compare_hf_vllm_eval.py --hf <episodes.jsonl> --vllm <episodes.jsonl>
  python compare_hf_vllm_eval.py --hf <...> --vllm <...> --output <result.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def load_episodes(path: Path) -> list[dict[str, Any]]:
    episodes = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, a_char in enumerate(a, 1):
        current = [i]
        for j, b_char in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a_char != b_char)))
        previous = current
    return previous[-1]


def episode_features(episode: dict[str, Any]) -> dict[str, Any]:
    steps = episode["steps"]
    return {
        "em": episode["reward"] >= 1.0,
        "won": episode["won"],
        "searches": sum(1 for step in steps if step.get("executed_search")),
        "invalid_actions": sum(1 for step in steps if not step["action_quality"]["projected_valid"]),
        "n_steps": len(steps),
        "raw_action_text": "\n".join(step["raw_action"] for step in steps),
        "raw_action_first_step": steps[0]["raw_action"] if steps else "",
    }


def compare(hf_episodes: list[dict[str, Any]], vllm_episodes: list[dict[str, Any]]) -> dict[str, Any]:
    hf_by_question = {normalize_question(ep["question"]): ep for ep in hf_episodes}
    vllm_by_question = {normalize_question(ep["question"]): ep for ep in vllm_episodes}

    questions = sorted(hf_by_question.keys() & vllm_by_question.keys())
    hf_only = sorted(hf_by_question.keys() - vllm_by_question.keys())
    vllm_only = sorted(vllm_by_question.keys() - hf_by_question.keys())
    if hf_only or vllm_only:
        raise RuntimeError(
            f"question sets differ: hf_only={len(hf_only)}, vllm_only={len(vllm_only)}"
        )

    rows: list[dict[str, Any]] = []
    byte_identical_actions = 0
    byte_identical_first_step = 0
    lev_sum = 0
    em_flips = 0
    search_rate_flips = 0
    invalid_flips = 0

    for question in questions:
        hf = episode_features(hf_by_question[question])
        vllm = episode_features(vllm_by_question[question])
        lev = levenshtein(hf["raw_action_text"], vllm["raw_action_text"])
        norm = lev / max(len(hf["raw_action_text"]), len(vllm["raw_action_text"]), 1)
        row = {
            "question": question,
            "source": hf_by_question[question]["source"],
            "em_hf": hf["em"],
            "em_vllm": vllm["em"],
            "searches_hf": hf["searches"],
            "searches_vllm": vllm["searches"],
            "invalid_hf": hf["invalid_actions"],
            "invalid_vllm": vllm["invalid_actions"],
            "n_steps_hf": hf["n_steps"],
            "n_steps_vllm": vllm["n_steps"],
            "raw_identical": hf["raw_action_text"] == vllm["raw_action_text"],
            "first_step_identical": hf["raw_action_first_step"] == vllm["raw_action_first_step"],
            "levenshtein": lev,
            "levenshtein_normalized": norm,
        }
        rows.append(row)
        if row["raw_identical"]:
            byte_identical_actions += 1
        if row["first_step_identical"]:
            byte_identical_first_step += 1
        lev_sum += norm
        if row["em_hf"] != row["em_vllm"]:
            em_flips += 1
        if row["searches_hf"] != row["searches_vllm"]:
            search_rate_flips += 1
        if row["invalid_hf"] != row["invalid_vllm"]:
            invalid_flips += 1

    n = len(rows)
    return {
        "n_questions": n,
        "per_question": rows,
        "aggregate": {
            "em_hf": sum(row["em_hf"] for row in rows),
            "em_vllm": sum(row["em_vllm"] for row in rows),
            "em_flips": em_flips,
            "search_rate_flips": search_rate_flips,
            "invalid_action_flips": invalid_flips,
            "byte_identical_actions": byte_identical_actions,
            "byte_identical_first_step": byte_identical_first_step,
            "mean_normalized_levenshtein": lev_sum / max(n, 1),
            "searches_hf": sum(row["searches_hf"] for row in rows),
            "searches_vllm": sum(row["searches_vllm"] for row in rows),
            "invalid_hf": sum(row["invalid_hf"] for row in rows),
            "invalid_vllm": sum(row["invalid_vllm"] for row in rows),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", type=Path, required=True, help="episodes.jsonl from run_p3_eval_heldout.py")
    parser.add_argument("--vllm", type=Path, required=True, help="episodes.jsonl from run_p3_eval_vllm.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    hf_episodes = load_episodes(args.hf)
    vllm_episodes = load_episodes(args.vllm)
    result = compare(hf_episodes, vllm_episodes)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_name(args.output.name + ".partial")
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        partial.replace(args.output)

    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    for row in result["per_question"]:
        markers = "".join(
            marker for marker, flag in (("E", row["em_hf"] != row["em_vllm"]), ("S", row["searches_hf"] != row["searches_vllm"]), ("I", row["invalid_hf"] != row["invalid_vllm"]))
            if flag
        )
        print(
            f"{row['source']:<12} em={int(row['em_hf'])}/{int(row['em_vllm'])} "
            f"search={row['searches_hf']}/{row['searches_vllm']} invalid={row['invalid_hf']}/{row['invalid_vllm']} "
            f"lev={row['levenshtein_normalized']:.3f} identical={int(row['raw_identical'])} {markers}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
