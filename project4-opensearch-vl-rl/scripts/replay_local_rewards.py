#!/usr/bin/env python3
"""Replay deterministic local rewards over immutable evaluation reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval.resnet50_encoder import sha256_file
from local_rl import score_evidence_fidelity_trajectory, score_trajectory

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--evaluation", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reward-version",
        choices=("rules-v1", "evidence-fidelity-v2"),
        default="rules-v1",
    )
    return parser.parse_args()


def mean_components(
    items: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, float]:
    return {
        field: sum(float(item[field]) for item in items) / len(items)
        for field in fields
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_DATA.resolve()):
        raise ValueError("output must be below the project data root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite reward report: {output}")
    with args.tasks.open(encoding="utf-8") as handle:
        tasks = {
            item["task_id"]: item
            for line in handle
            if line.strip() and (item := json.loads(line))
        }
    reports = {}
    if args.reward_version == "rules-v1":
        scorer = score_trajectory
        fields = ("r_accuracy", "r_query", "r_format", "reward")
    else:
        scorer = score_evidence_fidelity_trajectory
        fields = (
            "r_exact_success",
            "r_title",
            "r_evidence_f1",
            "r_answer",
            "r_query",
            "r_format",
            "reward",
        )
    for spec in args.evaluation:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or name in reports:
            raise ValueError(f"invalid or duplicate evaluation spec: {spec}")
        path = Path(raw_path).resolve(strict=True)
        with path.open(encoding="utf-8") as handle:
            evaluation = json.load(handle)
        if evaluation["tasks_sha256"] != sha256_file(args.tasks):
            raise ValueError(f"evaluation task hash mismatch: {name}")
        scored = []
        by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in evaluation["results"]:
            task = tasks[result["task_id"]]
            reward = scorer(result, task)
            item = {
                "task_id": result["task_id"],
                "task_type": result["task_type"],
                **reward,
            }
            scored.append(item)
            by_type[result["task_type"]].append(item)
        reports[name] = {
            "evaluation": str(path),
            "evaluation_sha256": sha256_file(path),
            "adapter_sha256": evaluation["adapter_sha256"],
            "task_count": len(scored),
            "mean": mean_components(scored, fields),
            "mean_by_task_type": {
                key: mean_components(value, fields)
                for key, value in sorted(by_type.items())
            },
            "fatal_count": sum(item["is_fatal"] for item in scored),
            "trajectories": scored,
        }
    payload = {
        "schema_version": 1 if args.reward_version == "rules-v1" else 2,
        "mode": f"deterministic-{args.reward_version}-no-api",
        "reward_version": args.reward_version,
        "weights": (
            {"accuracy": 0.8, "query": 0.2, "format_gate": "multiplicative"}
            if args.reward_version == "rules-v1"
            else {
                "answer": 0.8,
                "query": 0.2,
                "format_gate": "multiplicative",
                "answer_components": {
                    "strict_success": 0.5,
                    "title_exact": 0.2,
                    "evidence_token_f1": 0.3,
                },
            }
        ),
        "query_reward": (
            "gold-evidence-acquired-times-oracle-tool-efficiency"
            if args.reward_version == "rules-v1"
            else "gold-evidence-present-in-target-tool-observation-times-oracle-tool-efficiency"
        ),
        "fatal_policy": "mask-after-fatal-and-one-sided-nonnegative-advantage-clamp",
        "tasks": str(args.tasks.resolve()),
        "tasks_sha256": sha256_file(args.tasks),
        "reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {name: report["mean"] for name, report in reports.items()}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
