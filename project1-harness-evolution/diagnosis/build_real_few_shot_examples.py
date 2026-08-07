"""Build AgentRx few-shot files only from labeled Magentic-One root causes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXAMPLES = {
    "instruction_adherence_failure.json": (
        1,
        "08cae58d-4084-4616-b6dd-dd6534e4825b",
        "plan_adherence_failure.json",
    ),
    "misinterpretation_of_tool_output.json": (
        4,
        "42d4198c-5895-4f0a-b0c0-424a66465d83",
        "misinterpret_tool_info.json",
    ),
    "intent_plan_misalignment.json": (
        5,
        "3af8028c2a59e28ca88baff0e6d91f2a9f170c5ef91003f1c8406755a2760ad4",
        "intent_plan_misalignment.json",
    ),
    "intent_not_supported.json": (
        7,
        "1f975693-876d-457b-a649-393859e79bf3",
        "intent_not_supported.json",
    ),
    "guardrails_triggered.json": (
        8,
        "2aa5dd83fbcd0dce9a3dd4592106e5b5edf738008d932e357d477bba80e59ccf",
        "rai_policy_violation.json",
    ),
}


def root_failure(entry: dict[str, Any]) -> dict[str, Any]:
    root_id = int(entry["root_cause"]["failure_id"])
    matches = [failure for failure in entry["failures"] if int(failure["failure_id"]) == root_id]
    if len(matches) != 1:
        raise ValueError(f"expected one root failure for {entry['trajectory_id']}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    entries = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    by_id = {entry["trajectory_id"]: entry for entry in entries}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for filename, (category_id, trajectory_id, trajectory_filename) in EXAMPLES.items():
        entry = by_id[trajectory_id]
        failure = root_failure(entry)
        trajectory_path = args.trajectory_dir / trajectory_filename
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        root_step = int(failure["step_number"])
        window_start = max(0, root_step - 3)
        window_end = min(len(trajectory), root_step + 2)
        excerpt = []
        for position in range(window_start, window_end):
            message = trajectory[position]
            excerpt.append(
                {
                    "step_number": position + 1,
                    "role": message.get("role", "unknown"),
                    "content": str(message.get("content", ""))[:800],
                }
            )
        example = {
            "provenance": "AgentRx Magentic-One human ground truth",
            "source_domain": "magentic_one",
            "trajectory_id": trajectory_id,
            "failure_case": category_id,
            "failure_category": failure["failure_category"],
            "root_cause_step": failure["step_number"],
            "step_reason": failure["step_reason"],
            "category_reason": failure["category_reason"],
            "failed_agent": failure["failed_agent"],
            "reason_for_root_cause": entry["root_cause"]["reason_for_root_cause"],
            "trajectory_excerpt": excerpt,
            "expected_output": {
                "failure_case": category_id,
                "index": root_step,
            },
        }
        (args.output_dir / filename).write_text(
            json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest.append(
            {
                "filename": filename,
                "category_id": category_id,
                "trajectory_id": trajectory_id,
                "trajectory_file": str(trajectory_path),
            }
        )

    payload = {
        "schema_version": 1,
        "source_ground_truth": str(args.ground_truth),
        "evaluation_domain": "tau_retail",
        "leakage_control": "examples and evaluation use different domains and trajectory IDs",
        "examples": manifest,
        "intentionally_missing_categories": [2, 3, 6, 9],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
