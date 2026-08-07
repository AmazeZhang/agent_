"""Summarize single-run AgentRx judge outputs without relaxing correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FAILURE_CASES = {
    0: "NO_ERROR_PREDICTED",
    1: "INSTRUCTION_OR_PLAN_ADHERENCE_FAILURE",
    2: "INVENTION_OF_NEW_INFORMATION",
    3: "INVALID_INVOCATION",
    4: "MISINTERPRETATION_OF_TOOL_OUTPUT",
    5: "INTENT_PLAN_MISALIGNMENT",
    6: "UNDERSPECIFIED_USER_INTENT",
    7: "INTENT_NOT_SUPPORTED",
    8: "GUARDRAILS_TRIGGERED",
    9: "SYSTEM_FAILURE",
    10: "INCONCLUSIVE",
}


def load_case(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["detailed_results"][0]
    prediction = result["failures"][0]
    predicted_case = int(prediction["failure_case"])
    ground_truth_case = int(result["gt_failure_case"])
    predicted_step = int(prediction["step_number"])
    ground_truth_step = int(result["gt_step_number"])
    step_error = abs(predicted_step - ground_truth_step)
    return {
        "case": path.parents[2].name,
        "task_id": str(result["task_id"]),
        "ground_truth_category": FAILURE_CASES[ground_truth_case],
        "ground_truth_step": ground_truth_step,
        "predicted_category": FAILURE_CASES[predicted_case],
        "predicted_step": predicted_step,
        "category_correct": predicted_case == ground_truth_case,
        "exact_step": step_error == 0,
        "step_within_1": step_error <= 1,
        "step_within_2": step_error <= 2,
        "step_absolute_error": step_error,
        "prompt_tokens": payload["summary"]["total_prompt_tokens"],
        "output_tokens": payload["summary"]["total_output_tokens"],
        "total_tokens": payload["summary"]["total_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--few-shot-examples-loaded", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(args.input_root.glob("*/judge_output/runs/run1.json"))
    if not paths:
        raise SystemExit(f"no run1.json files found below {args.input_root}")
    entries = [load_case(path) for path in paths]
    total = len(entries)
    summary = {
        "total": total,
        "category_correct": sum(row["category_correct"] for row in entries),
        "category_accuracy": sum(row["category_correct"] for row in entries) / total,
        "exact_step": sum(row["exact_step"] for row in entries),
        "exact_step_accuracy": sum(row["exact_step"] for row in entries) / total,
        "step_within_1": sum(row["step_within_1"] for row in entries),
        "step_within_1_accuracy": sum(row["step_within_1"] for row in entries) / total,
        "step_within_2": sum(row["step_within_2"] for row in entries),
        "step_within_2_accuracy": sum(row["step_within_2"] for row in entries) / total,
        "mean_step_absolute_error": sum(row["step_absolute_error"] for row in entries) / total,
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in entries),
        "total_output_tokens": sum(row["output_tokens"] for row in entries),
        "total_tokens": sum(row["total_tokens"] for row in entries),
        "judge_runs_per_case": 1,
        "few_shot_examples_loaded": args.few_shot_examples_loaded,
    }
    output = {
        "schema_version": 1,
        "evaluation_note": (
            "Metrics use exact ground-truth root-cause category matching. Step tolerance does "
            "not turn an incorrect category into a correct diagnosis."
        ),
        "summary": summary,
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
