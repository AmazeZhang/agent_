#!/usr/bin/env python3
"""Audit small stochastic rollout groups before any RL parameter update."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import LocalTextIndex  # noqa: E402
from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402
from local_rl import (  # noqa: E402
    compute_group_advantages,
    score_evidence_fidelity_trajectory,
    score_trajectory,
)
from scripts.evaluate_local_agent import (  # noqa: E402
    DATASET_ROOT,
    MODEL_ROOT,
    evaluate_task,
    load_tasks,
    observation_schema,
    protocol_version,
    require_managed_run,
    validate_adapter,
    validate_dataset_root,
)

MINIMUM_VARIABLE_GROUP_FRACTION = 0.25
MINIMUM_FORMAT_VALID_FRACTION = 0.75
MAXIMUM_FATAL_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reward-version",
        choices=("rules-v1", "evidence-fidelity-v2"),
        default="rules-v1",
    )
    return parser.parse_args()


def load_selected_tasks(
    task_ids: list[str], dataset_root: Path = DATASET_ROOT
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 1 <= len(task_ids) <= 8 or len(task_ids) != len(set(task_ids)):
        raise ValueError("provide between one and eight unique task IDs")
    return load_tasks("train", len(task_ids), dataset_root, task_ids)


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(item["reward"]["reward"]) for item in items]
    fatal_flags = [bool(item["reward"]["is_fatal"]) for item in items]
    advantages = compute_group_advantages(rewards, fatal_flags)
    return {
        "rollout_count": len(items),
        "reward_mean": statistics.fmean(rewards),
        "reward_population_variance": statistics.pvariance(rewards),
        "unique_rewards": sorted(set(rewards)),
        "full_success_count": sum(item["result"]["score"]["full_success"] for item in items),
        "format_valid_count": sum(item["result"]["score"]["format_valid"] for item in items),
        "fatal_count": sum(fatal_flags),
        "query_only_reward_count": sum(
            item["reward"]["r_query"] > 0
            and item["reward"].get(
                "r_accuracy", item["reward"].get("r_exact_success")
            )
            == 0
            for item in items
        ),
        "advantages": advantages,
    }


def evaluate_batch_gate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        raise ValueError("cannot gate an empty rollout batch")
    rollout_count = sum(group["summary"]["rollout_count"] for group in groups)
    variable_group_count = sum(
        group["summary"]["reward_population_variance"] > 0 for group in groups
    )
    format_valid_count = sum(
        group["summary"]["format_valid_count"] for group in groups
    )
    fatal_count = sum(group["summary"]["fatal_count"] for group in groups)
    variable_group_fraction = variable_group_count / len(groups)
    format_valid_fraction = format_valid_count / rollout_count
    fatal_fraction = fatal_count / rollout_count
    return {
        "variable_group_count": variable_group_count,
        "group_count": len(groups),
        "variable_group_fraction": variable_group_fraction,
        "minimum_variable_group_fraction": MINIMUM_VARIABLE_GROUP_FRACTION,
        "format_valid_fraction": format_valid_fraction,
        "minimum_format_valid_fraction": MINIMUM_FORMAT_VALID_FRACTION,
        "fatal_fraction": fatal_fraction,
        "maximum_fatal_fraction": MAXIMUM_FATAL_FRACTION,
        "has_nonzero_advantage": variable_group_count > 0,
        "passed": (
            variable_group_fraction >= MINIMUM_VARIABLE_GROUP_FRACTION
            and format_valid_fraction >= MINIMUM_FORMAT_VALID_FRACTION
            and fatal_fraction <= MAXIMUM_FATAL_FRACTION
        ),
        "note": (
            "The batch gate allows zero-variance prompts because they contribute zero GRPO "
            "gradient, but requires a predeclared minimum variable-group fraction. Passing "
            "authorizes only a separately reviewed one-step GRPO smoke; it is not evidence "
            "of policy improvement. Query-only reward remains disclosed per group."
        ),
    }


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    if not 2 <= args.rollouts <= 8:
        raise ValueError("rollouts must be between 2 and 8")
    if not 0.0 < args.temperature <= 1.5 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("invalid sampling temperature/top-p")
    if not 32 <= args.max_new_tokens <= 256:
        raise ValueError("max-new-tokens must be between 32 and 256")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    adapter = validate_adapter(args.adapter)
    assert adapter is not None
    output = args.output.resolve()
    if output != (run_dir / "stochastic_rollout_audit.json").resolve():
        raise ValueError("output must be the managed Run stochastic_rollout_audit.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit report: {output}")
    dataset_root = validate_dataset_root(args.dataset_root)
    manifest, tasks = load_selected_tasks(args.task_id, dataset_root)
    runtime_observation_format = observation_schema(manifest)
    runtime_protocol = protocol_version(manifest)
    if runtime_protocol == "official-local-v1" and runtime_observation_format != "official-provider-v1":
        raise ValueError(
            "official-local-v1 stochastic rollout requires official-provider-v1 observations"
        )
    reward_function = (
        score_trajectory
        if args.reward_version == "rules-v1"
        else score_evidence_fidelity_trajectory
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ROOT, local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ROOT,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()

    groups = []
    text_path = Path(manifest["text_index"])
    with LocalTextIndex(text_path) as text_index:
        for task_index, task in enumerate(tasks):
            items = []
            for rollout_index in range(args.rollouts):
                seed = args.seed + task_index * 1000 + rollout_index
                result = evaluate_task(
                    model,
                    processor,
                    task,
                    text_index,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=seed,
                    dataset_root=dataset_root,
                    observation_format=runtime_observation_format,
                    tool_protocol=runtime_protocol,
                )
                reward = reward_function(result, task)
                items.append({"seed": seed, "result": result, "reward": reward})
                print(
                    json.dumps(
                        {
                            "task_id": task["task_id"],
                            "rollout": rollout_index + 1,
                            "seed": seed,
                            "fatal": result["fatal"],
                            "tools": result["tool_names"],
                            "full_success": result["score"]["full_success"],
                            "reward": reward["reward"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            groups.append(
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "items": items,
                    "summary": summarize_group(items),
                }
            )

    gate = evaluate_batch_gate(groups)
    repo_commit = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 2,
        "mode": "stochastic-rollout-only-no-optimizer-no-api",
        "repo_commit": repo_commit,
        "model": str(MODEL_ROOT),
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "tasks_sha256": sha256_file(dataset_root / "tasks.jsonl"),
        "tool_observation_schema": runtime_observation_format,
        "tool_protocol": runtime_protocol,
        "reward_version": args.reward_version,
        "physical_gpu": physical_gpu,
        "generation": {
            "do_sample": True,
            "rollouts_per_task": args.rollouts,
            "base_seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "groups": groups,
        "gate": gate,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"gate": payload["gate"]}, sort_keys=True))
    return 0 if payload["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
