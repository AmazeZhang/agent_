#!/usr/bin/env python3
"""One-step QLoRA policy update from a frozen on-policy rollout group."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402
from scripts.evaluate_local_agent import (  # noqa: E402
    MODEL_ROOT,
    RUN_ROOT,
    build_initial_messages,
    load_tasks,
    require_managed_run,
    tools_for_protocol,
    validate_adapter,
    validate_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rollout-report", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    return parser.parse_args()


def validate_rollout_report(
    report: dict[str, Any],
    *,
    adapter: Path,
    dataset_root: Path,
    task_id: str,
) -> dict[str, Any]:
    expected = {
        "mode": "stochastic-rollout-only-no-optimizer-no-api",
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "tasks_sha256": sha256_file(dataset_root / "tasks.jsonl"),
        "tool_protocol": "official-local-v1",
        "tool_observation_schema": "official-provider-v1",
        "reward_version": "evidence-fidelity-v2",
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f"rollout report {key} does not match frozen input")
    if report.get("gate", {}).get("passed") is not True:
        raise ValueError("rollout report did not pass its pre-RL gate")
    groups = [group for group in report.get("groups", []) if group.get("task_id") == task_id]
    if len(groups) != 1:
        raise ValueError("rollout report must contain exactly one requested task group")
    group = groups[0]
    items = group.get("items", [])
    advantages = group.get("summary", {}).get("advantages", {}).get("fatal_clamped", [])
    if len(items) < 2 or len(items) != len(advantages):
        raise ValueError("rollout group items/advantages are malformed")
    if not all(math.isfinite(float(value)) for value in advantages):
        raise ValueError("rollout advantages must be finite")
    if not any(float(value) != 0.0 for value in advantages):
        raise ValueError("rollout group has no optimizer signal")
    return group


def append_turn_messages(messages: list[dict[str, Any]], turn: dict[str, Any]) -> None:
    messages.append(
        {"role": "assistant", "content": [{"type": "text", "text": turn["assistant"]}]}
    )
    if "observation" in turn:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<tool_response>\n{turn['observation']}\n</tool_response>",
                    }
                ],
            }
        )


def main() -> int:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    args = parse_args()
    if not 0.0 < args.learning_rate <= 1e-4:
        raise ValueError("learning rate must be in (0, 1e-4]")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    adapter = validate_adapter(args.adapter)
    assert adapter is not None
    dataset_root = validate_dataset_root(args.dataset_root)
    output_dir = args.output_dir.resolve()
    if output_dir != (run_dir / "output").resolve() or output_dir.exists():
        raise ValueError("output-dir must be the absent managed Run output directory")
    report_path = args.rollout_report.resolve()
    if not report_path.is_relative_to(RUN_ROOT.resolve()) or not report_path.is_file():
        raise ValueError("rollout report must belong to a project4 managed Run")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    group = validate_rollout_report(
        report,
        adapter=adapter,
        dataset_root=dataset_root,
        task_id=args.task_id,
    )
    manifest, tasks = load_tasks("train", 1, dataset_root, [args.task_id])
    task = tasks[0]
    protocol = str(manifest["tool_protocol"])
    tools = tools_for_protocol(protocol)

    processor = AutoProcessor.from_pretrained(
        MODEL_ROOT, local_files_only=True, trust_remote_code=False
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ROOT,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("adapter exposes no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    advantages = [float(value) for value in group["summary"]["advantages"]["fatal_clamped"]]
    active_count = sum(value != 0.0 for value in advantages)
    replayed_turns = 0
    token_count = 0
    weighted_loss_total = 0.0
    for item, advantage in zip(group["items"], advantages, strict=True):
        if advantage == 0.0:
            continue
        turns = item["result"]["turns"]
        if not turns:
            raise ValueError("active rollout trajectory has no assistant turns")
        messages = build_initial_messages(task, dataset_root, protocol)
        for turn in turns:
            prompt = processor.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            full_messages = messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn["assistant"]}],
                }
            ]
            full = processor.apply_chat_template(
                full_messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_len = int(prompt["input_ids"].shape[1])
            if not torch.equal(full["input_ids"][:, :prompt_len], prompt["input_ids"]):
                raise RuntimeError("assistant replay does not preserve the prompt token prefix")
            labels = full["input_ids"].clone()
            labels[:, :prompt_len] = -100
            supervised = int((labels != -100).sum().item())
            if supervised <= 0:
                raise RuntimeError("assistant replay produced no supervised tokens")
            full = full.to("cuda:0")
            labels = labels.to("cuda:0")
            output = model(**full, labels=labels, use_cache=False)
            if not torch.isfinite(output.loss):
                raise FloatingPointError("non-finite replay cross-entropy")
            scale = advantage / (active_count * len(turns))
            weighted = output.loss * scale
            weighted.backward()
            weighted_loss_total += float(weighted.detach().cpu())
            replayed_turns += 1
            token_count += supervised
            append_turn_messages(messages, turn)

    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite adapter gradient norm")
    optimizer.step()

    staging = run_dir / ".output.partial"
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging output: {staging}")
    staging.mkdir()
    model.save_pretrained(staging, safe_serialization=True)
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    state = {
        "schema_version": 1,
        "algorithm": "single-epoch-on-policy-grpo-replay-fatal-clamped",
        "objective": "mean_i(mean_turn(advantage_i * assistant_cross_entropy))",
        "global_step": 1,
        "physical_gpu": physical_gpu,
        "task_id": args.task_id,
        "rollout_report": str(report_path),
        "rollout_report_sha256": sha256_file(report_path),
        "source_adapter": str(adapter),
        "source_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "tasks_sha256": sha256_file(dataset_root / "tasks.jsonl"),
        "learning_rate": args.learning_rate,
        "active_trajectory_count": active_count,
        "replayed_assistant_turns": replayed_turns,
        "supervised_token_count": token_count,
        "weighted_loss": weighted_loss_total,
        "grad_norm": float(grad_norm.detach().cpu()),
    }
    (staging / "trainer_state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.rename(output_dir)
    print(json.dumps(state, sort_keys=True), flush=True)
    print(
        json.dumps(
            {"adapter_sha256": sha256_file(output_dir / "adapter_model.safetensors")},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
