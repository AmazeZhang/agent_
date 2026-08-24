#!/usr/bin/env python3
"""Resume the local official-provider policy with fresh rollout per GRPO step."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import LocalTextIndex  # noqa: E402
from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402
from local_rl import score_evidence_fidelity_trajectory  # noqa: E402
from scripts.audit_stochastic_rollout_groups import (  # noqa: E402
    evaluate_batch_gate,
    summarize_group,
)
from scripts.evaluate_local_agent import (  # noqa: E402
    MODEL_ROOT,
    evaluate_task,
    load_tasks,
    require_managed_run,
    tools_for_protocol,
    validate_adapter,
    validate_dataset_root,
)
from scripts.run_official_grpo_replay_smoke import replay_group_backward  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--optimizer-state", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    return parser.parse_args()


def validate_resume_source(
    adapter: Path, optimizer_state: Path, start_step: int
) -> dict[str, Any]:
    state_path = adapter / "trainer_state.json"
    if not state_path.is_file() or not optimizer_state.is_file():
        raise FileNotFoundError("resume adapter requires trainer_state.json and optimizer state")
    if optimizer_state.resolve() != (adapter / "optimizer.pt").resolve():
        raise ValueError("optimizer state must belong to the resume adapter directory")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != start_step:
        raise ValueError("resume global_step does not match --start-step")
    if state.get("algorithm") not in {
        "single-epoch-on-policy-grpo-replay-fatal-clamped",
        "online-fresh-rollout-grpo-fatal-clamped",
    }:
        raise ValueError("resume state was not produced by the local GRPO chain")
    return state


def validate_online_group(group: dict[str, Any]) -> dict[str, Any]:
    summary = group["summary"]
    values = summary["advantages"]["fatal_clamped"]
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("online rollout advantages are empty or non-finite")
    gate = evaluate_batch_gate([group])
    if not gate["passed"] or not gate["has_nonzero_advantage"]:
        raise ValueError("fresh online rollout group did not pass the optimizer gate")
    return gate


def main() -> int:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    args = parse_args()
    if not 1 <= args.start_step < args.target_step <= 5:
        raise ValueError("online smoke requires 1 <= start-step < target-step <= 5")
    if args.rollouts != 4:
        raise ValueError("online smoke freezes exactly four rollouts per step")
    if not 0.0 < args.learning_rate <= 1e-4:
        raise ValueError("learning rate must be in (0, 1e-4]")
    if not 0.0 < args.temperature <= 1.5 or not 0.0 < args.top_p <= 1.0:
        raise ValueError("invalid sampling parameters")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    adapter = validate_adapter(args.adapter)
    assert adapter is not None
    optimizer_path = args.optimizer_state.resolve()
    source_state = validate_resume_source(adapter, optimizer_path, args.start_step)
    dataset_root = validate_dataset_root(args.dataset_root)
    output_dir = args.output_dir.resolve()
    if output_dir != (run_dir / "output").resolve() or output_dir.exists():
        raise ValueError("output-dir must be the absent managed Run output directory")
    manifest, tasks = load_tasks("train", 1, dataset_root, [args.task_id])
    task = tasks[0]
    protocol = str(manifest["tool_protocol"])
    if protocol != "official-local-v1" or manifest["tool_observation_schema"] != "official-provider-v1":
        raise ValueError("online GRPO requires the frozen official provider contract")
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
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("adapter exposes no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
    output_dir.mkdir()

    history = []
    current_adapter_sha = sha256_file(adapter / "adapter_model.safetensors")
    text_path = Path(manifest["text_index"])
    with LocalTextIndex(text_path) as text_index:
        for global_step in range(args.start_step + 1, args.target_step + 1):
            model.eval()
            items = []
            for rollout_index in range(args.rollouts):
                seed = args.seed + global_step * 10_000 + rollout_index
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
                    observation_format="official-provider-v1",
                    maximum_turns=int(manifest["maximum_agent_turns"]),
                    tool_protocol=protocol,
                )
                reward = score_evidence_fidelity_trajectory(result, task)
                items.append({"seed": seed, "result": result, "reward": reward})
                print(
                    json.dumps(
                        {
                            "global_step": global_step,
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
            group = {
                "task_id": task["task_id"],
                "task_type": task["task_type"],
                "items": items,
                "summary": summarize_group(items),
            }
            gate = validate_online_group(group)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            replay = replay_group_backward(
                model, processor, group, task, dataset_root, protocol, tools
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite adapter gradient norm")
            optimizer.step()

            staging = output_dir / f".checkpoint-{global_step}.partial"
            checkpoint = output_dir / f"checkpoint-{global_step}"
            if staging.exists() or checkpoint.exists():
                raise FileExistsError("refusing to overwrite online checkpoint")
            staging.mkdir()
            model.save_pretrained(staging, safe_serialization=True)
            torch.save(optimizer.state_dict(), staging / "optimizer.pt")
            adapter_sha = sha256_file(staging / "adapter_model.safetensors")
            if adapter_sha == current_adapter_sha:
                raise RuntimeError("optimizer step did not change the adapter hash")
            step_state = {
                "schema_version": 1,
                "algorithm": "online-fresh-rollout-grpo-fatal-clamped",
                "global_step": global_step,
                "physical_gpu": physical_gpu,
                "task_id": task["task_id"],
                "adapter_before_sha256": current_adapter_sha,
                "adapter_sha256": adapter_sha,
                "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
                "tasks_sha256": sha256_file(dataset_root / "tasks.jsonl"),
                "learning_rate": args.learning_rate,
                "generation": {
                    "rollouts": args.rollouts,
                    "seed_base": args.seed,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                },
                "group_summary": group["summary"],
                "gate": gate,
                **replay,
                "grad_norm": float(grad_norm.detach().cpu()),
            }
            (staging / "rollouts.json").write_text(
                json.dumps(group, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (staging / "trainer_state.json").write_text(
                json.dumps(step_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging.rename(checkpoint)
            history.append(step_state)
            current_adapter_sha = adapter_sha
            print(json.dumps({"step_state": step_state}, sort_keys=True), flush=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    final_state = {
        "schema_version": 1,
        "algorithm": "online-fresh-rollout-grpo-fatal-clamped",
        "global_step": args.target_step,
        "physical_gpu": physical_gpu,
        "source_adapter": str(adapter),
        "source_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "source_state": source_state,
        "history": history,
        "adapter_sha256": sha256_file(output_dir / "adapter_model.safetensors"),
    }
    (output_dir / "trainer_state.json").write_text(
        json.dumps(final_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_state, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
