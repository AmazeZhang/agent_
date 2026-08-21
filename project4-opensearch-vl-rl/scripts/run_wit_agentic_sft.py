#!/usr/bin/env python3
"""Run bounded LoRA SFT on the retrieval-verified WIT agentic pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
MODEL_ROOT = PROJECT_DATA / "models/Qwen3-VL-8B-Instruct"
DATASET_ROOT = PROJECT_DATA / "datasets/processed/wit-agentic-challenge-v4"
RUN_ROOT = PROJECT_DATA / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def require_managed_run(environment: dict[str, str]) -> tuple[Path, str]:
    run_id = environment.get("PROJECT4_RUN_ID", "")
    run_token = environment.get("PROJECT4_RUN_TOKEN", "")
    raw_run_dir = environment.get("PROJECT4_RUN_DIR", "")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not run_id or not run_token or not raw_run_dir:
        raise RuntimeError("WIT SFT must run inside scripts/run_managed.sh")
    if len(visible.split(",")) != 1 or visible in {"0", "5"}:
        raise RuntimeError("WIT SFT requires one managed stable GPU, excluding GPU0/GPU5")
    run_dir = Path(raw_run_dir).resolve()
    if run_dir != (RUN_ROOT / run_id).resolve() or not run_dir.is_dir():
        raise RuntimeError(f"unexpected managed Run directory: {run_dir}")
    return run_dir, visible


def validate_dataset(root: Path) -> dict[str, object]:
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {
        "status": "challenge-ready",
        "purpose": "local-agentic-sft-rl-challenge",
        "image_observation_contains_text_summary": False,
        "image_runtime_handle": "img_1",
        "final_response_format": "Title: <exact title>\\nEvidence: <first sentence-or-no-match>",
        "evidence_extraction": "first_terminal_punctuation_or_360_characters",
        "split_unit": "entity_id-or-synthetic-probe-id",
        "maximum_agent_turns": 4,
        "text_lookup_summary_max_characters": 360,
        "image_search_top_k_maximum": 3,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"dataset manifest {field} is not {expected!r}")
    if manifest.get("split_counts") != {"dev": 20, "test": 20, "train": 80}:
        raise ValueError("dataset split counts are not the fixed 80/20/20 pilot")
    if manifest.get("task_type_counts") != {
        "candidate-conflict": 48,
        "clean": 12,
        "no-match": 24,
        "transient-tool-failure": 36,
    }:
        raise ValueError("dataset task type counts are not fixed")
    return manifest


def validate_checkpoint(raw_checkpoint: Path | None, max_steps: int) -> Path | None:
    if raw_checkpoint is None:
        return None
    checkpoint = raw_checkpoint.resolve()
    if not checkpoint.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("resume checkpoint must belong to a project4 managed Run")
    state_path = checkpoint / "trainer_state.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not state_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError("resume checkpoint lacks trainer state or LoRA adapter")
    with state_path.open(encoding="utf-8") as handle:
        previous_step = int(json.load(handle).get("global_step", -1))
    if previous_step < 1 or max_steps <= previous_step:
        raise ValueError(
            f"max_steps={max_steps} must exceed checkpoint global_step={previous_step}"
        )
    return checkpoint


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_steps <= 20:
        raise ValueError("WIT SFT pilot is bounded to 1..20 optimizer steps")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    dataset_manifest = validate_dataset(DATASET_ROOT)
    checkpoint = validate_checkpoint(args.resume_from_checkpoint, args.max_steps)
    output_dir = run_dir / "output"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite SFT output: {output_dir}")

    cache_root = PROJECT_DATA / "cache/huggingface"
    os.environ.update(
        {
            "HF_HOME": str(cache_root),
            "HF_DATASETS_CACHE": str(cache_root / "datasets"),
            "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    config: dict[str, object] = {
        "model_name_or_path": str(MODEL_ROOT),
        "image_max_pixels": 65536,
        "video_max_pixels": 16384,
        "trust_remote_code": False,
        "flash_attn": "fa2",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 8,
        "lora_dropout": 0.0,
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "dataset": "wit_agentic_train_v1",
        "dataset_dir": str(DATASET_ROOT),
        "template": "qwen3_vl_nothink",
        "cutoff_len": 2048,
        "max_samples": 80,
        "overwrite_cache": False,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": str(output_dir),
        "logging_steps": 1,
        "save_steps": 1,
        "save_total_limit": args.max_steps,
        "save_only_model": False,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "learning_rate": 0.0001,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "optim": "adamw_torch",
        "max_steps": args.max_steps,
        "bf16": True,
        "seed": 42,
        "data_seed": 42,
        "ddp_timeout": 600,
    }
    if checkpoint is not None:
        config["resume_from_checkpoint"] = str(checkpoint)
    config_path = run_dir / "wit-agentic-sft-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    provenance = {
        "synthetic": False,
        "derived_dataset": True,
        "purpose": "local-agentic-sft-rl-pilot",
        "dataset_manifest_sha256": sha256_file(DATASET_ROOT / "manifest.json"),
        "tasks_sha256": sha256_file(DATASET_ROOT / "tasks.jsonl"),
        "dataset_manifest": dataset_manifest,
        "template": "qwen3_vl_nothink",
        "max_steps": args.max_steps,
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "physical_gpu": physical_gpu,
    }
    (run_dir / "wit-agentic-sft-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    cli = Path(sys.executable).with_name("llamafactory-cli")
    completed = subprocess.run([str(cli), "train", str(config_path)], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
