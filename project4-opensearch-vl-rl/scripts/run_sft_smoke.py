#!/usr/bin/env python3
"""Run a bounded, offline LoRA SFT smoke inside a project4 managed Run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
MODEL_ROOT = PROJECT_DATA / "models/Qwen3-VL-8B-Instruct"
DATASET_ROOT = PROJECT_DATA / "datasets/processed/sft-smoke-v1"
RUN_ROOT = PROJECT_DATA / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def require_managed_run() -> Path:
    run_id = os.environ.get("PROJECT4_RUN_ID", "")
    run_token = os.environ.get("PROJECT4_RUN_TOKEN", "")
    raw_run_dir = os.environ.get("PROJECT4_RUN_DIR", "")
    if not run_id or not run_token or not raw_run_dir:
        raise RuntimeError("SFT smoke must run inside scripts/run_managed.sh")
    run_dir = Path(raw_run_dir).resolve()
    expected = (RUN_ROOT / run_id).resolve()
    if run_dir != expected or not run_dir.is_dir():
        raise RuntimeError(f"unexpected managed Run directory: {run_dir}")
    return run_dir


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
    state = json.loads(state_path.read_text())
    previous_step = int(state.get("global_step", -1))
    if previous_step < 1 or max_steps <= previous_step:
        raise ValueError(
            f"max_steps={max_steps} must exceed checkpoint global_step={previous_step}"
        )
    return checkpoint


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_steps <= 5:
        raise ValueError("engineering smoke is bounded to 1..5 optimizer steps")
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if len(visible_gpus.split(",")) != 1 or visible_gpus == "0":
        raise RuntimeError("SFT smoke requires exactly one managed non-GPU0 device")

    run_dir = require_managed_run()
    checkpoint = validate_checkpoint(args.resume_from_checkpoint, args.max_steps)
    output_dir = run_dir / "output"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite SFT output: {output_dir}")

    synthetic_manifest = json.loads(
        (DATASET_ROOT / "SYNTHETIC_MANIFEST.json").read_text()
    )
    if synthetic_manifest.get("purpose") != "pipeline-smoke-only":
        raise RuntimeError("training input is not marked pipeline-smoke-only")

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
        "dataset": "sft_smoke_agentic",
        "dataset_dir": str(DATASET_ROOT),
        "template": "qwen3_vl",
        "cutoff_len": 1024,
        "max_samples": 4,
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

    config_path = run_dir / "sft-smoke-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    provenance = {
        "synthetic": True,
        "purpose": "pipeline-smoke-only",
        "max_steps": args.max_steps,
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "physical_gpu": visible_gpus,
    }
    (run_dir / "sft-smoke-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )

    cli = Path(sys.executable).with_name("llamafactory-cli")
    completed = subprocess.run([str(cli), "train", str(config_path)], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
