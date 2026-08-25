#!/usr/bin/env python3
"""Continue SFT-50 on the frozen official-only tool-rich subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from local_retrieval.resnet50_encoder import sha256_file

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
MODEL_ROOT = PROJECT_DATA / "models/Qwen3-VL-8B-Instruct"
RUN_ROOT = PROJECT_DATA / "runs"
DATASET_ROOT = (
    PROJECT_DATA
    / "datasets/processed/search-vl-sft-wiki-en-official-toolrich-97-r2c1c460-c5120"
)
DATASET_NAME = "wiki_en_official_toolrich"
DATASET_SHA256 = "65e823e7bbbe070ed1b17fc60573be927e745f8482b9808666357577cd37df02"
DATASET_INFO_SHA256 = "7386ea37c0830023ffc63f7f034a2fd1b47057eab21bd2f7b1e1e09bb1d08aae"
MANIFEST_SHA256 = "56e4a0673ad2323a30d71ccb567dec1ec98a306a297e23e219bd8ef25cea6b17"
ALIGNMENT_SHA256 = "b882ab3ff063d8d9fc6fa5949e2722ac823ff2a598bbec1b416e6ef1b72ed5d3"
SFT50_ADAPTER = (
    RUN_ROOT / "official-sft-wiki-en-safe960-v4-step50-20260825/output"
)
SFT50_ADAPTER_SHA256 = "8b7e3e49526da33730868ba4b84dce0a5e3310bb602d480067fc6a4909a57955"
TRAINING_PROFILE = "official-toolrich97-sft50-continuation-qlora-v1"


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
        raise RuntimeError("tool-rich SFT must run inside scripts/run_managed.sh")
    if len(visible.split(",")) != 1 or visible in {"0", "5"}:
        raise RuntimeError("tool-rich SFT requires one stable GPU, excluding GPU0/GPU5")
    run_dir = Path(raw_run_dir).resolve()
    if run_dir != (RUN_ROOT / run_id).resolve() or not run_dir.is_dir():
        raise RuntimeError("unexpected managed Run directory")
    return run_dir, visible


def validate_dataset() -> dict[str, Any]:
    manifest_path = DATASET_ROOT / "manifest.json"
    data_path = DATASET_ROOT / "wiki_en_official_toolrich.json"
    info_path = DATASET_ROOT / "dataset_info.json"
    alignment_path = DATASET_ROOT / "alignment-audit-cutoff5120-v1.json"
    expected = {
        manifest_path: MANIFEST_SHA256,
        data_path: DATASET_SHA256,
        info_path: DATASET_INFO_SHA256,
        alignment_path: ALIGNMENT_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"frozen tool-rich asset changed: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "official-toolrich-sft-ready"
        or manifest.get("sample_size") != 97
        or manifest.get("rows_modified") != 0
        or manifest.get("duplicates_added") != 0
        or manifest.get("real_tool_error_rows") != 7
        or manifest.get("tool_row_counts", {}).get("crop") != 56
    ):
        raise ValueError("tool-rich manifest violates the frozen official-only contract")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if (
        alignment.get("checked") != 97
        or alignment.get("mismatch_count") != 0
        or alignment.get("zero_supervision_count") != 0
    ):
        raise ValueError("tool-rich alignment audit did not pass all 97 rows")
    return manifest


def validate_resume(checkpoint: Path | None, max_steps: int) -> tuple[Path, int]:
    if checkpoint is None:
        if sha256_file(SFT50_ADAPTER / "adapter_model.safetensors") != SFT50_ADAPTER_SHA256:
            raise ValueError("frozen SFT-50 source adapter changed")
        return SFT50_ADAPTER, 0
    resolved = checkpoint.resolve()
    if not resolved.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("resume checkpoint escaped project4 runs")
    state_path = resolved / "trainer_state.json"
    adapter_path = resolved / "adapter_model.safetensors"
    if not state_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError("resume checkpoint lacks trainer state or adapter")
    previous_step = int(json.loads(state_path.read_text()).get("global_step", -1))
    if previous_step < 1 or max_steps <= previous_step:
        raise ValueError("max_steps must exceed the resume global step")
    provenance = resolved.parents[1] / "official-toolrich-sft-provenance.json"
    if not provenance.is_file():
        raise FileNotFoundError("resume checkpoint lacks tool-rich provenance")
    recorded = json.loads(provenance.read_text(encoding="utf-8"))
    if (
        recorded.get("training_profile") != TRAINING_PROFILE
        or recorded.get("dataset_manifest_sha256") != MANIFEST_SHA256
    ):
        raise ValueError("resume checkpoint belongs to another training contract")
    return resolved, previous_step


def training_config(
    run_dir: Path, max_steps: int, adapter: Path, resume_step: int
) -> dict[str, object]:
    config: dict[str, object] = {
        "model_name_or_path": str(MODEL_ROOT),
        "adapter_name_or_path": str(adapter),
        "create_new_adapter": False,
        "image_max_pixels": 65536,
        "video_max_pixels": 16384,
        "trust_remote_code": False,
        "flash_attn": "fa2",
        "quantization_bit": 4,
        "quantization_method": "bnb",
        "quantization_type": "nf4",
        "double_quantization": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 8,
        "lora_dropout": 0.0,
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "dataset": DATASET_NAME,
        "dataset_dir": str(DATASET_ROOT),
        "template": "qwen3_vl",
        "cutoff_len": 5120,
        "max_samples": 97,
        "overwrite_cache": False,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "output_dir": str(run_dir / "output"),
        "logging_steps": 1,
        "save_steps": max_steps,
        "save_total_limit": 1,
        "save_only_model": False,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": True,
        "learning_rate": 0.00005,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "optim": "adamw_torch",
        "max_steps": max_steps,
        "bf16": True,
        "seed": 42,
        "data_seed": 42,
        "ddp_timeout": 600,
    }
    if resume_step:
        config["resume_from_checkpoint"] = str(adapter)
    return config


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_steps <= 20:
        raise ValueError("tool-rich SFT pilot is bounded to 1..20 optimizer steps")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    manifest = validate_dataset()
    adapter, resume_step = validate_resume(args.resume_from_checkpoint, args.max_steps)
    if (run_dir / "output").exists():
        raise FileExistsError("refusing to overwrite tool-rich SFT output")
    config = training_config(run_dir, args.max_steps, adapter, resume_step)
    config_path = run_dir / "official-toolrich-sft-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    provenance = {
        "training_profile": TRAINING_PROFILE,
        "source_is_official": True,
        "source_revision": manifest["source_revision"],
        "dataset_root": str(DATASET_ROOT),
        "dataset_name": DATASET_NAME,
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "alignment_sha256": ALIGNMENT_SHA256,
        "source_adapter": str(adapter),
        "source_adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "resume_global_step": resume_step,
        "max_steps": args.max_steps,
        "physical_gpu": physical_gpu,
        "method": "official-only tool-rich QLoRA continuation from SFT-50",
    }
    (run_dir / "official-toolrich-sft-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
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
    cli = Path(sys.executable).with_name("llamafactory-cli")
    return subprocess.run([str(cli), "train", str(config_path)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
