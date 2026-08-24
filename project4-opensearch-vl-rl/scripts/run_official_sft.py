#!/usr/bin/env python3
"""Run bounded offline LoRA SFT on the audited official Search-VL subset."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
MODEL_ROOT = PROJECT_DATA / "models/Qwen3-VL-8B-Instruct"
DATASET_ROOT = (
    PROJECT_DATA
    / "datasets/processed/search-vl-sft-wiki-en-official-1000-r2c1c460"
)
RUN_ROOT = PROJECT_DATA / "runs"
DATASET_NAME = "wiki_en_official_1000"
SOURCE_REVISION = "2c1c460af4fa15bd63210cbf426a96664b959944"
SOURCE_SHA256 = "a22a44c6a04d79d6dfd0064c89d8a792045278eed70a8e27c14b7c5e2f4850e3"
DATASET_SHA256 = "af5eb4adc2a9e4fcc0529ed2c6cfc523fca3753740aec1178c57acd54b4a3dd7"
DATASET_INFO_SHA256 = "6b065a7c32ddc1ac5ac79c4575fe84cc71322fd73f53739ac62d9443d0b3641f"
INDICES_SHA256 = "3195eafee69202c74cfb382cb7572fc198a471a357ec6af625f58d653d072018"
CUTOFF_LEN = 5120


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def require_managed_run(environment: dict[str, str]) -> tuple[Path, str]:
    run_id = environment.get("PROJECT4_RUN_ID", "")
    run_token = environment.get("PROJECT4_RUN_TOKEN", "")
    raw_run_dir = environment.get("PROJECT4_RUN_DIR", "")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not run_id or not run_token or not raw_run_dir:
        raise RuntimeError("official SFT must run inside scripts/run_managed.sh")
    if len(visible.split(",")) != 1 or visible in {"0", "5"}:
        raise RuntimeError(
            "official SFT requires one managed stable physical GPU, excluding GPU0/GPU5"
        )
    run_dir = Path(raw_run_dir).resolve()
    if run_dir != (RUN_ROOT / run_id).resolve() or not run_dir.is_dir():
        raise RuntimeError(f"unexpected managed Run directory: {run_dir}")
    return run_dir, visible


def validate_dataset(root: Path = DATASET_ROOT) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_relative_to((PROJECT_DATA / "datasets/processed").resolve()):
        raise ValueError("official SFT dataset escaped the processed data root")
    manifest_path = root / "manifest.json"
    info_path = root / "dataset_info.json"
    data_path = root / "wiki_en_official_1000.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {
        "dataset_name": DATASET_NAME,
        "sample_size": 1000,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "dataset_info_sha256": DATASET_INFO_SHA256,
        "selected_indices_sha256": INDICES_SHA256,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"dataset manifest {field} is not frozen value {expected!r}")
    if sha256_file(data_path) != DATASET_SHA256:
        raise ValueError("official SFT JSON no longer matches the frozen hash")
    if sha256_file(info_path) != DATASET_INFO_SHA256:
        raise ValueError("official SFT dataset_info no longer matches the frozen hash")
    if 1900 in manifest.get("selected_indices", []):
        raise ValueError("incomplete official source row 1900 entered the training subset")
    with info_path.open(encoding="utf-8") as handle:
        info = json.load(handle)
    entry = info.get(DATASET_NAME)
    if not isinstance(entry, dict) or entry.get("file_name") != data_path.name:
        raise ValueError("dataset_info does not point to the frozen official SFT JSON")
    if len(list(root.glob("images/**/*"))) < int(manifest["selected_image_files"]):
        raise ValueError("official SFT image payload is incomplete")
    return manifest


def validate_checkpoint(
    raw_checkpoint: Path | None,
    max_steps: int,
    dataset_manifest_sha256: str,
) -> Path | None:
    if raw_checkpoint is None:
        return None
    checkpoint = raw_checkpoint.resolve()
    if not checkpoint.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("resume checkpoint must belong to a project4 managed Run")
    state_path = checkpoint / "trainer_state.json"
    adapter_path = checkpoint / "adapter_model.safetensors"
    if not state_path.is_file() or not adapter_path.is_file():
        raise FileNotFoundError("resume checkpoint lacks trainer state or LoRA adapter")
    previous_step = int(json.loads(state_path.read_text()).get("global_step", -1))
    if previous_step < 1 or max_steps <= previous_step:
        raise ValueError(
            f"max_steps={max_steps} must exceed checkpoint global_step={previous_step}"
        )
    provenance_path = checkpoint.parents[1] / "official-sft-provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError("resume checkpoint lacks official SFT provenance")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise ValueError("resume checkpoint belongs to a different dataset manifest")
    if provenance.get("training_profile") != "official-wiki-en-lora-v1":
        raise ValueError("resume checkpoint belongs to a different training profile")
    return checkpoint


def training_config(
    run_dir: Path, max_steps: int, checkpoint: Path | None
) -> dict[str, object]:
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
        "dataset": DATASET_NAME,
        "dataset_dir": str(DATASET_ROOT),
        "template": "qwen3_vl",
        "cutoff_len": CUTOFF_LEN,
        "max_samples": 1000,
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
        "learning_rate": 0.0001,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "optim": "adamw_torch",
        "max_steps": max_steps,
        "bf16": True,
        "seed": 42,
        "data_seed": 42,
        "ddp_timeout": 600,
    }
    if checkpoint is not None:
        config["resume_from_checkpoint"] = str(checkpoint)
    return config


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_steps <= 50:
        raise ValueError("official SFT pilot is bounded to 1..50 optimizer steps")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    manifest = validate_dataset()
    manifest_hash = sha256_file(DATASET_ROOT / "manifest.json")
    checkpoint = validate_checkpoint(
        args.resume_from_checkpoint, args.max_steps, manifest_hash
    )
    output_dir = run_dir / "output"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite SFT output: {output_dir}")
    if not MODEL_ROOT.is_dir():
        raise FileNotFoundError(f"local base model is missing: {MODEL_ROOT}")

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
    config = training_config(run_dir, args.max_steps, checkpoint)
    config_path = run_dir / "official-sft-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    provenance = {
        "training_profile": "official-wiki-en-lora-v1",
        "source_is_official": True,
        "source_revision": SOURCE_REVISION,
        "dataset_root": str(DATASET_ROOT),
        "dataset_name": DATASET_NAME,
        "dataset_manifest_sha256": manifest_hash,
        "dataset_manifest": manifest,
        "base_model": str(MODEL_ROOT),
        "template": "qwen3_vl",
        "max_steps": args.max_steps,
        "resume_from_checkpoint": str(checkpoint) if checkpoint else None,
        "physical_gpu": physical_gpu,
        "method_deviations": {
            "full_finetuning_replaced_by_lora": True,
            "vision_tower_frozen": True,
            "projector_frozen": True,
            "cutoff_len": CUTOFF_LEN,
            "official_cutoff_len": 32000,
            "reason": "single RTX 4090 24GB bounded engineering reproduction",
        },
    }
    (run_dir / "official-sft-provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    cli = Path(sys.executable).with_name("llamafactory-cli")
    completed = subprocess.run([str(cli), "train", str(config_path)], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
