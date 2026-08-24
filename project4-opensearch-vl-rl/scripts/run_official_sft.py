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
    / "datasets/processed/search-vl-sft-wiki-en-official-960-safe-v2-r2c1c460-c5120"
)
RUN_ROOT = PROJECT_DATA / "runs"
DATASET_NAME = "wiki_en_official_960_safe"
SOURCE_REVISION = "2c1c460af4fa15bd63210cbf426a96664b959944"
SOURCE_SHA256 = "a22a44c6a04d79d6dfd0064c89d8a792045278eed70a8e27c14b7c5e2f4850e3"
DATASET_SHA256 = "571c9c59a02309e8962d10ac0a0fdb14d86aa2c54cd8c9f86f4cfcbfa8e964a5"
DATASET_INFO_SHA256 = "21d191168c39087f5bdc26081bf70432f38da9af13345c9447f1b0c13f10958a"
INDICES_SHA256 = "e11f2f3e74e18a05e5c4d57e57d7557b57aacc9eef4f86ec60a53e82a06d5bd0"
ALIGNMENT_REPORT_SHA256 = "d39cee732c158005c020a6117ec8dfae2b1200116458933d5f47febf59fc66e6"
CUTOFF_LEN = 5120
QUANTIZATION_METHOD = "bnb"
TRAINING_PROFILE = "official-wiki-en-safe960-qlora-v4"


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
    data_path = root / "wiki_en_official_960_safe.json"
    alignment_path = root / "alignment-audit-cutoff5120-v1.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {
        "dataset_name": DATASET_NAME,
        "sample_size": 960,
        "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "dataset_info_sha256": DATASET_INFO_SHA256,
        "selected_indices_sha256": INDICES_SHA256,
        "cutoff_len": CUTOFF_LEN,
        "rows_modified": 0,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"dataset manifest {field} is not frozen value {expected!r}")
    if sha256_file(data_path) != DATASET_SHA256:
        raise ValueError("official SFT JSON no longer matches the frozen hash")
    if sha256_file(info_path) != DATASET_INFO_SHA256:
        raise ValueError("official SFT dataset_info no longer matches the frozen hash")
    if sha256_file(alignment_path) != ALIGNMENT_REPORT_SHA256:
        raise ValueError("official SFT alignment report no longer matches the frozen hash")
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    if alignment.get("checked") != 960 or alignment.get("mismatch_count") != 0:
        raise ValueError("official SFT alignment report does not prove all 960 rows safe")
    if alignment.get("zero_supervision_count") != 0:
        raise ValueError("official SFT alignment report contains rows without supervision")
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
    if provenance.get("training_profile") != TRAINING_PROFILE:
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
        "quantization_bit": 4,
        "quantization_method": QUANTIZATION_METHOD,
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
        "cutoff_len": CUTOFF_LEN,
        "max_samples": 960,
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


def validate_training_config(config: dict[str, object]) -> None:
    """Fail closed if the frozen QLoRA contract would silently load BF16 weights."""
    if config.get("quantization_bit") != 4:
        raise ValueError("official SFT requires a 4-bit frozen base")
    if config.get("quantization_method") != QUANTIZATION_METHOD:
        raise ValueError(
            "this LLaMA Factory revision only accepts quantization_method='bnb'"
        )
    if (
        config.get("quantization_type") != "nf4"
        or config.get("double_quantization") is not True
    ):
        raise ValueError("official SFT requires NF4 double quantization")


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
    validate_training_config(config)
    config_path = run_dir / "official-sft-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    provenance = {
        "training_profile": TRAINING_PROFILE,
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
            "frozen_base_quantized_to_4bit_nf4": True,
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
