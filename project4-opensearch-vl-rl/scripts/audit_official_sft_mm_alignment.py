"""Audit token/feature alignment for every example in the frozen official SFT subset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from llamafactory.data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from run_official_sft import DATASET_ROOT as DEFAULT_DATASET_ROOT
from run_official_sft import PROJECT_DATA, training_config, validate_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-name", default="wiki_en_official_1000")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.report is not None and args.report.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.report}")
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_relative_to((PROJECT_DATA / "datasets/processed").resolve()):
        raise ValueError("dataset root escaped the processed data root")

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

    config = training_config(Path("/tmp/p4-official-sft-alignment-audit"), 1, None)
    config.update(
        {
            "dataset": args.dataset_name,
            "dataset_dir": str(dataset_root),
            "max_samples": None,
        }
    )
    validate_training_config(config)
    model_args, data_args, training_args, _, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    if processor is None:
        raise RuntimeError("official SFT alignment audit requires the Qwen3-VL processor")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset = get_dataset(
        template,
        model_args,
        data_args,
        training_args,
        stage="sft",
        **tokenizer_module,
    )["train_dataset"]
    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=None,
        tokenizer=tokenizer,
        processor=processor,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation="flash_attention_2",
        compute_dtype=model_args.compute_dtype,
    )

    image_token_id = int(processor.image_token_id)
    merge_length = int(processor.image_processor.merge_size) ** 2
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    source_indices = manifest["selected_indices"]
    if len(source_indices) != len(dataset):
        raise ValueError("manifest selected_indices no longer aligns with the tokenized dataset")
    mismatches: list[dict[str, int]] = []
    zero_supervision: list[int] = []
    checked = min(len(dataset), args.limit or len(dataset))
    token_min: int | None = None
    token_max = 0
    for dataset_index in range(checked):
        row = dataset[dataset_index]
        image_count = len(row.get("images") or [])
        batch = collator([row])
        token_count = int((batch["input_ids"] == image_token_id).sum().item())
        grids = batch.get("image_grid_thw")
        feature_count = 0
        if grids is not None:
            feature_count = sum(int(math.prod(grid.tolist())) // merge_length for grid in grids)
        token_min = token_count if token_min is None else min(token_min, token_count)
        token_max = max(token_max, token_count)
        if token_count != feature_count:
            mismatches.append(
                {
                    "dataset_index": dataset_index,
                    "source_index": int(source_indices[dataset_index]),
                    "image_count": image_count,
                    "image_tokens": token_count,
                    "image_features": feature_count,
                    "sequence_length": len(row["input_ids"]),
                }
            )
        if not any(label != IGNORE_INDEX for label in row["labels"]):
            zero_supervision.append(dataset_index)

    report = {
        "checked": checked,
        "dataset_size": len(dataset),
        "dataset_name": args.dataset_name,
        "dataset_root": str(dataset_root),
        "cutoff_len": data_args.cutoff_len,
        "image_token_min": token_min,
        "image_token_max": token_max,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "zero_supervision_count": len(zero_supervision),
        "zero_supervision_indices": zero_supervision,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 1 if mismatches or zero_supervision else 0


if __name__ == "__main__":
    raise SystemExit(main())
