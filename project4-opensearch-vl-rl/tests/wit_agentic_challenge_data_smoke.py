"""Parse all challenge-train trajectories with the real LLaMA Factory template."""

import json
import tempfile
from pathlib import Path

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer

MODEL_ROOT = (
    "/media/imc/data/yzy/agent/project4-opensearch-vl-rl/models/Qwen3-VL-8B-Instruct"
)
DATASET_ROOT = Path(
    "/media/imc/data/yzy/agent/project4-opensearch-vl-rl/datasets/processed/"
    "wit-agentic-challenge-v5"
)


def main() -> None:
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text())
    if (
        manifest.get("status") != "challenge-ready"
        or manifest.get("network_required") is not False
        or manifest.get("split_counts", {}).get("train") != 80
    ):
        raise RuntimeError("dataset is not the fixed offline challenge")

    with tempfile.TemporaryDirectory(prefix="p4-wit-challenge-smoke.") as output_dir:
        model_args, data_args, training_args, _, _ = get_train_args(
            {
                "model_name_or_path": MODEL_ROOT,
                "stage": "sft",
                "do_train": True,
                "finetuning_type": "lora",
                "dataset": "wit_agentic_train_v1",
                "dataset_dir": str(DATASET_ROOT),
                "template": "qwen3_vl_nothink",
                "cutoff_len": 2048,
                "max_samples": 80,
                "preprocessing_num_workers": 1,
                "output_dir": output_dir,
                "overwrite_cache": True,
                "report_to": "none",
            }
        )
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        dataset = get_dataset(
            template,
            model_args,
            data_args,
            training_args,
            stage="sft",
            **tokenizer_module,
        )["train_dataset"]
        if len(dataset) != 80:
            raise RuntimeError(f"expected 80 records, got {len(dataset)}")
        lengths = [len(sample["input_ids"]) for sample in dataset]
        supervised = [
            sum(label != IGNORE_INDEX for label in sample["labels"])
            for sample in dataset
        ]
        image_counts = [len(sample["images"] or []) for sample in dataset]
        if min(supervised) <= 0 or set(image_counts) != {1}:
            raise RuntimeError(
                "challenge has an unsupervised or non-single-image record"
            )
        if max(lengths) >= 2048:
            raise RuntimeError(
                "challenge trajectory reaches cutoff and may be truncated"
            )
        print(
            json.dumps(
                {
                    "records": len(dataset),
                    "tokens_min_max": [min(lengths), max(lengths)],
                    "supervised_tokens_min_max": [min(supervised), max(supervised)],
                    "images_per_record": 1,
                    "template": "qwen3_vl_nothink",
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
