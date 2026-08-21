"""Parse and tokenize the explicitly synthetic agentic SFT smoke dataset."""

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
    "/media/imc/data/yzy/agent/project4-opensearch-vl-rl/datasets/processed/sft-smoke-v1"
)


def main() -> None:
    manifest = json.loads((DATASET_ROOT / "SYNTHETIC_MANIFEST.json").read_text())
    if (
        manifest.get("synthetic") is not True
        or manifest.get("purpose") != "pipeline-smoke-only"
    ):
        raise RuntimeError("dataset is not explicitly marked as synthetic smoke data")

    with tempfile.TemporaryDirectory(prefix="p4-sft-data-smoke.") as output_dir:
        model_args, data_args, training_args, _, _ = get_train_args(
            {
                "model_name_or_path": MODEL_ROOT,
                "stage": "sft",
                "do_train": True,
                "finetuning_type": "lora",
                "dataset": "sft_smoke_agentic",
                "dataset_dir": str(DATASET_ROOT),
                "template": "qwen3_vl",
                "cutoff_len": 1024,
                "max_samples": 4,
                "preprocessing_num_workers": 1,
                "output_dir": output_dir,
                "overwrite_cache": True,
                "report_to": "none",
            }
        )
        tokenizer_module = load_tokenizer(model_args)
        tokenizer = tokenizer_module["tokenizer"]
        template = get_template_and_fix_tokenizer(tokenizer, data_args)
        dataset_module = get_dataset(
            template,
            model_args,
            data_args,
            training_args,
            stage="sft",
            **tokenizer_module,
        )
        train_dataset = dataset_module["train_dataset"]
        if len(train_dataset) != 4:
            raise RuntimeError(
                f"expected 4 synthetic records, got {len(train_dataset)}"
            )
        sample = train_dataset[0]
        supervised_tokens = sum(label != IGNORE_INDEX for label in sample["labels"])
        if supervised_tokens == 0:
            raise RuntimeError("synthetic sample has no supervised tokens")

        print(
            json.dumps(
                {
                    "records": len(train_dataset),
                    "sample_tokens": len(sample["input_ids"]),
                    "sample_supervised_tokens": supervised_tokens,
                    "sample_images": len(sample["images"] or []),
                    "synthetic": True,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
