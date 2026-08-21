"""Create a tiny, explicitly synthetic agentic VL dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

DATA_ROOT = Path(
    "/media/imc/data/yzy/agent/project4-opensearch-vl-rl/datasets/processed"
)
COLORS = ("red", "blue", "green", "yellow")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tool_schema() -> str:
    return json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "crop",
                    "description": "Crop a rectangular region from an image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image": {"type": "string"},
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                        },
                        "required": ["image", "x", "y", "width", "height"],
                    },
                },
            }
        ],
        separators=(",", ":"),
    )


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError(f"output must be below {DATA_ROOT}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic dataset: {output}")

    images_dir = output / "images"
    images_dir.mkdir(parents=True)
    records = []
    for index, color in enumerate(COLORS):
        image = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((48, 48, 176, 176), fill=color)
        source_name = f"images/sample-{index}-source.png"
        crop_name = f"images/sample-{index}-crop.png"
        image.save(output / source_name)
        image.crop((48, 48, 176, 176)).save(output / crop_name)
        records.append(
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image> Inspect the center square and report its dominant color.",
                    },
                    {
                        "from": "function",
                        "value": json.dumps(
                            {
                                "name": "crop",
                                "arguments": {
                                    "image": "img_1",
                                    "x": 48,
                                    "y": 48,
                                    "width": 128,
                                    "height": 128,
                                },
                            },
                            separators=(",", ":"),
                        ),
                    },
                    {
                        "from": "observation",
                        "value": "<image> This is the cropped center region.",
                    },
                    {"from": "gpt", "value": f"The dominant color is {color}."},
                ],
                "images": [source_name, crop_name],
                "system": "Use the provided visual tool when useful, then answer concisely.",
                "tools": tool_schema(),
            }
        )

    data_path = output / "smoke.json"
    data_path.write_text(json.dumps(records, indent=2) + "\n")
    dataset_info = {
        "sft_smoke_agentic": {
            "file_name": "smoke.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
                "system": "system",
                "tools": "tools",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function",
            },
        }
    }
    info_path = output / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, indent=2) + "\n")

    generated_files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "synthetic": True,
        "purpose": "pipeline-smoke-only",
        "records": len(records),
        "files": [
            {
                "path": str(path.relative_to(output)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated_files
        ],
    }
    manifest_path = output / "SYNTHETIC_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "records": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
