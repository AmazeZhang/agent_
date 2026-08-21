#!/usr/bin/env python3
"""Re-embed one WIT pilot shard with the fixed query-time image encoder."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from local_retrieval import (  # noqa: E402
    build_exact_index,
    encode_pil_images,
    load_resnet50_v1,
)
from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402

from build_wit_pilot import load_wit_records  # noqa: E402

REQUIRED_RUN_ENV = ("PROJECT4_RUN_ID", "PROJECT4_RUN_DIR", "PROJECT4_RUN_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--weights-sha256-prefix", default="0676ba61")
    parser.add_argument("--read-batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    return parser.parse_args()


def require_managed_environment(environment: Mapping[str, str]) -> None:
    missing = [key for key in REQUIRED_RUN_ENV if not environment.get(key)]
    if missing:
        raise RuntimeError(f"GPU encoding requires a managed Project 4 Run: {missing}")


def decode_image(value: dict[str, object] | None) -> Image.Image:
    image_bytes = (value or {}).get("bytes")
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise ValueError("WIT image row contains no embedded bytes")
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        return image.convert("RGB")


def reembed(
    input_path: Path,
    output: Path,
    weights_path: Path,
    *,
    source_sha256: str,
    corpus_revision: str,
    weights_sha256_prefix: str,
    read_batch_size: int,
    encode_batch_size: int,
) -> Path:
    import torch

    require_managed_environment(os.environ)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "managed encoder requires exactly one visible CUDA device; "
            f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
        )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite re-embedded WIT pilot: {output}")
    if not 1 <= read_batch_size <= 1024:
        raise ValueError("read_batch_size must be between 1 and 1024")
    if not 1 <= encode_batch_size <= read_batch_size:
        raise ValueError("encode_batch_size must be between 1 and read_batch_size")
    actual_source_sha256 = sha256_file(input_path)
    if actual_source_sha256 != source_sha256.lower():
        raise ValueError(
            f"WIT shard SHA256 mismatch: {actual_source_sha256}/{source_sha256}"
        )

    _, visual_metadata, _ = load_wit_records(
        input_path,
        expected_dimension=2048,
        batch_size=read_batch_size,
        revision=corpus_revision,
    )
    vectors = np.empty((len(visual_metadata), 2048), dtype=np.float32)
    device = "cuda:0"
    model, preprocess, encoder_info = load_resnet50_v1(
        weights_path, device=device, sha256_prefix=weights_sha256_prefix
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    offset = 0
    parquet = pq.ParquetFile(input_path)
    for batch in parquet.iter_batches(batch_size=read_batch_size, columns=["image"]):
        images = [decode_image(row["image"]) for row in batch.to_pylist()]
        for start in range(0, len(images), encode_batch_size):
            encoded = encode_pil_images(
                model,
                preprocess,
                images[start : start + encode_batch_size],
                device=device,
            )
            destination = offset + start
            vectors[destination : destination + len(encoded)] = encoded
        offset += len(images)
        print(f"encoded={offset}/{len(visual_metadata)}", flush=True)
    if offset != len(visual_metadata) or not np.isfinite(vectors).all():
        raise RuntimeError("re-embedded vector count mismatch or non-finite values")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite re-embedding staging: {staging}")
    staging.mkdir()
    try:
        encoder_revision = encoder_info["weights_sha256"]
        build_exact_index(
            staging / "visual",
            vectors,
            visual_metadata,
            corpus="wikimedia/wit_base:train-00000-of-00330:torchvision-resnet50-v1",
            corpus_revision=f"{corpus_revision}+{encoder_revision}",
        )
        manifest = {
            "schema_version": 1,
            "source_path": str(input_path.resolve()),
            "source_sha256": actual_source_sha256,
            "corpus_revision": corpus_revision,
            "count": len(visual_metadata),
            "dimension": 2048,
            "encoder": encoder_info,
            "physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "logical_device": device,
            "device_name": torch.cuda.get_device_name(0),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        if output.exists():
            raise FileExistsError(
                f"destination appeared during re-embedding; preserved {staging}"
            )
        staging.rename(output)
    except Exception as error:
        raise RuntimeError(f"WIT re-embedding failed; staging preserved at {staging}") from error
    return output


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    weights_path = args.weights.resolve()
    output = args.output.resolve()
    if not input_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("input shard and encoder weights must exist")
    built = reembed(
        input_path,
        output,
        weights_path,
        source_sha256=args.source_sha256,
        corpus_revision=args.corpus_revision,
        weights_sha256_prefix=args.weights_sha256_prefix,
        read_batch_size=args.read_batch_size,
        encode_batch_size=args.encode_batch_size,
    )
    print(f"re-embedded WIT pilot: {built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
