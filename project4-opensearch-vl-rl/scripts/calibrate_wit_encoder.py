#!/usr/bin/env python3
"""Test whether a fixed local ResNet-50 reproduces WIT-published embeddings."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=16)
    parser.add_argument("--weights-sha256-prefix", default="0676ba61")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def alignment_metrics(
    published: np.ndarray, computed: np.ndarray
) -> dict[str, float | list[int] | bool]:
    first = np.asarray(published, dtype=np.float64)
    second = np.asarray(computed, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(f"embedding shape mismatch: {first.shape}/{second.shape}")
    first_norms = np.linalg.norm(first, axis=1)
    second_norms = np.linalg.norm(second, axis=1)
    if np.any(first_norms == 0) or np.any(second_norms == 0):
        raise ValueError("embeddings contain a zero-norm row")
    first_normalised = first / first_norms[:, None]
    second_normalised = second / second_norms[:, None]
    paired_cosine = np.sum(first_normalised * second_normalised, axis=1)
    cross_similarity = second_normalised @ first_normalised.T
    top1 = np.argmax(cross_similarity, axis=1)
    identity_rate = float(np.mean(top1 == np.arange(len(top1))))
    relative_l2 = np.linalg.norm(first - second, axis=1) / first_norms
    mean_cosine = float(np.mean(paired_cosine))
    return {
        "paired_cosine_mean": mean_cosine,
        "paired_cosine_minimum": float(np.min(paired_cosine)),
        "paired_cosine_maximum": float(np.max(paired_cosine)),
        "relative_l2_mean": float(np.mean(relative_l2)),
        "identity_top1_rate": identity_rate,
        "identity_top1_indices": top1.tolist(),
        "aligned": mean_cosine >= 0.99 and identity_rate >= 0.95,
    }


def load_samples(path: Path, sample_rows: int) -> tuple[list[Image.Image], np.ndarray]:
    if not 2 <= sample_rows <= 64:
        raise ValueError("sample_rows must be between 2 and 64")
    batch = next(
        pq.ParquetFile(path).iter_batches(
            batch_size=sample_rows, columns=["image", "embedding"]
        ),
        None,
    )
    if batch is None or len(batch) < 2:
        raise ValueError("WIT shard contains fewer than two sample rows")
    images = []
    published = []
    for row in batch.to_pylist():
        image_value = row["image"] or {}
        image_bytes = image_value.get("bytes")
        if not image_bytes:
            raise ValueError("sample image contains no embedded bytes")
        with Image.open(io.BytesIO(image_bytes)) as image:
            images.append(image.convert("RGB"))
        published.append(row["embedding"])
    return images, np.asarray(published, dtype=np.float32)


def calibrate(
    shard: Path,
    weights_path: Path,
    *,
    sample_rows: int,
    weights_sha256_prefix: str,
) -> dict[str, Any]:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights_sha256 = sha256_file(weights_path)
    if not weights_sha256.startswith(weights_sha256_prefix.lower()):
        raise ValueError(
            f"weight SHA256 prefix mismatch: {weights_sha256}/{weights_sha256_prefix}"
        )
    images, published = load_samples(shard, sample_rows)
    weight_spec = ResNet50_Weights.IMAGENET1K_V1
    preprocess = weight_spec.transforms()
    model = resnet50(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval()
    batch = torch.stack([preprocess(image) for image in images])
    with torch.inference_mode():
        computed = model(batch).cpu().numpy()
    metrics = alignment_metrics(published, computed)
    return {
        "schema_version": 1,
        "source": {
            "shard_path": str(shard.resolve()),
            "shard_sha256": sha256_file(shard),
            "sample_rows": len(images),
        },
        "encoder": {
            "implementation": "torchvision.models.resnet50",
            "weights": "ResNet50_Weights.IMAGENET1K_V1",
            "weights_url": weight_spec.url,
            "weights_path": str(weights_path.resolve()),
            "weights_sha256": weights_sha256,
            "preprocess": str(preprocess),
            "device": "cpu",
        },
        "metrics": metrics,
        "decision": (
            "published-space-compatible"
            if metrics["aligned"]
            else "do-not-mix-computed-queries-with-published-candidates"
        ),
    }


def main() -> int:
    args = parse_args()
    shard = args.input.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    if not shard.is_file() or not weights.is_file():
        raise FileNotFoundError("input shard and weights must both exist")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    report = calibrate(
        shard,
        weights,
        sample_rows=args.sample_rows,
        weights_sha256_prefix=args.weights_sha256_prefix,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(f"decision: {report['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
