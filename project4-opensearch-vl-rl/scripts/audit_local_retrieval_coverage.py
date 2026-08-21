#!/usr/bin/env python3
"""Run answer-independent local image retrieval on a fixed RL audit manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from local_retrieval import (  # noqa: E402
    ExactVisualIndex,
    encode_pil_images,
    load_resnet50_v1,
)
from local_retrieval.image_search_backend import resolve_local_image  # noqa: E402
from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402

from reembed_wit_pilot import require_managed_environment  # noqa: E402

CONFIDENCE_THRESHOLDS = {"high": 0.75, "medium": 0.60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_selection(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("selection", {}).get("uses_answer_for_selection") is not False:
        raise ValueError("selection must explicitly record uses_answer_for_selection=false")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("selection contains no samples")
    return manifest


def load_selected_rows(
    dataset: Path, selection: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_sha256 = selection.get("source", {}).get("sha256")
    actual_sha256 = sha256_file(dataset)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"selection/dataset SHA256 mismatch: {expected_sha256}/{actual_sha256}")
    requested = {int(sample["row_index"]): sample for sample in selection["samples"]}
    found: dict[int, dict[str, Any]] = {}
    with dataset.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index not in requested:
                continue
            item = json.loads(line)
            sample = requested[row_index]
            if str(item.get("dataset")) != str(sample["dataset"]):
                raise ValueError(f"dataset stratum mismatch at row {row_index}")
            images = item.get("images")
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError(f"selected row {row_index} must contain exactly one image")
            found[row_index] = {
                "sample_id": str(sample["sample_id"]),
                "row_index": row_index,
                "dataset": str(sample["dataset"]),
                "question": str(item.get("question", "")),
                "image_reference": str(images[0]),
            }
    missing = sorted(set(requested) - set(found))
    if missing:
        raise ValueError(f"selected dataset rows are missing: {missing[:10]}")
    return [found[int(sample["row_index"])] for sample in selection["samples"]]


def confidence_label(similarity: float) -> str:
    if similarity >= CONFIDENCE_THRESHOLDS["high"]:
        return "high"
    if similarity >= CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


def summarise_scores(scores: list[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    labels = [confidence_label(score) for score in scores]
    return {
        "count": len(scores),
        "minimum": round(float(np.min(values)), 8),
        "p50": round(float(np.percentile(values, 50)), 8),
        "p90": round(float(np.percentile(values, 90)), 8),
        "maximum": round(float(np.max(values)), 8),
        "mean": round(float(np.mean(values)), 8),
        "confidence_proxy_counts": {
            label: labels.count(label) for label in ("high", "medium", "low")
        },
    }


def run_audit(
    dataset: Path,
    selection_path: Path,
    image_root: Path,
    index_root: Path,
    weights_path: Path,
    *,
    encode_batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    import torch

    require_managed_environment(os.environ)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("coverage audit requires exactly one managed visible CUDA device")
    if not 1 <= encode_batch_size <= 128:
        raise ValueError("encode_batch_size must be between 1 and 128")
    selection = load_selection(selection_path)
    rows = load_selected_rows(dataset, selection)
    index = ExactVisualIndex(index_root)
    model, preprocess, encoder = load_resnet50_v1(weights_path, device="cuda:0")
    if not str(index.manifest["corpus_revision"]).endswith(
        f"+{encoder['weights_sha256']}"
    ):
        raise ValueError("coverage index and query encoder revisions do not match")

    vectors = np.empty((len(rows), index.vectors.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), encode_batch_size):
        batch_rows = rows[start : start + encode_batch_size]
        images = []
        for row in batch_rows:
            path = resolve_local_image(Path(row["image_reference"]), image_root)
            with Image.open(path) as image:
                image.load()
                images.append(image.convert("RGB"))
        encoded = encode_pil_images(
            model, preprocess, images, device="cuda:0"
        )
        vectors[start : start + len(encoded)] = encoded
        print(f"encoded_audit={start + len(encoded)}/{len(rows)}", flush=True)

    search_results = index.search_batch(vectors, top_k=top_k)
    samples = []
    scores_by_dataset: dict[str, list[float]] = defaultdict(list)
    for row, results in zip(rows, search_results, strict=True):
        top1 = float(results[0]["similarity"])
        scores_by_dataset[row["dataset"]].append(top1)
        samples.append(
            {
                **row,
                "top1_similarity": top1,
                "confidence_proxy": confidence_label(top1),
                "results": results,
            }
        )
    all_scores = [sample["top1_similarity"] for sample in samples]
    return {
        "schema_version": 1,
        "answer_used_for_selection_or_retrieval": False,
        "interpretation": (
            "Similarity buckets are an answer-independent confidence proxy, not semantic "
            "coverage or answer correctness. Candidate relevance requires separate audit."
        ),
        "thresholds": CONFIDENCE_THRESHOLDS,
        "source": {
            "dataset_path": str(dataset.resolve()),
            "dataset_sha256": sha256_file(dataset),
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "image_root": str(image_root.resolve()),
            "index_root": str(index_root.resolve()),
            "index_revision": index.manifest["corpus_revision"],
            "encoder_weights_sha256": encoder["weights_sha256"],
        },
        "overall": summarise_scores(all_scores),
        "by_dataset": {
            dataset_name: summarise_scores(scores)
            for dataset_name, scores in sorted(scores_by_dataset.items())
        },
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite coverage report: {output}")
    report = run_audit(
        args.dataset.resolve(),
        args.selection.resolve(),
        args.image_root.resolve(),
        args.index.resolve(),
        args.weights.resolve(),
        encode_batch_size=args.encode_batch_size,
        top_k=args.top_k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"overall": report["overall"], "by_dataset": report["by_dataset"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
