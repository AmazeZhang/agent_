#!/usr/bin/env python3
"""Prepare deterministic, transformed WIT query images for an agentic pilot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from build_wit_pilot import choose_evidence, sha256_file  # noqa: E402

PROJECT_DATA_ROOT = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
REQUIRED_COLUMNS = {
    "image",
    "image_url",
    "metadata_url",
    "caption_attribution_description",
    "wit_features",
}
SPLIT_ORDER = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--dev", type=int, default=20)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--seed", default="wit-agentic-pilot-v1")
    return parser.parse_args()


def stable_rank(seed: str, revision: str, row_index: int) -> str:
    payload = f"{seed}\0{revision}\0{row_index}".encode()
    return hashlib.sha256(payload).hexdigest()


def usable_evidence(evidence: dict[str, str]) -> bool:
    title = evidence["title"].strip()
    summary = " ".join(evidence["summary"].split())
    return (
        evidence["language"] == "en"
        and "wikipedia.org/wiki/" in evidence["source"]
        and 2 <= len(title) <= 120
        and title.casefold() != summary.casefold()
        and 80 <= len(summary) <= 500
    )


def iter_candidates(
    path: Path, *, revision: str, seed: str, batch_size: int = 512
) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    missing = sorted(REQUIRED_COLUMNS - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"WIT shard missing required columns: {missing}")
    offset = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=sorted(REQUIRED_COLUMNS)
    ):
        for local_index, row in enumerate(batch.to_pylist()):
            row_index = offset + local_index
            evidence = choose_evidence(
                row["wit_features"],
                caption=str(row["caption_attribution_description"] or ""),
                image_url=str(row["image_url"] or ""),
                metadata_url=str(row["metadata_url"] or ""),
            )
            image = row.get("image") or {}
            image_bytes = image.get("bytes") if isinstance(image, dict) else None
            if not usable_evidence(evidence) or not image_bytes:
                continue
            yield {
                "row_index": row_index,
                "entity_id": f"wit:{revision[:12]}:{row_index:08d}",
                "rank": stable_rank(seed, revision, row_index),
                "image_bytes": image_bytes,
                "title": evidence["title"].strip(),
                "source": evidence["source"].strip(),
                "summary": " ".join(evidence["summary"].split()),
            }
        offset += len(batch)


def transform_query(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        rgb = image.convert("RGB")
    width, height = rgb.size
    if min(width, height) < 64:
        raise ValueError(f"source image is too small for query transform: {rgb.size}")
    margin_x = max(1, width // 20)
    margin_y = max(1, height // 20)
    return rgb.crop((margin_x, margin_y, width - margin_x, height - margin_y))


def assign_splits(
    candidates: list[dict[str, Any]], split_counts: dict[str, int]
) -> list[tuple[str, dict[str, Any]]]:
    if any(count < 1 for count in split_counts.values()):
        raise ValueError("every split must contain at least one item")
    total = sum(split_counts.values())
    ordered = sorted(candidates, key=lambda item: (item["rank"], item["row_index"]))
    if len(ordered) < total:
        raise ValueError(f"only {len(ordered)} usable candidates for {total} requested")
    assigned = []
    offset = 0
    for split in SPLIT_ORDER:
        count = split_counts[split]
        assigned.extend((split, item) for item in ordered[offset : offset + count])
        offset += count
    return assigned


def prepare(
    input_path: Path,
    output: Path,
    *,
    revision: str,
    source_sha256: str,
    split_counts: dict[str, int],
    seed: str,
) -> Path:
    root = PROJECT_DATA_ROOT.resolve()
    destination = output.resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"output must be below {root}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite candidate dataset: {destination}")
    actual_sha256 = sha256_file(input_path)
    if actual_sha256 != source_sha256:
        raise ValueError(f"WIT shard SHA256 mismatch: {actual_sha256}/{source_sha256}")
    selected = assign_splits(
        list(iter_candidates(input_path, revision=revision, seed=seed)), split_counts
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {staging}")
    (staging / "images").mkdir(parents=True)
    records = []
    try:
        for split, candidate in selected:
            task_id = f"wit-{candidate['row_index']:08d}"
            relative_image = Path("images") / f"{task_id}.jpg"
            query = transform_query(candidate.pop("image_bytes"))
            query.save(staging / relative_image, format="JPEG", quality=92)
            records.append(
                {
                    "task_id": task_id,
                    "split": split,
                    "query_image": relative_image.as_posix(),
                    "query_transform": "center_crop_90_percent_then_jpeg_q92",
                    "row_index": candidate["row_index"],
                    "entity_id": candidate["entity_id"],
                    "title": candidate["title"],
                    "source": candidate["source"],
                    "evidence": candidate["summary"],
                }
            )
        with (staging / "candidates.jsonl").open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        manifest = {
            "schema_version": 1,
            "status": "retrieval-unverified",
            "purpose": "local-agentic-sft-rl-pilot",
            "source_path": str(input_path.resolve()),
            "source_sha256": actual_sha256,
            "corpus_revision": revision,
            "selection_seed": seed,
            "selection_uses_model_or_gold_answer": False,
            "split_unit": "entity_id",
            "split_counts": split_counts,
            "query_transform": "center_crop_90_percent_then_jpeg_q92",
            "records": len(records),
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        if destination.exists():
            raise FileExistsError(f"destination appeared during build: {destination}")
        staging.rename(destination)
    except Exception as error:
        raise RuntimeError(f"candidate build failed; staging preserved at {staging}") from error
    return destination


def main() -> int:
    args = parse_args()
    output = prepare(
        args.input.resolve(),
        args.output,
        revision=args.corpus_revision,
        source_sha256=args.source_sha256.lower(),
        split_counts={"train": args.train, "dev": args.dev, "test": args.test},
        seed=args.seed,
    )
    print(json.dumps({"output": str(output), "status": "retrieval-unverified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
