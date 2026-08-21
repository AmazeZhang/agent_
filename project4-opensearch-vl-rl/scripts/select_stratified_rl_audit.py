#!/usr/bin/env python3
"""Select a deterministic, answer-independent stratified RL audit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", default="opensearch-vl-offline-audit-v1")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_strata(path: Path) -> dict[str, list[int]]:
    strata: dict[str, list[int]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {row_index + 1}: {error}") from error
            if not isinstance(item, dict) or not str(item.get("dataset", "")).strip():
                raise ValueError(f"row {row_index} has no non-empty dataset field")
            strata[str(item["dataset"])].append(row_index)
    if not strata:
        raise ValueError("input contains no data rows")
    return dict(strata)


def proportional_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    if sample_size <= 0 or sample_size > total:
        raise ValueError(f"sample_size must be between 1 and {total}")
    exact = {name: sample_size * count / total for name, count in counts.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remaining = sample_size - sum(quotas.values())
    order = sorted(counts, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def stable_priority(seed: str, dataset: str, row_index: int) -> bytes:
    payload = f"{seed}\0{dataset}\0{row_index}".encode()
    return hashlib.sha256(payload).digest()


def select_rows(
    strata: dict[str, list[int]], quotas: dict[str, int], seed: str
) -> list[dict[str, int | str]]:
    selected = []
    for dataset, row_indices in strata.items():
        quota = quotas[dataset]
        ranked = sorted(
            row_indices, key=lambda row: (stable_priority(seed, dataset, row), row)
        )
        for row_index in ranked[:quota]:
            sample_id = hashlib.sha256(
                f"{dataset}\0{row_index}".encode()
            ).hexdigest()[:16]
            selected.append(
                {
                    "sample_id": sample_id,
                    "row_index": row_index,
                    "dataset": dataset,
                }
            )
    return sorted(selected, key=lambda item: int(item["row_index"]))


def build_manifest(path: Path, sample_size: int, seed: str) -> dict[str, object]:
    strata = load_strata(path)
    counts = {name: len(rows) for name, rows in strata.items()}
    quotas = proportional_quotas(counts, sample_size)
    samples = select_rows(strata, quotas, seed)
    selected_counts = Counter(str(item["dataset"]) for item in samples)
    return {
        "schema_version": 1,
        "source": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "selection": {
            "policy": "dataset-proportional-largest-remainder-sha256-priority",
            "seed": seed,
            "uses_answer_for_selection": False,
            "sample_size": sample_size,
        },
        "source_dataset_counts": dict(sorted(counts.items())),
        "selected_dataset_counts": dict(sorted(selected_counts.items())),
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(input_path, args.sample_size, args.seed)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(manifest["selected_dataset_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
