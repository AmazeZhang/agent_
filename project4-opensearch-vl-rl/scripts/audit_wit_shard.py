#!/usr/bin/env python3
"""Audit one WIT Parquet shard without exporting image bytes or embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarise_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, str):
        return {"type": "string", "length": len(value), "preview": value[:160]}
    if isinstance(value, dict):
        return {
            "type": "object",
            "fields": {
                str(key): summarise_value(item) for key, item in sorted(value.items())
            },
        }
    if isinstance(value, (list, tuple)):
        summary: dict[str, Any] = {"type": "array", "length": len(value)}
        if value and all(isinstance(item, (int, float)) for item in value):
            numbers = [float(item) for item in value]
            finite = [number for number in numbers if math.isfinite(number)]
            summary.update(
                {
                    "finite_values": len(finite),
                    "minimum": min(finite, default=None),
                    "maximum": max(finite, default=None),
                    "l2_norm": math.sqrt(sum(number * number for number in finite)),
                }
            )
        elif value:
            summary["first"] = summarise_value(value[0])
        return summary
    if isinstance(value, (bool, int, float)):
        return {"type": type(value).__name__, "value": value}
    return {"type": type(value).__name__, "preview": str(value)[:160]}


def audit_shard(path: Path, sample_rows: int) -> dict[str, Any]:
    if not 1 <= sample_rows <= 10:
        raise ValueError("sample_rows must be between 1 and 10")
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    codecs: set[str] = set()
    compressed_bytes = uncompressed_bytes = 0
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            codecs.add(column.compression)
            compressed_bytes += column.total_compressed_size
            uncompressed_bytes += column.total_uncompressed_size

    batch = next(parquet.iter_batches(batch_size=sample_rows), None)
    samples = []
    if batch is not None:
        for row in batch.to_pylist():
            samples.append(
                {str(key): summarise_value(value) for key, value in row.items()}
            )
    return {
        "schema_version": 1,
        "source": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "parquet": {
            "created_by": metadata.created_by,
            "format_version": metadata.format_version,
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
            "columns": metadata.num_columns,
            "serialized_metadata_bytes": metadata.serialized_size,
            "column_compressed_bytes": compressed_bytes,
            "column_uncompressed_bytes": uncompressed_bytes,
            "compression_codecs": sorted(codecs),
        },
        "fields": [{"name": field.name, "type": str(field.type)} for field in schema],
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_path}")
    report = audit_shard(input_path, args.sample_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report["parquet"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
