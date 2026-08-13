#!/usr/bin/env python3
"""Build a compact random-access offset table for the prepared JSONL corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_offsets(corpus_path: Path, output_path: Path, expected_rows: int) -> dict:
    partial = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite offsets output or partial: {output_path}")
    started = time.monotonic()
    offsets = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint64, shape=(expected_rows + 1,))
    offsets[0] = 0
    position = 0
    rows = 0
    with corpus_path.open("rb") as corpus:
        for rows, line in enumerate(corpus, start=1):
            if rows > expected_rows:
                raise RuntimeError("corpus has more rows than the preparation manifest")
            position += len(line)
            offsets[rows] = position
    if rows != expected_rows or position != corpus_path.stat().st_size:
        raise RuntimeError(
            f"corpus offset mismatch: rows={rows}/{expected_rows}, bytes={position}/{corpus_path.stat().st_size}"
        )
    offsets.flush()
    del offsets
    partial.replace(output_path)
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "rows": rows,
        "sentinel": position,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.resource_root.resolve()
    prepared = root / "prepared"
    prepare_manifest = json.loads((prepared / "prepare-complete.json").read_text())
    corpus_path = prepared / "wiki-18.jsonl"
    output_path = prepared / "wiki-18.offsets.npy"
    complete_path = prepared / "offsets-complete.json"
    if complete_path.exists():
        raise FileExistsError(f"refusing to overwrite completed offsets: {complete_path}")
    result = build_offsets(corpus_path, output_path, prepare_manifest["corpus"]["rows"])
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": prepare_manifest["corpus"]["sha256"],
        "offsets": result,
        "next_gate": "start localhost-only CPU retriever and validate HTTP requests",
    }
    complete_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
