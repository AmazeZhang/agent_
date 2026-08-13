#!/usr/bin/env python3
"""Prepare the downloaded Search-R1 index and corpus without deleting sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


INDEX_PARTS = (
    ("index-download/part_aa", 42949672960, "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023"),
    ("index-download/part_ab", 21609402413, "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11"),
)
CORPUS_ARCHIVE = "corpus-download/wiki-18.jsonl.gz"
CORPUS_ARCHIVE_SIZE = 5123307260
CORPUS_ARCHIVE_SHA256 = "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"
CORPUS_MEMBER = "data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
CORPUS_MEMBER_SIZE = 14393573105
MIN_FREE_BYTES = 200 * 1024**3
COPY_BLOCK_BYTES = 16 * 1024 * 1024


def copy_and_hash(source: BinaryIO, destination: BinaryIO, *digests) -> int:
    written = 0
    for block in iter(lambda: source.read(COPY_BLOCK_BYTES), b""):
        destination.write(block)
        for digest in digests:
            digest.update(block)
        written += len(block)
    return written


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(COPY_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def require_download_manifest(root: Path) -> dict:
    path = root / "download-complete.json"
    if not path.is_file():
        raise FileNotFoundError(f"download completion marker is missing: {path}")
    manifest = json.loads(path.read_text())
    for relative, size, sha256 in (*INDEX_PARTS, (CORPUS_ARCHIVE, CORPUS_ARCHIVE_SIZE, CORPUS_ARCHIVE_SHA256)):
        record = manifest.get("verified", {}).get(relative)
        if record != {"bytes": size, "sha256": sha256}:
            raise RuntimeError(f"download manifest mismatch for {relative}: {record}")
        if (root / relative).stat().st_size != size:
            raise RuntimeError(f"source file size changed after download verification: {relative}")
    return manifest


def prepare_index(root: Path, prepared: Path) -> dict:
    output = prepared / "e5_Flat.index"
    partial = prepared / "e5_Flat.index.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite existing index output or partial: {output}")
    digest = hashlib.sha256()
    started = time.monotonic()
    written = 0
    with partial.open("xb") as destination:
        for relative, expected_size, expected_sha256 in INDEX_PARTS:
            part_digest = hashlib.sha256()
            with (root / relative).open("rb") as source:
                part_written = copy_and_hash(source, destination, digest, part_digest)
            if part_written != expected_size:
                raise RuntimeError(f"short index part while preparing {relative}: {part_written}")
            if part_digest.hexdigest() != expected_sha256:
                raise RuntimeError(f"source index part hash changed after download: {relative}")
            written += part_written
        destination.flush()
    expected_total = sum(item[1] for item in INDEX_PARTS)
    if written != expected_total or partial.stat().st_size != expected_total:
        raise RuntimeError(f"prepared index size mismatch: {written} != {expected_total}")
    partial.replace(output)
    return {
        "path": str(output),
        "bytes": written,
        "sha256": digest.hexdigest(),
        "source_order": [item[0] for item in INDEX_PARTS],
        "elapsed_seconds": time.monotonic() - started,
    }


def prepare_corpus(root: Path, prepared: Path) -> dict:
    output = prepared / "wiki-18.jsonl"
    partial = prepared / "wiki-18.jsonl.partial"
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite existing corpus output or partial: {output}")
    digest = hashlib.sha256()
    started = time.monotonic()
    lines = 0
    invalid_lines = 0
    observed_keys: set[str] = set()
    archive_path = root / CORPUS_ARCHIVE
    if sha256_file(archive_path) != CORPUS_ARCHIVE_SHA256:
        raise RuntimeError("source corpus archive hash changed after download")
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.next()
        if member is None or not member.isfile():
            raise RuntimeError("expected the first TAR member to be the regular corpus file")
        if member.name != CORPUS_MEMBER or member.size != CORPUS_MEMBER_SIZE:
            raise RuntimeError(
                f"unexpected corpus member: name={member.name!r}, size={member.size}"
            )
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError("could not open the corpus TAR member")
        written = 0
        with source, partial.open("xb") as destination:
            for raw_line in source:
                destination.write(raw_line)
                digest.update(raw_line)
                written += len(raw_line)
                lines += 1
                try:
                    record = json.loads(raw_line)
                    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                        raise ValueError("missing string id")
                    if not isinstance(record.get("contents"), str):
                        raise ValueError("missing string contents")
                    if lines <= 100:
                        observed_keys.update(record)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                    invalid_lines += 1
            destination.flush()
        if archive.next() is not None:
            raise RuntimeError("expected exactly one member in the corpus TAR archive")
    if written != CORPUS_MEMBER_SIZE or partial.stat().st_size != CORPUS_MEMBER_SIZE:
        raise RuntimeError(f"prepared corpus size mismatch: {written} != {CORPUS_MEMBER_SIZE}")
    if invalid_lines:
        raise RuntimeError(f"corpus contains {invalid_lines} invalid JSONL records")
    partial.replace(output)
    return {
        "path": str(output),
        "bytes": written,
        "sha256": digest.hexdigest(),
        "rows": lines,
        "invalid_rows": invalid_lines,
        "observed_keys_first_100": sorted(observed_keys),
        "archive_member": member.name,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.resource_root.resolve()
    prepared = root / "prepared"
    complete = prepared / "prepare-complete.json"
    if complete.exists():
        raise FileExistsError(f"refusing to overwrite completed preparation: {complete}")
    download_manifest = require_download_manifest(root)
    prepared.mkdir(parents=True, exist_ok=True)
    free_before = shutil.disk_usage(root).free
    if free_before < MIN_FREE_BYTES:
        raise RuntimeError(f"need at least {MIN_FREE_BYTES} free bytes, found {free_before}")

    started = time.monotonic()
    index = prepare_index(root, prepared)
    print(json.dumps({"index_prepared": index}, indent=2), flush=True)
    corpus = prepare_corpus(root, prepared)
    print(json.dumps({"corpus_prepared": corpus}, indent=2), flush=True)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "resource_root": str(root),
        "download_completed_at": download_manifest["completed_at"],
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(root).free,
        "elapsed_seconds": time.monotonic() - started,
        "sources_preserved": True,
        "index": index,
        "corpus": corpus,
        "next_gate": "validate FAISS dimensions/vector count and corpus ID alignment",
    }
    complete.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
