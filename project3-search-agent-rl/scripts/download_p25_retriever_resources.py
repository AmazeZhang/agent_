#!/usr/bin/env python3
"""Safely download revision-pinned Search-R1 retriever resources.

This script downloads only. It never concatenates, decompresses, deletes, or
overwrites a completed resource tree. Preparation and validation are separate
gates so interrupted downloads cannot be mistaken for usable inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


INDEX_REPO = "PeterJinGo/wiki-18-e5-index"
INDEX_REVISION = "a4d31160a035f30764604f4827cd8f1d0315eb86"
CORPUS_REPO = "PeterJinGo/wiki-18-corpus"
CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"
MODEL_REPO = "intfloat/e5-base-v2"
MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
MIN_FREE_BYTES = 250 * 1024**3

EXPECTED = {
    "index-download/part_aa": (42949672960, "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023"),
    "index-download/part_ab": (21609402413, "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11"),
    "corpus-download/wiki-18.jsonl.gz": (5123307260, "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"),
    "model/e5-base-v2/model.safetensors": (437955512, "d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693"),
}
MODEL_FILES = [
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_empty_or_resumable(root: Path) -> None:
    completed = root / "download-complete.json"
    if completed.exists():
        raise FileExistsError(f"refusing to overwrite completed resource tree: {root}")
    root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    require_empty_or_resumable(root)
    free_bytes = shutil.disk_usage(root).free
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"need at least {MIN_FREE_BYTES} free bytes, found {free_bytes}")

    index_dir = root / "index-download"
    corpus_dir = root / "corpus-download"
    model_dir = root / "model" / "e5-base-v2"
    for filename in ("part_aa", "part_ab"):
        hf_hub_download(
            repo_id=INDEX_REPO,
            repo_type="dataset",
            revision=INDEX_REVISION,
            filename=filename,
            local_dir=index_dir,
        )
    hf_hub_download(
        repo_id=CORPUS_REPO,
        repo_type="dataset",
        revision=CORPUS_REVISION,
        filename="wiki-18.jsonl.gz",
        local_dir=corpus_dir,
    )
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        allow_patterns=MODEL_FILES,
    )

    verified = {}
    for relative, (expected_size, expected_hash) in EXPECTED.items():
        path = root / relative
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if (actual_size, actual_hash) != (expected_size, expected_hash):
            raise RuntimeError(
                f"integrity mismatch for {path}: "
                f"size={actual_size}, sha256={actual_hash}"
            )
        verified[relative] = {"bytes": actual_size, "sha256": actual_hash}

    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(root),
        "free_bytes_before_download": free_bytes,
        "revisions": {
            INDEX_REPO: INDEX_REVISION,
            CORPUS_REPO: CORPUS_REVISION,
            MODEL_REPO: MODEL_REVISION,
        },
        "verified": verified,
        "next_gate": "assemble index and decompress corpus while preserving source files",
    }
    (root / "download-complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
