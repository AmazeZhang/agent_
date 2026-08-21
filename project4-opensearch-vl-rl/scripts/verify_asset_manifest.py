"""Verify a local snapshot against a committed immutable asset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ignore-path", action="append", default=[])
    return parser.parse_args()


def digest_file(path: Path, algorithm: str, git_blob_size: int | None = None) -> str:
    digest = hashlib.new(algorithm)
    if git_blob_size is not None:
        digest.update(f"blob {git_blob_size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(root: Path, item: dict[str, Any]) -> str:
    path = root / item["path"]
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest file: {path}")
    expected_size = item["size"]
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"size mismatch for {path}: {actual_size}/{expected_size}")

    if "sha256" in item:
        actual_digest = digest_file(path, "sha256")
        expected_digest = item["sha256"]
        digest_name = "sha256"
    else:
        actual_digest = digest_file(path, "sha1", git_blob_size=expected_size)
        expected_digest = item["blob_sha1"]
        digest_name = "git_blob_sha1"
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"{digest_name} mismatch for {path}: {actual_digest}/{expected_digest}"
        )
    return item["path"]


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    manifest = json.loads(args.manifest.read_text())
    asset = next(
        (item for item in manifest["assets"] if item["repo_id"] == args.repo_id),
        None,
    )
    if asset is None:
        raise ValueError(f"repo_id is absent from manifest: {args.repo_id}")
    root = args.root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    manifest_paths = {item["path"] for item in asset["files"]}
    ignored_paths = set(args.ignore_path)
    unknown_ignored = ignored_paths - manifest_paths
    if unknown_ignored:
        raise ValueError(
            f"ignored paths are absent from manifest: {sorted(unknown_ignored)}"
        )
    files_to_verify = [
        item for item in asset["files"] if item["path"] not in ignored_paths
    ]

    with ThreadPoolExecutor(
        max_workers=args.workers, thread_name_prefix="asset-hash"
    ) as executor:
        verified = list(
            executor.map(lambda item: verify_file(root, item), files_to_verify)
        )

    expected_paths = manifest_paths
    extra_paths = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and str(path.relative_to(root)) not in expected_paths
    )
    print(
        json.dumps(
            {
                "repo_id": asset["repo_id"],
                "revision": asset["revision"],
                "verified_files": len(verified),
                "verified_bytes": sum(item["size"] for item in files_to_verify),
                "ignored_manifest_paths": sorted(ignored_paths),
                "extra_files": extra_paths,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
