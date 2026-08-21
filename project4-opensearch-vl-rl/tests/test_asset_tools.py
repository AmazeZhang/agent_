"""CPU-only tests for verified asset download and manifest validation helpers."""

import hashlib
import importlib.util
import tempfile
from itertools import pairwise
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(content)}\0".encode() + content, usedforsecurity=False
    ).hexdigest()


def main() -> None:
    downloader = load_module("download_hf_asset", "scripts/download_hf_asset.py")
    verifier = load_module("verify_asset_manifest", "scripts/verify_asset_manifest.py")

    ranges = downloader.split_ranges(101, 8)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 100
    assert sum(end - start + 1 for start, end in ranges) == 101
    assert all(left[1] + 1 == right[0] for left, right in pairwise(ranges))
    assert downloader.CHUNK_BYTES == 1024 * 1024

    with tempfile.TemporaryDirectory(prefix="p4-asset-tools.") as temporary:
        root = Path(temporary)
        lfs_content = b"verified-lfs-content"
        git_content = b"verified-git-blob"
        (root / "large.bin").write_bytes(lfs_content)
        (root / "small.txt").write_bytes(git_content)

        assert (
            verifier.verify_file(
                root,
                {
                    "path": "large.bin",
                    "size": len(lfs_content),
                    "sha256": hashlib.sha256(lfs_content).hexdigest(),
                },
            )
            == "large.bin"
        )
        assert (
            verifier.verify_file(
                root,
                {
                    "path": "small.txt",
                    "size": len(git_content),
                    "blob_sha1": git_blob_sha1(git_content),
                },
            )
            == "small.txt"
        )

        try:
            downloader.validate_output(root / "outside-data-root")
        except ValueError:
            pass
        else:
            raise AssertionError("download output escaped the project data root")

    print("asset tool tests: PASS")


if __name__ == "__main__":
    main()
