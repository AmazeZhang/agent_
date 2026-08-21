#!/usr/bin/env python3
"""Verify every RL image reference below a safely extracted image root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_reference(reference: str) -> PurePosixPath:
    if not reference or "\x00" in reference or "\\" in reference:
        raise ValueError(f"unsafe image reference: {reference!r}")
    path = PurePosixPath(reference)
    windows_path = PureWindowsPath(reference)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"absolute image reference: {reference!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"non-canonical image reference: {reference!r}")
    return path


def reject_symlink_components(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"image reference traverses a symlink: {current}")


def referenced_images(jsonl: Path) -> tuple[int, list[str]]:
    rows = 0
    references = []
    with jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            images = item.get("images") if isinstance(item, dict) else None
            if not isinstance(images, list) or not images:
                raise ValueError(f"line {line_number} has no non-empty images list")
            rows += 1
            references.extend(str(reference) for reference in images)
    return rows, references


def audit_images(jsonl: Path, image_root: Path) -> dict[str, object]:
    root = image_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    rows, references = referenced_images(jsonl)
    unique_references = sorted(set(references))
    formats: Counter[str] = Counter()
    minimum_width = minimum_height = None
    maximum_width = maximum_height = 0

    for reference in unique_references:
        relative = safe_reference(reference)
        candidate = root.joinpath(*relative.parts)
        reject_symlink_components(candidate, root)
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"image reference escaped root or is not a file: {reference!r}")
        try:
            with Image.open(resolved) as image:
                image.verify()
            with Image.open(resolved) as image:
                width, height = image.size
                image_format = image.format or "<unknown>"
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"image decode failed: {reference!r}: {error}") from error
        if width <= 0 or height <= 0:
            raise ValueError(f"image has invalid dimensions: {reference!r} {width}x{height}")
        formats[image_format] += 1
        minimum_width = width if minimum_width is None else min(minimum_width, width)
        minimum_height = height if minimum_height is None else min(minimum_height, height)
        maximum_width = max(maximum_width, width)
        maximum_height = max(maximum_height, height)

    return {
        "schema_version": 1,
        "source": {
            "jsonl_path": str(jsonl.resolve()),
            "jsonl_size_bytes": jsonl.stat().st_size,
            "jsonl_sha256": sha256_file(jsonl),
            "image_root": str(root),
        },
        "rows": rows,
        "image_references": len(references),
        "unique_image_references": len(unique_references),
        "duplicate_image_reference_rows": len(references) - len(unique_references),
        "formats": dict(sorted(formats.items())),
        "dimensions": {
            "minimum_width": minimum_width or 0,
            "minimum_height": minimum_height or 0,
            "maximum_width": maximum_width,
            "maximum_height": maximum_height,
        },
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {output_path}")
    report = audit_images(input_path, args.image_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
