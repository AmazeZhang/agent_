"""Audit and safely extract a ZIP archive below the project data root."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

DEFAULT_DATA_ROOT = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
COPY_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ArchiveAudit:
    files: int
    directories: int
    compressed_bytes: int
    uncompressed_bytes: int
    maximum_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-files", type=int, default=20_000)
    parser.add_argument("--max-uncompressed-bytes", type=int, default=100 << 30)
    parser.add_argument("--max-ratio", type=float, default=200.0)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extract after a successful audit; --output is required",
    )
    return parser.parse_args()


def below_data_root(path: Path, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    resolved = path.resolve()
    root = data_root.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path must be below {root}: {resolved}")
    return resolved


def safe_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe ZIP member name: {name!r}")
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"absolute ZIP member path: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"non-canonical ZIP member path: {name!r}")
    return path


def is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def audit_archive(
    archive: Path,
    *,
    max_files: int,
    max_uncompressed_bytes: int,
    max_ratio: float,
) -> ArchiveAudit:
    if max_files <= 0 or max_uncompressed_bytes <= 0 or max_ratio <= 0:
        raise ValueError("audit limits must be positive")

    seen: set[PurePosixPath] = set()
    files = directories = compressed = uncompressed = 0
    maximum_ratio = 0.0
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC check failed: {bad_member!r}")
        for info in handle.infolist():
            member = safe_member_path(info.filename.rstrip("/"))
            if member in seen:
                raise ValueError(f"duplicate ZIP member: {info.filename!r}")
            seen.add(member)
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member: {info.filename!r}")
            if is_symlink(info):
                raise ValueError(f"symbolic link ZIP member: {info.filename!r}")
            if info.is_dir():
                directories += 1
                continue

            files += 1
            compressed += info.compress_size
            uncompressed += info.file_size
            ratio = info.file_size / max(info.compress_size, 1)
            maximum_ratio = max(maximum_ratio, ratio)
            if files > max_files:
                raise ValueError(f"ZIP file count exceeds limit: {files}/{max_files}")
            if uncompressed > max_uncompressed_bytes:
                raise ValueError(
                    "ZIP uncompressed size exceeds limit: "
                    f"{uncompressed}/{max_uncompressed_bytes}"
                )
            if ratio > max_ratio:
                raise ValueError(
                    f"ZIP member compression ratio exceeds limit: "
                    f"{info.filename!r} {ratio:.2f}/{max_ratio:.2f}"
                )

    return ArchiveAudit(
        files=files,
        directories=directories,
        compressed_bytes=compressed,
        uncompressed_bytes=uncompressed,
        maximum_ratio=maximum_ratio,
    )


def extract_archive(archive: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite extraction output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.extracting.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite extraction staging: {staging}")
    staging.mkdir()

    try:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                member = safe_member_path(info.filename.rstrip("/"))
                destination = staging.joinpath(*member.parts)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(info) as source, destination.open("xb") as target:
                    while chunk := source.read(COPY_CHUNK_BYTES):
                        target.write(chunk)
        if output.exists():
            raise FileExistsError(
                f"destination appeared during extraction; preserved {staging}"
            )
        staging.rename(output)
    except Exception as error:
        raise RuntimeError(f"extraction failed; partial output preserved at {staging}") from error
    return output


def write_report(path: Path, audit: ArchiveAudit, archive: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"archive": str(archive), **asdict(audit)}
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    archive = below_data_root(args.archive)
    if not archive.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    if args.extract and args.output is None:
        raise ValueError("--output is required with --extract")

    audit = audit_archive(
        archive,
        max_files=args.max_files,
        max_uncompressed_bytes=args.max_uncompressed_bytes,
        max_ratio=args.max_ratio,
    )
    print(json.dumps(asdict(audit), sort_keys=True))
    if args.report is not None:
        write_report(below_data_root(args.report), audit, archive)
    if args.extract:
        output = below_data_root(args.output)
        print(f"extracted: {extract_archive(archive, output)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ZIP operation failed: {error}", file=sys.stderr)
        raise
