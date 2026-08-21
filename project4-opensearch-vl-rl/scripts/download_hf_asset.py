"""Download one immutable Hugging Face asset with bounded Range workers.

The downloader never uses inherited proxy settings, refuses to overwrite an
existing destination, preserves part files for resume/audit, and verifies both
the expected byte count and SHA256 before publishing the final file.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_DATA_ROOT = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
CHUNK_BYTES = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--attempts", type=int, default=8)
    return parser.parse_args()


def validate_public_https(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"only HTTPS URLs with a hostname are allowed: {url}")
    for answer in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(answer[4][0])
        if not address.is_global:
            raise ValueError(
                f"refusing non-public address for {parsed.hostname}: {address}"
            )


def validate_output(output: Path) -> Path:
    data_root = DEFAULT_DATA_ROOT.resolve()
    resolved = output.resolve()
    if not resolved.is_relative_to(data_root):
        raise ValueError(f"output must be below {data_root}: {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def split_ranges(size: int, workers: int) -> list[tuple[int, int]]:
    part_size = (size + workers - 1) // workers
    return [
        (start, min(start + part_size, size) - 1) for start in range(0, size, part_size)
    ]


def make_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers["User-Agent"] = "OpenSearch-VL-reproduction/1.0"
    return session


def download_range(
    url: str,
    part_path: Path,
    start: int,
    end: int,
    timeout: int,
    attempts: int,
) -> None:
    expected = end - start + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        existing = part_path.stat().st_size if part_path.exists() else 0
        if existing > expected:
            raise RuntimeError(f"part is larger than expected: {part_path}")
        if existing == expected:
            return

        request_start = start + existing
        try:
            with (
                make_session() as session,
                session.get(
                    url,
                    headers={"Range": f"bytes={request_start}-{end}"},
                    stream=True,
                    timeout=(timeout, timeout),
                ) as response,
            ):
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(
                        f"server ignored Range request: HTTP {response.status_code}"
                    )
                validate_public_https(response.url)
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {request_start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range}")

                written = existing
                with part_path.open("ab") as handle:
                    for chunk in response.iter_content(CHUNK_BYTES):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > expected:
                            raise RuntimeError(
                                f"server sent too many bytes for {part_path}"
                            )
                        handle.write(chunk)
            if written == expected:
                return
            last_error = RuntimeError(
                f"short Range response for {part_path}: {written}/{expected}"
            )
        except requests.RequestException as error:
            last_error = error

        if attempt < attempts:
            delay = min(2 ** (attempt - 1), 30)
            current = part_path.stat().st_size if part_path.exists() else 0
            print(
                f"retrying {part_path.name}: attempt={attempt + 1}/{attempts} "
                f"bytes={current}/{expected} delay={delay}s error={last_error}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Range failed after {attempts} attempts: {part_path}: {last_error}"
    )


def main() -> int:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("size must be positive")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    if args.timeout < 10:
        raise ValueError("timeout must be at least 10 seconds")
    if not 1 <= args.attempts <= 20:
        raise ValueError("attempts must be between 1 and 20")
    expected_sha256 = args.sha256.lower()
    if len(expected_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in expected_sha256
    ):
        raise ValueError("sha256 must contain exactly 64 hexadecimal characters")

    validate_public_https(args.url)
    output = validate_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if (
            output.stat().st_size == args.size
            and sha256_file(output) == expected_sha256
        ):
            print(f"already verified: {output}")
            return 0
        raise FileExistsError(
            f"refusing to overwrite existing unverified output: {output}"
        )

    parts_dir = Path(f"{output}.parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    ranges = split_ranges(args.size, args.workers)
    tasks = []
    for index, (start, end) in enumerate(ranges):
        part_path = parts_dir / f"part-{index:02d}-{start}-{end}"
        tasks.append((args.url, part_path, start, end, args.timeout, args.attempts))

    with ThreadPoolExecutor(
        max_workers=args.workers, thread_name_prefix="hf-range"
    ) as executor:
        list(executor.map(lambda task: download_range(*task), tasks))

    assembling = output.with_name(f".{output.name}.assembling.{os.getpid()}")
    if assembling.exists():
        raise FileExistsError(f"refusing to overwrite assembly file: {assembling}")
    digest = hashlib.sha256()
    total = 0
    with assembling.open("xb") as destination:
        for _, part_path, _, _, _, _ in tasks:
            with part_path.open("rb") as source:
                while chunk := source.read(CHUNK_BYTES):
                    destination.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        destination.flush()
        os.fsync(destination.fileno())

    actual_sha256 = digest.hexdigest()
    if total != args.size or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"verification failed; preserved {assembling}: "
            f"size={total}/{args.size} sha256={actual_sha256}/{expected_sha256}"
        )
    if output.exists():
        raise FileExistsError(
            f"destination appeared during assembly; preserved {assembling}"
        )
    assembling.rename(output)
    print(f"verified: {output} size={total} sha256={actual_sha256}")
    print(f"resume parts preserved: {parts_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"download failed: {error}", file=sys.stderr)
        raise
