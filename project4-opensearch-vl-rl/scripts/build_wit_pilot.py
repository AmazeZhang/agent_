#!/usr/bin/env python3
"""Build paired visual/text pilot indexes from one verified WIT shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import build_exact_index, build_text_index  # noqa: E402

REQUIRED_COLUMNS = {
    "embedding",
    "image_url",
    "metadata_url",
    "caption_attribution_description",
    "wit_features",
}
PREFERRED_LANGUAGES = ("en", "zh", "zh-TW", "zh-CN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--expected-dimension", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_at(values: list[Any], index: int) -> str:
    if index >= len(values) or values[index] is None:
        return ""
    return str(values[index]).strip()


def _url_title(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = path.rsplit("/", 1)[-1].replace("_", " ")
    return name.removeprefix("File:").strip() or "Untitled Wikimedia image"


def choose_evidence(
    wit_features: dict[str, list[Any]] | None,
    *,
    caption: str,
    image_url: str,
    metadata_url: str,
) -> dict[str, str]:
    features = wit_features or {}
    languages = [str(value) if value is not None else "" for value in features.get("language", [])]
    preferred = [
        index
        for language in PREFERRED_LANGUAGES
        for index, value in enumerate(languages)
        if value == language
    ]
    remaining = [index for index in range(len(languages)) if index not in preferred]
    indexes = preferred + remaining
    page_titles = features.get("page_title", [])
    page_urls = features.get("page_url", [])
    for index in indexes:
        title = _text_at(page_titles, index)
        source = _text_at(page_urls, index)
        if not title and not source:
            continue
        candidates = (
            features.get("context_page_description", []),
            features.get("context_section_description", []),
            features.get("caption_title_and_reference_description", []),
            features.get("caption_reference_description", []),
            features.get("caption_alt_text_description", []),
        )
        summary = next(
            (
                value
                for values in candidates
                if (value := _text_at(values, index))
            ),
            caption.strip(),
        )
        return {
            "title": title or _url_title(source),
            "source": source or metadata_url or image_url,
            "summary": summary or title or _url_title(image_url),
            "language": languages[index],
        }

    source = metadata_url.strip() or image_url.strip()
    title = _url_title(source)
    return {
        "title": title,
        "source": source,
        "summary": caption.strip() or title,
        "language": "",
    }


def load_wit_records(
    path: Path, *, expected_dimension: int, batch_size: int, revision: str
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, str]]]:
    if expected_dimension <= 0:
        raise ValueError("expected_dimension must be positive")
    if not 1 <= batch_size <= 4096:
        raise ValueError("batch_size must be between 1 and 4096")
    parquet = pq.ParquetFile(path)
    fields = set(parquet.schema_arrow.names)
    missing = sorted(REQUIRED_COLUMNS - fields)
    if missing:
        raise ValueError(f"WIT shard missing required columns: {missing}")
    vectors = np.empty((parquet.metadata.num_rows, expected_dimension), dtype=np.float32)
    visual_metadata: list[dict[str, object]] = []
    text_documents: list[dict[str, str]] = []
    offset = 0
    columns = sorted(REQUIRED_COLUMNS)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        embedding = batch.column(batch.schema.get_field_index("embedding"))
        if embedding.type.list_size != expected_dimension:
            raise ValueError(
                f"embedding dimension mismatch: {embedding.type.list_size}/{expected_dimension}"
            )
        values = np.asarray(
            embedding.values.to_numpy(zero_copy_only=False), dtype=np.float32
        ).reshape(len(batch), expected_dimension)
        if not np.isfinite(values).all():
            raise ValueError(f"embedding batch at row {offset} contains non-finite values")
        vectors[offset : offset + len(batch)] = values

        for local_index, row in enumerate(batch.to_pylist()):
            row_index = offset + local_index
            entity_id = f"wit:{revision[:12]}:{row_index:08d}"
            evidence = choose_evidence(
                row["wit_features"],
                caption=str(row["caption_attribution_description"] or ""),
                image_url=str(row["image_url"] or ""),
                metadata_url=str(row["metadata_url"] or ""),
            )
            visual_metadata.append(
                {
                    "entity_id": entity_id,
                    "title": evidence["title"],
                    "source": evidence["source"],
                    "summary": evidence["summary"][:2_000],
                }
            )
            text_documents.append(
                {
                    "entity_id": entity_id,
                    "title": evidence["title"],
                    "source": evidence["source"],
                    "text": evidence["summary"][:10_000],
                }
            )
        offset += len(batch)
    if offset != parquet.metadata.num_rows:
        raise RuntimeError(f"Parquet row count changed while reading: {offset}")
    return vectors, visual_metadata, text_documents


def build_pilot(
    input_path: Path,
    output: Path,
    *,
    corpus_revision: str,
    source_sha256: str,
    expected_dimension: int,
    batch_size: int,
) -> Path:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WIT pilot: {output}")
    expected_sha256 = source_sha256.lower()
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("source_sha256 must be 64 lowercase hexadecimal characters")
    actual_sha256 = sha256_file(input_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"WIT shard SHA256 mismatch: {actual_sha256}/{expected_sha256}")

    vectors, visual_metadata, text_documents = load_wit_records(
        input_path,
        expected_dimension=expected_dimension,
        batch_size=batch_size,
        revision=corpus_revision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite WIT pilot staging: {staging}")
    staging.mkdir()
    try:
        corpus = "wikimedia/wit_base:train-00000-of-00330"
        build_exact_index(
            staging / "visual",
            vectors,
            visual_metadata,
            corpus=corpus,
            corpus_revision=corpus_revision,
        )
        build_text_index(
            staging / "text.sqlite",
            text_documents,
            corpus=corpus,
            corpus_revision=corpus_revision,
        )
        manifest = {
            "schema_version": 1,
            "corpus": corpus,
            "corpus_revision": corpus_revision,
            "source_path": str(input_path.resolve()),
            "source_sha256": actual_sha256,
            "count": len(visual_metadata),
            "embedding_dimension": expected_dimension,
            "embedding_provenance": (
                "WIT-published ImageNet ResNet-50 second-to-last layer; exact "
                "checkpoint is not identified by the dataset card"
            ),
            "preferred_languages": list(PREFERRED_LANGUAGES),
            "license": "CC-BY-SA-4.0",
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        if output.exists():
            raise FileExistsError(
                f"destination appeared during WIT pilot build; preserved {staging}"
            )
        staging.rename(output)
    except Exception as error:
        raise RuntimeError(f"WIT pilot build failed; staging preserved at {staging}") from error
    return output


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    built = build_pilot(
        input_path,
        output_path,
        corpus_revision=args.corpus_revision,
        source_sha256=args.source_sha256,
        expected_dimension=args.expected_dimension,
        batch_size=args.batch_size,
    )
    print(f"built WIT pilot: {built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
