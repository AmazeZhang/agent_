"""Auditable exact-cosine visual index for small offline retrieval pilots.

This backend deliberately targets pilot corpora. Full WIT indexing will require
an approximate index after the schema and retrieval contract are validated.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

INDEX_FORMAT = "opensearch-vl.numpy-exact-cosine.v1"
REQUIRED_METADATA = ("title", "source", "entity_id")


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("vectors must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("vectors contain non-finite values")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        raise ValueError("vectors contain a zero-norm row")
    return np.ascontiguousarray(array / norms[:, None], dtype=np.float32)


def _validate_metadata(item: Mapping[str, object]) -> dict[str, object]:
    missing = [key for key in REQUIRED_METADATA if not str(item.get(key, "")).strip()]
    if missing:
        raise ValueError(f"candidate metadata missing non-empty fields: {missing}")
    return {
        "title": str(item["title"]),
        "source": str(item["source"]),
        "summary": str(item.get("summary", "")),
        "entity_id": str(item["entity_id"]),
    }


def build_exact_index(
    output: Path,
    vectors: np.ndarray,
    metadata: Iterable[Mapping[str, object]],
    *,
    corpus: str,
    corpus_revision: str,
) -> Path:
    """Build a non-overwriting exact index and publish it atomically."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite visual index: {output}")
    if not corpus.strip() or not corpus_revision.strip():
        raise ValueError("corpus and corpus_revision must be non-empty")
    normalised = _normalise_rows(vectors)
    records = [_validate_metadata(item) for item in metadata]
    if len(records) != normalised.shape[0]:
        raise ValueError(
            f"metadata/vector count mismatch: {len(records)}/{normalised.shape[0]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite index staging: {staging}")
    staging.mkdir()
    try:
        np.save(staging / "vectors.npy", normalised, allow_pickle=False)
        with (staging / "metadata.jsonl").open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        manifest = {
            "format": INDEX_FORMAT,
            "corpus": corpus,
            "corpus_revision": corpus_revision,
            "count": normalised.shape[0],
            "dimension": normalised.shape[1],
            "similarity": "cosine",
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if output.exists():
            raise FileExistsError(
                f"destination appeared during index build; preserved {staging}"
            )
        staging.rename(output)
    except Exception as error:
        raise RuntimeError(f"index build failed; staging preserved at {staging}") from error
    return output


class ExactVisualIndex:
    """Read-only, memory-mapped exact cosine index for bounded pilots."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        with (self.root / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format") != INDEX_FORMAT:
            raise ValueError(f"unsupported visual index format: {self.manifest.get('format')}")
        self.vectors = np.load(self.root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        with (self.root / "metadata.jsonl").open(encoding="utf-8") as handle:
            self.metadata = [_validate_metadata(json.loads(line)) for line in handle]
        expected_shape = (self.manifest.get("count"), self.manifest.get("dimension"))
        if self.vectors.shape != expected_shape:
            raise ValueError(f"vector shape does not match manifest: {self.vectors.shape}/{expected_shape}")
        if len(self.metadata) != self.vectors.shape[0]:
            raise ValueError("metadata count does not match vector count")

    def search(
        self,
        query_vector: Sequence[float] | np.ndarray,
        *,
        top_k: int = 5,
        minimum_similarity: float = -1.0,
    ) -> list[dict[str, object]]:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1 and 1")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.shape != (self.vectors.shape[1],):
            raise ValueError(
                f"query dimension mismatch: {query.shape}/{(self.vectors.shape[1],)}"
            )
        norm = float(np.linalg.norm(query))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("query must contain finite values and have non-zero norm")
        scores = np.asarray(self.vectors @ (query / norm), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")[:top_k]
        results = []
        for index in order:
            similarity = float(scores[index])
            if similarity < minimum_similarity:
                continue
            results.append(
                {
                    **self.metadata[int(index)],
                    "similarity": round(similarity, 8),
                    "corpus": self.manifest["corpus"],
                    "corpus_revision": self.manifest["corpus_revision"],
                }
            )
        return results

    def search_batch(
        self,
        query_vectors: Sequence[Sequence[float]] | np.ndarray,
        *,
        top_k: int = 5,
        minimum_similarity: float = -1.0,
    ) -> list[list[dict[str, object]]]:
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between -1 and 1")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                "batch query dimension mismatch: "
                f"{queries.shape}/(*, {self.vectors.shape[1]})"
            )
        if not len(queries) or not np.isfinite(queries).all():
            raise ValueError("batch queries must be non-empty and finite")
        norms = np.linalg.norm(queries, axis=1)
        if np.any(norms == 0):
            raise ValueError("batch queries contain a zero-norm row")
        scores = (queries / norms[:, None]) @ self.vectors.T
        batches = []
        for query_scores in scores:
            order = np.argsort(-query_scores, kind="stable")[:top_k]
            results = []
            for index in order:
                similarity = float(query_scores[index])
                if similarity < minimum_similarity:
                    continue
                results.append(
                    {
                        **self.metadata[int(index)],
                        "similarity": round(similarity, 8),
                        "corpus": self.manifest["corpus"],
                        "corpus_revision": self.manifest["corpus_revision"],
                    }
                )
            batches.append(results)
        return batches


def tool_observation(results: Sequence[Mapping[str, object]]) -> str:
    """Return the stable payload expected after an ``image_search`` call."""

    payload = {
        "backend": "local_visual_index",
        "match_count": len(results),
        "results": list(results),
    }
    return "Tool execution result:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    )


def entity_tool_observation(results: Sequence[Mapping[str, object]]) -> str:
    """Return visual entity candidates without leaking page-text evidence."""

    projected = [
        {key: value for key, value in result.items() if key != "summary"}
        for result in results
    ]
    payload = {
        "backend": "local_visual_index",
        "evidence_scope": "entity-candidates-only",
        "match_count": len(projected),
        "results": projected,
    }
    return "Tool execution result:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    )
