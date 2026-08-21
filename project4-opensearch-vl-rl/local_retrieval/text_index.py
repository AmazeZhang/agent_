"""Read-only SQLite FTS5 text index for local Wikipedia evidence lookup."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Mapping

INDEX_FORMAT = "opensearch-vl.sqlite-fts5.v1"
QUERY_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)
REQUIRED_FIELDS = ("entity_id", "title", "source", "text")
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


def _validate_document(document: Mapping[str, object]) -> dict[str, str]:
    missing = [
        field for field in REQUIRED_FIELDS if not str(document.get(field, "")).strip()
    ]
    if missing:
        raise ValueError(f"text document missing non-empty fields: {missing}")
    return {field: str(document[field]).strip() for field in REQUIRED_FIELDS}


def _safe_match_expression(query: str) -> str:
    if len(query) > 1_000:
        raise ValueError("text query exceeds 1,000 characters")
    raw_tokens = QUERY_TOKEN_RE.findall(query)
    tokens = [token for token in raw_tokens if token.casefold() not in QUERY_STOPWORDS]
    if not tokens:
        raise ValueError("text query contains no searchable tokens")
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens[:32])


def build_text_index(
    output: Path,
    documents: Iterable[Mapping[str, object]],
    *,
    corpus: str,
    corpus_revision: str,
) -> Path:
    """Build a non-overwriting FTS5 database and atomically publish it."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite text index: {output}")
    if not corpus.strip() or not corpus_revision.strip():
        raise ValueError("corpus and corpus_revision must be non-empty")
    records = [_validate_document(document) for document in documents]
    if not records:
        raise ValueError("text index requires at least one document")
    entity_ids = [record["entity_id"] for record in records]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("text index contains duplicate entity_id values")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite text index staging: {staging}")
    connection = sqlite3.connect(staging)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute(
            "CREATE VIRTUAL TABLE docs USING fts5("
            "entity_id UNINDEXED, title, body, source UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        connection.executemany(
            "INSERT INTO docs(entity_id, title, body, source) VALUES (?, ?, ?, ?)",
            [
                (record["entity_id"], record["title"], record["text"], record["source"])
                for record in records
            ],
        )
        metadata = {
            "format": INDEX_FORMAT,
            "corpus": corpus,
            "corpus_revision": corpus_revision,
            "count": str(len(records)),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.close()
        if output.exists():
            raise FileExistsError(
                f"destination appeared during text index build; preserved {staging}"
            )
        staging.rename(output)
    except Exception as error:
        connection.close()
        raise RuntimeError(f"text index build failed; staging preserved at {staging}") from error
    return output


class LocalTextIndex:
    """Immutable FTS5 search and entity lookup backend."""

    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self.metadata = dict(
            self.connection.execute("SELECT key, value FROM metadata").fetchall()
        )
        if self.metadata.get("format") != INDEX_FORMAT:
            self.close()
            raise ValueError(f"unsupported text index format: {self.metadata.get('format')}")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> LocalTextIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _result(self, row: sqlite3.Row, *, score: float | None) -> dict[str, object]:
        result: dict[str, object] = {
            "title": row["title"],
            "source": row["source"],
            "summary": row["summary"],
            "entity_id": row["entity_id"],
            "corpus": self.metadata["corpus"],
            "corpus_revision": self.metadata["corpus_revision"],
        }
        if score is not None:
            result["score"] = round(score, 8)
        return result

    def lookup(self, entity_id: str) -> dict[str, object] | None:
        if not entity_id.strip() or len(entity_id) > 256:
            raise ValueError("entity_id must contain 1 to 256 non-space characters")
        row = self.connection.execute(
            "SELECT entity_id, title, source, substr(body, 1, 500) AS summary "
            "FROM docs WHERE entity_id = ? LIMIT 1",
            (entity_id,),
        ).fetchone()
        return None if row is None else self._result(row, score=None)

    def search(self, query: str, *, top_k: int = 5) -> list[dict[str, object]]:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        expression = _safe_match_expression(query)
        rows = self.connection.execute(
            "SELECT entity_id, title, source, "
            "snippet(docs, 2, '', '', ' … ', 32) AS summary, "
            "bm25(docs, 0.0, 4.0, 1.0, 0.0) AS rank "
            "FROM docs WHERE docs MATCH ? ORDER BY rank, rowid LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [self._result(row, score=-float(row["rank"])) for row in rows]


def text_tool_observation(results: list[Mapping[str, object]]) -> str:
    payload = {
        "backend": "local_text_index",
        "match_count": len(results),
        "results": results,
    }
    return "Tool execution result:\n" + json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    )
