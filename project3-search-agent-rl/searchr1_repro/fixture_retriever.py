"""Deterministic lexical retriever used only for P1 protocol smoke tests.

The fixture corpus is derived from selected ground-truth answers. It must never
be used for benchmark scores or model-quality claims.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class FixtureRetriever:
    def __init__(self, corpus_path: str | Path):
        self.corpus_path = Path(corpus_path)
        self.documents = [json.loads(line) for line in self.corpus_path.read_text().splitlines() if line.strip()]
        if not self.documents:
            raise ValueError("fixture corpus must contain at least one document")
        self._doc_terms = [Counter(tokenize(doc["contents"])) for doc in self.documents]
        self._document_frequency = Counter()
        for terms in self._doc_terms:
            self._document_frequency.update(terms.keys())

    def _score(self, query: str, doc_index: int) -> float:
        query_terms = Counter(tokenize(query))
        doc_terms = self._doc_terms[doc_index]
        score = 0.0
        for term, query_count in query_terms.items():
            if term not in doc_terms:
                continue
            inverse_document_frequency = math.log(
                1.0 + len(self.documents) / (1.0 + self._document_frequency[term])
            )
            score += min(query_count, doc_terms[term]) * inverse_document_frequency
        return score

    def search(self, query: str, topk: int = 3) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            return []
        if not isinstance(topk, int) or topk < 1:
            raise ValueError("topk must be a positive integer")
        ranked = sorted(
            ((self._score(query, index), index, doc) for index, doc in enumerate(self.documents)),
            key=lambda item: (-item[0], item[2]["id"]),
        )
        return [
            {"document": doc, "score": float(score)}
            for score, _, doc in ranked[: min(topk, len(ranked))]
        ]

    def api_response(self, query: str, topk: int = 3, return_scores: bool = True) -> dict[str, Any]:
        results = self.search(query=query, topk=topk)
        if not return_scores:
            results = [item["document"] for item in results]
        # Match the upstream retrieval_server.py response nesting exactly.
        return {"result": [results]}
