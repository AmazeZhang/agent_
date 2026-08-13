from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np
from fastapi.testclient import TestClient

from scripts.build_p25_corpus_offsets import build_offsets
from searchr1_repro.cpu_dense_retriever import CorpusStore, CpuDenseRetriever, create_app


class FakeEncoder:
    def encode(self, query: str) -> np.ndarray:
        return np.array([[1.0, 0.0]], dtype=np.float32)


def build_tiny_retriever(tmp_path: Path) -> CpuDenseRetriever:
    records = [
        {"id": "0", "contents": "first document"},
        {"id": "1", "contents": "second document"},
        {"id": "2", "contents": "third document"},
    ]
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    offsets_path = tmp_path / "offsets.npy"
    result = build_offsets(corpus_path, offsets_path, expected_rows=3)
    assert result["rows"] == 3
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32))
    return CpuDenseRetriever(index=index, corpus=CorpusStore(corpus_path, offsets_path), encoder=FakeEncoder())


class CpuRetrieverServiceTest(unittest.TestCase):
    def test_random_access_store_and_retrieval_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            retriever = build_tiny_retriever(Path(directory))
            self.assertEqual(retriever.corpus.get(2)["contents"], "third document")
            client = TestClient(create_app(retriever, default_topk=2, max_topk=3))
            self.assertEqual(client.get("/health").json()["vectors"], 3)
            response = client.post("/retrieve", json={"query": "test", "topk": 2, "return_scores": True})
            self.assertEqual(response.status_code, 200)
            hits = response.json()["result"][0]
            self.assertEqual([hit["document"]["id"] for hit in hits], ["0", "1"])
            self.assertTrue(all(set(hit) == {"document", "score"} for hit in hits))

    def test_service_rejects_blank_query_and_excessive_topk(self):
        with tempfile.TemporaryDirectory() as directory:
            client = TestClient(create_app(build_tiny_retriever(Path(directory)), max_topk=3))
            self.assertEqual(client.post("/retrieve", json={"query": "   "}).status_code, 422)
            self.assertEqual(client.post("/retrieve", json={"query": "test", "topk": 4}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
