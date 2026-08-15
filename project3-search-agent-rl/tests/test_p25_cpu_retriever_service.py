from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import faiss
import numpy as np
import uvicorn
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

    def test_global_concurrency_limit_queues_excess_requests(self):
        """max_concurrent_queries must serialize searches at the queue level.

        A real uvicorn server + urllib clients (the production path the GRPO
        training envs hit) with a 0.2s slow retriever: with limit 2 the wall
        clock must be >= 3 rounds * 0.2s (~0.6s); with limit 6 it must stay
        near one round (~0.2s). This is the anti-wedge guarantee for the
        330-env training burst.

        Note: this exercises the live serving path on purpose — starlette's
        TestClient portal is not thread-safe for concurrent posts, which
        deadlocks non-deterministically.
        """
        import json as _json
        import urllib.error

        class _SlowIndex:
            d = 2
            ntotal = 3

        class _SlowCorpus:
            rows = 3

        class SlowRetriever:
            index = _SlowIndex()
            corpus = _SlowCorpus()

            def search(self, query, topk):
                time.sleep(0.2)
                return [{"document": {"id": "0", "contents": "doc"}, "score": 1.0}]

        def hammer(port, results, index):
            started = time.monotonic()
            body = _json.dumps({"query": "x", "topk": 1}).encode()
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/retrieve",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    results[index] = (response.status, time.monotonic() - started)
            except urllib.error.HTTPError as error:
                results[index] = (error.code, time.monotonic() - started)

        for limit, lower, upper in ((2, 0.5, 1.5), (6, 0.0, 0.45)):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            server = self._start_server(SlowRetriever(), port, limit)
            try:
                results = [None] * 6
                threads = [threading.Thread(target=hammer, args=(port, results, i)) for i in range(6)]
                started = time.monotonic()
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                elapsed = time.monotonic() - started
                self.assertTrue(all(code == 200 for code, _ in results), f"responses: {results}")
                if lower:
                    self.assertGreaterEqual(elapsed, lower, f"limit {limit}: not queued (elapsed {elapsed:.3f}s)")
                self.assertLess(elapsed, upper, f"limit {limit}: unexpectedly slow (elapsed {elapsed:.3f}s)")
            finally:
                server.should_exit = True
                server_thread = getattr(server, "_thread", None)
                if server_thread:
                    server_thread.join(timeout=10)

    @staticmethod
    def _start_server(retriever, port, limit):
        class _QuietServer(uvicorn.Server):
            def install_signal_handlers(self):
                pass  # keep signal handling out of unit-test threads

        config = uvicorn.Config(
            create_app(retriever, max_concurrent_queries=limit),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = _QuietServer(config)
        thread = threading.Thread(target=server.run, daemon=True)
        server._thread = thread
        thread.start()
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("uvicorn server failed to start")
        return server

    def test_health_reports_concurrency_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            client = TestClient(create_app(build_tiny_retriever(Path(directory)), max_concurrent_queries=42))
            self.assertEqual(client.get("/health").json()["max_concurrent_queries"], 42)


if __name__ == "__main__":
    unittest.main()
