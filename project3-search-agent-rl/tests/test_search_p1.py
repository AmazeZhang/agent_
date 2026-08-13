from __future__ import annotations

import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from searchr1_repro.fixture_retriever import FixtureRetriever
from agent_system.environments.env_package.search.projection import search_projection
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (
    compute_score,
    extract_solution,
    normalize_answer,
)


SMOKE_ROOT = Path("/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke")


@pytest.fixture(scope="module")
def retriever() -> FixtureRetriever:
    return FixtureRetriever(SMOKE_ROOT / "fixture_corpus.jsonl")


@contextmanager
def retrieval_server(retriever: FixtureRetriever):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/retrieve":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            response = retriever.api_response(
                query=payload["query"],
                topk=payload.get("topk", 3),
                return_scores=payload.get("return_scores", False),
            )
            encoded = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/retrieve"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_smoke_manifest_is_disjoint_and_sized():
    manifest = json.loads((SMOKE_ROOT / "manifest.json").read_text())
    assert manifest["outputs"]["train"]["rows"] == 8
    assert manifest["outputs"]["test"]["rows"] == 16
    train_questions = {item["question"].casefold().strip() for item in manifest["records"] if item["split"] == "train"}
    test_questions = {item["question"].casefold().strip() for item in manifest["records"] if item["split"] == "test"}
    assert train_questions.isdisjoint(test_questions)
    assert manifest["fixture_corpus_policy"]["ground_truth_derived"] is True


def test_reward_normalization_and_last_answer_semantics():
    truth = {"target": ["Wilhelm Conrad Röntgen", "Röntgen"]}
    assert normalize_answer("The Wilhelm Conrad Röntgen!") == "wilhelm conrad röntgen"
    assert compute_score("<answer>Wilhelm Conrad Röntgen</answer>", truth) == 1
    assert compute_score("<answer>wrong</answer><answer>Röntgen</answer>", truth) == 1
    assert extract_solution("<answer>first</answer><answer>last</answer>") == "last"
    assert compute_score("Röntgen", truth) == 0
    assert compute_score("<answer>Wilhelm</answer>", truth) == 0


@pytest.mark.parametrize(
    ("action", "projected", "valid"),
    [
        ("thinking <search> alpha </search> trailing", "<search>alpha</search>", 1),
        ("<answer>beta</answer>", "<answer>beta</answer>", 1),
        ("<search>a</search><answer>b</answer>", "<search>a</search>", 0),
        ("<search>a</search><search>b</search>", "<search>a</search>", 0),
        ("no complete action", "", 0),
    ],
)
def test_action_projection(action, projected, valid):
    assert search_projection([action]) == ([projected], [valid])


def test_fixture_retriever_contract(retriever: FixtureRetriever):
    record = json.loads((SMOKE_ROOT / "records.jsonl").read_text().splitlines()[0])
    response = retriever.api_response(record["question"], topk=3, return_scores=True)
    assert list(response) == ["result"]
    assert len(response["result"]) == 1
    assert len(response["result"][0]) == 3
    assert response["result"][0][0]["document"]["id"] == f"fixture-{record['id']}"
    assert response["result"][0][0]["document"]["ground_truth_derived"] is True


def test_every_fixture_question_retrieves_its_own_marked_document(retriever: FixtureRetriever):
    records = [json.loads(line) for line in (SMOKE_ROOT / "records.jsonl").read_text().splitlines()]
    for record in records:
        first = retriever.search(record["question"], topk=1)[0]["document"]
        assert first["id"] == f"fixture-{record['id']}"
        assert first["fixture_only"] is True


def test_search_environment_end_to_end(retriever: FixtureRetriever):
    record = json.loads((SMOKE_ROOT / "records.jsonl").read_text().splitlines()[0])
    with retrieval_server(retriever) as url:
        config = OmegaConf.create({"search_url": url, "topk": 3, "timeout": 2, "log_requests": False})
        env = SearchEnv(config)
        env.reset({"ground_truth": {"target": record["answers"]}, "max_turns": 4, "data_source": record["source"]})
        search_step = env.step(f"<search>{record['question']}</search>")
        assert search_step["done"] is False
        assert search_step["reward"] == 0
        assert "<information>" in search_step["observations"][0]["content"]
        assert record["answers"][0] in search_step["observations"][0]["content"]
        assert search_step["metadata"]["retrieval"]["status"] == "success"
        assert search_step["metadata"]["retrieval"]["document_ids"][0] == f"fixture-{record['id']}"
        assert search_step["metadata"]["retrieval_failed"] is False
        answer_step = env.step(f"<answer>{record['answers'][0]}</answer>")
        assert answer_step["done"] is True
        assert answer_step["reward"] == 1


def test_max_turn_termination_scores_last_complete_answer(retriever: FixtureRetriever):
    record = json.loads((SMOKE_ROOT / "records.jsonl").read_text().splitlines()[1])
    with retrieval_server(retriever) as url:
        config = OmegaConf.create({"search_url": url, "topk": 1, "timeout": 2, "log_requests": False})
        env = SearchEnv(config)
        env.reset({"ground_truth": {"target": record["answers"]}, "max_turns": 1, "data_source": record["source"]})
        out = env.step(f"<search>{record['question']}</search>")
        assert out["done"] is True
        assert out["reward"] == 0
        assert out["observations"] == []


def test_retriever_failure_is_typed_not_silently_model_failure(monkeypatch):
    record = json.loads((SMOKE_ROOT / "records.jsonl").read_text().splitlines()[2])

    def fail_immediately(**_):
        return None, "fixture timeout"

    monkeypatch.setattr(
        "agent_system.environments.env_package.search.third_party.skyrl_gym.tools.search.call_search_api",
        fail_immediately,
    )
    config = OmegaConf.create(
        {"search_url": "http://127.0.0.1:1/retrieve", "topk": 1, "timeout": 1, "log_requests": False}
    )
    env = SearchEnv(config)
    env.reset({"ground_truth": {"target": record["answers"]}, "max_turns": 4, "data_source": record["source"]})
    out = env.step(f"<search>{record['question']}</search>")
    assert out["done"] is False
    assert out["reward"] == 0
    assert out["metadata"]["retrieval_failed"] is True
    assert out["metadata"]["retrieval"]["status"] == "api_error"
    assert out["metadata"]["retrieval"]["document_ids"] == []
    assert "fixture timeout" in out["metadata"]["retrieval"]["api_request_error"]
