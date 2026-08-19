"""P3 Phase 4B.1 item 3: question passthrough through the REAL env path.

SearchMultiProcessEnv.reset() -> SearchEnv.reset() runs in-process (thread
pool); reset never contacts the retriever, only step() does. These tests
exercise the whole real path on CPU with a local fake retriever:

- patch 0008 extras carries "question" (envs.py _sync_reset): SearchEnv.question
  must be byte-identical to the input question
- env.search_aware_step_reward=true at the env top level must propagate into
  the per-env SearchEnv config (patch 0008), or the v1 metadata never exists
- the answer-leak rule sees the question: an alias in BOTH question and query
  is NOT a new leak; an alias appearing only in the query IS a leak
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv  # noqa: E402

# Fake retriever: every query returns one document whose body contains the
# answer token "Paris" (so evidence hits are exercised end-to-end).
DOC_CONTENTS = "Paris is the capital of France and the most populous city."


class _RetrieverHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # payload {"query","topk","return_scores"} unused
        resp = {"result": [[{"document": {"id": "0", "contents": DOC_CONTENTS}}]]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture(scope="module")
def retriever_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetrieverHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/retrieve"
    finally:
        server.shutdown()
        server.server_close()


def make_env(retriever_url: str, *, batch_size: int = 2, search_aware: bool = True) -> SearchMultiProcessEnv:
    env_config = DictConfig(
        {
            "max_steps": 2,
            "search": {
                "search_url": retriever_url,
                "topk": 3,
                "timeout": 10,
                "log_requests": False,
            },
            "search_aware_step_reward": search_aware,
        }
    )
    return SearchMultiProcessEnv(seed=0, env_num=batch_size, group_n=1, is_train=False, env_config=env_config)


class TestQuestionPassthrough:
    def test_question_byte_identical_and_both_question_and_query_alias_not_leak(self, retriever_url):
        q = "Where is the Eiffel Tower in Paris, France?"
        env = make_env(retriever_url)
        try:
            obs, info = env.reset([{"question": q, "ground_truth": {"target": ["Paris"]},
                                    "data_source": "confirm256"}])
            # byte-identical: env keeps the exact input string
            assert env.envs[0].question == q
            assert env.envs[0].question is not None
            assert len(obs) == 1 and obs[0].endswith(q)
            assert info[0]["data_source"] == "confirm256"

            obs, reward, done, info = env.step(["<search>Paris France capital</search>"])
            assert info[0]["retrieval"]["status"] == "success"
            sv = info[0]["search_v1"]
            assert sv["answer_leak"] is False          # alias in question too
            assert sv["answer_leak_alias"] is None
            assert sv["evidence_effective"] is True    # real doc hit survives
            assert sv["evidence_credit"] is True
            assert sv["step_shaping_c"] == 15

            # terminal step carries the v1 terminal metadata
            obs, reward, done, info = env.step(["<answer>Paris</answer>"])
            assert done[0] and reward[0] == 1.0
            assert info[0]["search_v1"]["terminal"] is True
            assert info[0]["search_v1"]["answer_reward_c"] == 100
        finally:
            env.close()

    def test_alias_only_in_query_is_leak(self, retriever_url):
        q = "Which European capital is famous for the Eiffel Tower?"
        env = make_env(retriever_url)
        try:
            env.reset([{"question": q, "ground_truth": {"target": ["Paris"]},
                        "data_source": "confirm256"}])
            obs, reward, done, info = env.step(["<search>Paris metro</search>"])
            assert info[0]["retrieval"]["status"] == "success"
            sv = info[0]["search_v1"]
            assert sv["answer_leak"] is True           # alias ONLY in query
            assert sv["answer_leak_alias"] == "paris"
            assert sv["evidence_effective"] is False   # leak zeroes evidence
            assert sv["step_shaping_c"] == -20
        finally:
            env.close()

    def test_env_flag_propagates_to_per_env_config(self, retriever_url):
        on = make_env(retriever_url, search_aware=True)
        try:
            assert on.envs[0].search_aware_step_reward is True
        finally:
            on.close()
        off = make_env(retriever_url, search_aware=False)
        try:
            assert off.envs[0].search_aware_step_reward is False  # default path untouched
        finally:
            off.close()

    def test_padding_dummy_kwarg_gets_empty_question(self, retriever_url):
        q = "Which European capital is famous for the Eiffel Tower?"
        env = make_env(retriever_url, batch_size=2)
        try:
            obs, info = env.reset([{"question": q, "ground_truth": {"target": ["Paris"]},
                                    "data_source": "confirm256"}])
            assert len(obs) == 1                       # padded slot filtered out
            assert env.envs[1].question == ""          # dummy kwarg passes through
            # the padded slot must ALSO survive a step (empty ground_truth)
            obs, reward, done, info = env.step(["<search>Paris metro</search>"])
            assert info[0]["retrieval"]["status"] == "success"
            assert info[0]["search_v1"]["answer_leak"] is True
            assert env.envs[1].search_count == 1       # padded step executed, no crash
        finally:
            env.close()
