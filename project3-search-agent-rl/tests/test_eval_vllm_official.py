"""CPU-only tests for the OFFICIAL-LOOSE eval harness (run_p3_eval_vllm_official.py).

The official line differs from the strict line in exactly the places that
matter for its claim: the model's RAW action string must reach the vendored
skyrl SearchEnv directly (no SearchEnvironmentManager, no search_projection),
error observations (no query / failed retrieval) must be recorded as
retry-able steps without any penalty, and the episode loop must stay chunked
(max_envs_per_batch) so the CPU retriever is never wedged again.

This test verifies with stub envs and stub generation:
- chunk construction (env instantiations) and global-index bookkeeping
- raw-action passthrough: the stub env receives exactly the actions the
  generation produced (nothing projected, nothing rewritten)
- done/reward/won propagation and step-2 continuation
- error-observation accounting (tool_exception status)
- results schema: line=official-loose, semantics block, no adapter, EM counts
- --tokenizer defaulting to --model

Requires the searchr1-repro-cu124 environment and
PYTHONPATH="$PWD/vendor/verl-agent:$PWD".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_p3_eval_vllm_official as ev  # noqa: E402

N_ROWS = 40
BATCH = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StubEnv:
    """SearchMultiProcessEnv stand-in: no HTTP; keyed on the GLOBAL qid.

    qid % 3 == 0 finishes on step 1 (reward 1.0); the rest finish on step 2
    (reward 0.0). qid % 5 == 1 additionally simulates a failed retrieval
    (tool_exception -> error observation) on every step it takes.
    Keying on the global qid (not the in-chunk position) is what verifies
    that per-episode semantics are preserved across chunk boundaries.
    """

    created: list[int] = []

    def __init__(self, seed, env_num, group_n, is_train, env_config):
        self.n = env_num
        StubEnv.created.append(env_num)

    def reset(self, kwargs):
        self.qids = [int(k["question"].rsplit(" ", 1)[-1]) for k in kwargs]
        self.round = 0
        return [f"prompt {q}" for q in self.qids], [{}] * len(kwargs)

    def step(self, actions):
        self.round += 1
        qids = self.qids
        done = [self.round >= 2 or q % 3 == 0 for q in qids]
        rewards = [1.0 if q % 3 == 0 else 0.0 for q in qids]
        infos, obs = [], []
        for idx, q in enumerate(qids):
            if q % 5 == 1:
                status, err_obs = "tool_exception", "Error: the query is empty. Please try again."
            else:
                status, err_obs = "success", None
            infos.append(
                {
                    "tool_calling": True,
                    "retrieval": {"query": f"q{q}", "status": status, "total_results": 0, "document_ids": []},
                    "retrieval_failed": status != "success",
                    "won": bool(done[idx] and rewards[idx] >= 1.0),
                }
            )
            obs.append(err_obs if err_obs is not None else f"obs q{q}")
        return obs, rewards, done, infos

    def close(self):
        pass


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    rows = [
        {
            "data_source": "nq" if i % 2 == 0 else "hotpotqa",
            "env_kwargs": {
                "question": f"question {i}",
                "ground_truth": {"target": [f"answer {i}"]},
            },
        }
        for i in range(N_ROWS)
    ]
    data_file = tmp_path / "heldout.parquet"
    pd.DataFrame(rows).to_parquet(data_file)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "train.parquet").write_bytes(b"")

    monkeypatch.setattr(
        ev, "validate_managed_environment",
        lambda: {"run_id": "cpu-official-test", "run_dir": str(tmp_path), "cuda_visible_devices": "1"},
    )
    monkeypatch.setattr(ev, "validate_vllm_engine_parity", lambda: {"note": "stubbed"})
    monkeypatch.setattr(ev, "retriever_health_check", lambda *a, **k: {"status": "ready", "vectors": 21_015_324})
    monkeypatch.setattr(ev, "leakage_check", lambda *a, **k: {"eval_questions": N_ROWS, "reference_questions": 0, "overlap": 0})
    monkeypatch.setattr(ev, "verify_data_hash", lambda *a, **k: {"checked": True, "sha256": "stub"})
    monkeypatch.setattr(ev, "build_engine", lambda *a, **k: None)
    monkeypatch.setattr(
        ev, "generate_actions",
        lambda llm, tokenizer, prompts, args: [f"<search>query q{i}</search>" for i in range(len(prompts))],
    )
    monkeypatch.setattr(ev, "offline_rescore", lambda steps, answers: {"em": 0.0, "has_answer": False, "score": 0.0})
    monkeypatch.setattr(ev, "AutoTokenizer", type("StubTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: type("Tok", (), {"padding_side": None, "pad_token_id": None, "eos_token": "</s>"})())}))
    monkeypatch.setattr(ev, "SearchMultiProcessEnv", StubEnv)
    StubEnv.created = []

    import torch

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "stub")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: None)

    argv = [
        "run_p3_eval_vllm_official.py",
        "--model", "/dummy",
        "--data-files", str(data_file),
        "--manifest", str(tmp_path / "manifest.json"),
        "--manifest-key", "heldout",
        "--leakage-reference", str(tmp_path / "train.parquet"),
        "--search-url", "http://127.0.0.1:1/retrieve",
        "--max-steps", "2",
        "--seed", "0",
        "--max-envs-per-batch", str(BATCH),
        "--output", str(tmp_path / "results.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return tmp_path


def test_official_chunked_loop_end_to_end(stub_pipeline):
    assert ev.main() == 0

    results = json.loads((stub_pipeline / "results.json").read_text())
    episodes = [json.loads(line) for line in (stub_pipeline / "episodes.jsonl").read_text().splitlines()]

    # 40 rows / batch 8 -> 5 env instantiations of exactly 8 envs each.
    assert StubEnv.created == [BATCH] * (N_ROWS // BATCH)

    assert len(episodes) == N_ROWS
    assert results["metrics"]["overall"]["n"] == N_ROWS
    assert results["metrics"]["overall"]["em"] == 14  # ids 0,3,...,39 (14 of 40)
    assert results["parameters"]["max_envs_per_batch"] == BATCH
    assert results["runtime_script_sha256"] == sha256_file(PROJECT_ROOT / "scripts" / "run_p3_eval_vllm_official.py")
    assert results["decoding_backend"] == "vllm-native-greedy"

    # Official-line markers: line tag, no adapter, semantics block.
    assert results["line"] == "official-loose"
    assert results["engine"]["enable_lora"] is False
    assert "action_parse" in results["semantics"]

    for i, episode in enumerate(episodes):
        if i % 3 == 0:
            assert len(episode["steps"]) == 1, f"episode {i} expected 1 step"
            assert episode["reward"] == 1.0
            assert episode["done"] and episode["won"]
        else:
            assert len(episode["steps"]) == 2, f"episode {i} expected 2 steps"
            assert episode["reward"] == 0.0
            assert episode["done"] and not episode["won"]

    # Raw-action passthrough: every recorded step action is exactly the
    # generation output — the env must never see a projected/rewritten action.
    for episode in episodes:
        for step in episode["steps"]:
            assert step["raw_action"].startswith("<search>query q"), step["raw_action"]

    # Error observations: qid % 5 == 1 envs are {1,6,11,16,21,26,31,36}.
    # Of these, 6/21/36 are also %3==0 -> 1 step; the other five take 2 steps.
    # error steps = 3*1 + 5*2 = 13.
    error_steps = sum(1 for e in episodes for s in e["steps"] if s["error_observation"])
    assert error_steps == 13, error_steps
    assert results["metrics"]["action_stats"]["error_observation_steps"] == 13
    assert results["metrics"]["retrieval"]["statuses"]["tool_exception"] == 13

    # Global-index bookkeeping at chunk boundaries (episodes 32 and 39 are
    # the first/last rows of chunk 4).
    assert episodes[32]["steps"][0]["prompt"] == "prompt 32"
    assert episodes[32]["steps"][0]["raw_action"] == "<search>query q0</search>"  # chunk-local generation index
    assert episodes[39]["reward"] == 1.0  # 39 % 3 == 0 -> finished on step 1
    assert episodes[34]["steps"][1]["observation"] == "obs q34"  # 34 % 3 == 1 -> step 2 ran
