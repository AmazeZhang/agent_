"""CPU-only chunking logic test for the vLLM eval harness.

2026-08-15: the confirm-256 run wedged the CPU retriever (uvicorn sync pool
+ faiss OMP-24 per query) because the episode loop ran all 256 envs
concurrently (256 simultaneous searches; health starved, every search timed
out). The fix chunked the loop into sequential batches of
max_envs_per_batch envs — a pure concurrency control, since the eval envs
are deterministic and seedless (per-episode semantics identical).

This test exercises the chunked loop end to end on CPU with stub gates,
stub envs and stub generation, verifying: chunk construction (env
instantiations), global-index bookkeeping at chunk boundaries, step-2
continuation for unfinished envs, done/reward/won propagation, and the
results/episodes schema.

Requires the searchr1-repro-cu124 environment and
PYTHONPATH="$PWD/vendor/verl-agent:$PWD" (same as test_eval_heldout.py).
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

import run_p3_eval_vllm as ev  # noqa: E402

N_ROWS = 40
BATCH = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StubEnv:
    """SearchMultiProcessEnv stand-in: records instantiations, no HTTP."""

    created: list[int] = []

    def __init__(self, seed, env_num, group_n, is_train, env_config):
        self.n = env_num
        self.seed = seed
        self.is_train = is_train
        StubEnv.created.append(env_num)

    def close(self):
        pass


class StubManager:
    """SearchEnvironmentManager stand-in with deterministic step outcomes.

    Outcomes key on each env's GLOBAL question id (parsed from the reset
    question): envs with id % 3 == 0 finish on step 1 (reward 1.0); the rest
    finish on step 2 (reward 0.0). max_steps=2, so every env completes.
    Keying on the global id (rather than the in-chunk position) is what
    verifies that per-episode semantics are preserved across chunk
    boundaries — the whole point of the 2026-08-15 concurrency fix.
    """

    def __init__(self, raw_envs, projection, cfg):
        self.n = raw_envs.n
        self.steps_taken = 0
        self.qids: list[int] = []

    def reset(self, kwargs):
        n = len(kwargs)
        self.qids = [int(k["question"].rsplit(" ", 1)[-1]) for k in kwargs]
        return {"text": [k["question"] for k in kwargs], "anchor": [""] * n}, [{}] * n

    def step(self, actions):
        n = len(actions)
        self.steps_taken += 1
        done = [self.steps_taken >= 2 or q % 3 == 0 for q in self.qids]
        rewards = [1.0 if q % 3 == 0 else 0.0 for q in self.qids]
        infos = [{"tool_calling": False, "won": d and q % 3 == 0} for q, d in zip(self.qids, done)]
        observations = {
            "text": [f"step{self.steps_taken} q{q}" for q in self.qids],
            "anchor": ["obs"] * n,
        }
        return observations, rewards, done, infos


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path):
    """Fabricate eval data and replace every GPU/network gate with stubs."""
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
        lambda: {"run_id": "cpu-chunking-test", "run_dir": str(tmp_path), "cuda_visible_devices": "1"},
    )
    monkeypatch.setattr(ev, "validate_vllm_engine_parity", lambda: {"note": "stubbed"})
    monkeypatch.setattr(ev, "retriever_health_check", lambda *a, **k: {"status": "ready", "vectors": 21_015_324})
    monkeypatch.setattr(ev, "leakage_check", lambda *a, **k: {"eval_questions": N_ROWS, "reference_questions": 0, "overlap": 0})
    monkeypatch.setattr(ev, "verify_data_hash", lambda *a, **k: {"checked": True, "sha256": "stub"})
    monkeypatch.setattr(ev, "build_engine", lambda *a, **k: None)
    monkeypatch.setattr(
        ev, "generate_actions",
        lambda llm, tokenizer, prompts, args: ["<answer>42</answer>"] * len(prompts),
    )
    monkeypatch.setattr(ev, "offline_rescore", lambda steps, answers: {"em": 0.0, "has_answer": False, "score": 0.0})
    monkeypatch.setattr(ev, "search_projection", lambda actions: (list(actions), [True] * len(actions)))
    monkeypatch.setattr(ev, "AutoTokenizer", type("StubTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: type("Tok", (), {"padding_side": None, "pad_token_id": None, "eos_token": "</s>"})())}))
    monkeypatch.setattr(ev, "SearchMultiProcessEnv", StubEnv)
    monkeypatch.setattr(ev, "SearchEnvironmentManager", StubManager)
    StubEnv.created = []

    import torch

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "stub")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a, **k: None)

    argv = [
        "run_p3_eval_vllm.py",
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


def test_chunked_loop_end_to_end(stub_pipeline):
    assert ev.main() == 0

    results = json.loads((stub_pipeline / "results.json").read_text())
    episodes = [json.loads(line) for line in (stub_pipeline / "episodes.jsonl").read_text().splitlines()]

    # 40 rows / batch 8 -> 5 env instantiations of exactly 8 envs each.
    assert StubEnv.created == [BATCH] * (N_ROWS // BATCH)

    assert len(episodes) == N_ROWS
    assert results["metrics"]["overall"]["n"] == N_ROWS
    assert results["metrics"]["overall"]["em"] == 14  # ids 0,3,...,39 (14 of 40)
    assert results["metrics"]["action_stats"]["total_steps"] == 14 * 1 + 26 * 2
    assert results["parameters"]["max_envs_per_batch"] == BATCH
    assert results["runtime_script_sha256"] == sha256_file(PROJECT_ROOT / "scripts" / "run_p3_eval_vllm.py")
    assert results["decoding_backend"] == "vllm-native-greedy"

    for i, episode in enumerate(episodes):
        if i % 3 == 0:
            assert len(episode["steps"]) == 1, f"episode {i} expected 1 step"
            assert episode["reward"] == 1.0
            assert episode["done"] and episode["won"]
        else:
            assert len(episode["steps"]) == 2, f"episode {i} expected 2 steps"
            assert episode["reward"] == 0.0
            assert episode["done"] and not episode["won"]

    # Global-index bookkeeping at chunk boundaries: episode 32 is the first
    # row of chunk 4, episode 39 the last of chunk 4.
    assert episodes[32]["steps"][0]["prompt"] == "question 32"
    assert episodes[32]["steps"][0]["raw_action"] == "<answer>42</answer>"
    assert episodes[39]["steps"][0]["prompt"] == "question 39"
    assert episodes[39]["reward"] == 1.0  # 39 % 3 == 0 -> finished on step 1
    assert episodes[34]["steps"][1]["observation"] == "obs"  # 34 % 3 == 1 -> step 2 ran
