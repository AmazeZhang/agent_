"""CPU-only tests for the P3 held-out evaluation pipeline.

Requires the searchr1-repro-cu124 environment and
PYTHONPATH="$PWD/vendor/verl-agent:$PWD" (same as test_search_p1.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_p3_eval_heldout import (
    action_quality,
    aggregate_metrics,
    atomic_write_jsonl,
    leakage_check,
    offline_rescore,
    retriever_health_check,
)

BUILDER = PROJECT_ROOT / "scripts" / "build_p3_heldout_eval.py"

HELDOUT_QUOTAS = {
    "nq": 8,
    "hotpotqa": 8,
    "popqa": 4,
    "2wikimultihopqa": 4,
    "triviaqa": 4,
    "musique": 2,
    "bamboogle": 2,
}
SOURCES = list(HELDOUT_QUOTAS)


def make_frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """rows: (source, question, answer) -> DataFrame with upstream-like schema."""
    return pd.DataFrame(
        [
            {
                "data_source": source,
                "env_kwargs": {
                    "data_source": source,
                    "question": question,
                    "ground_truth": {"target": [answer]},
                },
                "extra_info": {"index": index},
            }
            for index, (source, question, answer) in enumerate(rows)
        ]
    )


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def run_builder(inputs: dict[str, Path], output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--test-source", str(inputs["test"]),
            "--train-source", str(inputs["train"]),
            "--smoke-train", str(inputs["smoke_train"]),
            "--smoke-test", str(inputs["smoke_test"]),
            "--output-dir", str(output_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def builder_inputs(tmp_path: Path) -> dict[str, Path]:
    train_rows = [(source, f"train {source} {i}", f"T{source}{i}") for source in ("nq", "hotpotqa") for i in range(10)]
    test_rows = [(source, f"test {source} {i}", f"A{source}{i}") for source in SOURCES for i in range(10)]
    # smoke train reuses one upstream-test question: must be excluded by the builder.
    smoke_train_rows = [("nq", "test nq 0", "sneaky")]
    smoke_test_rows = [("nq", "smoke test nq 0", "covered")]
    inputs = {
        "train": tmp_path / "upstream" / "train.parquet",
        "test": tmp_path / "upstream" / "test.parquet",
        "smoke_train": tmp_path / "smoke" / "train.parquet",
        "smoke_test": tmp_path / "smoke" / "test.parquet",
    }
    write_parquet(inputs["train"], make_frame(train_rows))
    write_parquet(inputs["test"], make_frame(test_rows))
    write_parquet(inputs["smoke_train"], make_frame(smoke_train_rows))
    write_parquet(inputs["smoke_test"], make_frame(smoke_test_rows))
    return inputs


def test_builder_deterministic_and_leakage_free(builder_inputs: dict[str, Path], tmp_path: Path):
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    run1 = run_builder(builder_inputs, out1)
    assert run1.returncode == 0, run1.stderr
    run2 = run_builder(builder_inputs, out2)
    assert run2.returncode == 0, run2.stderr

    heldout = pd.read_parquet(out1 / "heldout.parquet")
    assert len(heldout) == 32
    assert heldout.groupby("data_source").size().to_dict() == HELDOUT_QUOTAS

    # Byte-identical outputs across rebuilds.
    assert (out1 / "heldout.parquet").read_bytes() == (out2 / "heldout.parquet").read_bytes()
    assert (out1 / "records.jsonl").read_bytes() == (out2 / "records.jsonl").read_bytes()

    # Manifest identical once absolute paths are stripped (output dirs differ).
    manifest1 = json.loads((out1 / "manifest.json").read_text())
    manifest2 = json.loads((out2 / "manifest.json").read_text())
    for record in (manifest1["records"], manifest2["records"]):
        pass
    assert manifest1["records"] == manifest2["records"]
    assert manifest1["leakage"]["smoke_train_normalized_overlap"] == 0
    assert manifest1["leakage"]["smoke_test_normalized_overlap"] == 0
    assert manifest1["outputs"]["heldout"]["rows"] == 32

    # The sneaky smoke-train question ("test nq 0") must not be selected.
    eval_questions = {str(row["env_kwargs"]["question"]) for _, row in heldout.iterrows()}
    assert "test nq 0" not in eval_questions
    # Coverage exclusion: smoke test question not selected either.
    assert "smoke test nq 0" not in eval_questions


def test_real_heldout_disjoint_from_smoke_train():
    heldout_dir = Path("/media/imc/data/project3-search-agent-rl/datasets/searchr1-heldout32")
    if not (heldout_dir / "heldout.parquet").is_file():
        pytest.skip("heldout-32 not built yet")
    heldout = pd.read_parquet(heldout_dir / "heldout.parquet")
    smoke_train = pd.read_parquet(
        "/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke/train.parquet"
    )
    norm = lambda s: " ".join(s.casefold().split())
    eval_questions = {norm(str(row["env_kwargs"]["question"])) for _, row in heldout.iterrows()}
    train_questions = {norm(str(row["env_kwargs"]["question"])) for _, row in smoke_train.iterrows()}
    assert eval_questions.isdisjoint(train_questions)


def test_leakage_check_aborts_on_overlap(tmp_path: Path):
    records = [{"question": "when did x happen?", "answers": ["2001"], "data_source": "nq"}]
    reference = tmp_path / "train.parquet"
    write_parquet(reference, make_frame([("nq", "When did X happen?", "2001")]))
    with pytest.raises(RuntimeError, match="leakage"):
        leakage_check(records, reference)
    clean = tmp_path / "clean.parquet"
    write_parquet(clean, make_frame([("nq", "unrelated question", "x")]))
    assert leakage_check(records, clean)["overlap"] == 0


@contextmanager
def fake_health_server(status: str, vectors: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            payload = {"status": status, "vectors": vectors}
            encoded = json.dumps(payload).encode()
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


def test_retriever_health_gate():
    with fake_health_server("ready", 21_015_324) as url:
        assert retriever_health_check(url)["vectors"] == 21_015_324
    with fake_health_server("ready", 3) as url:
        with pytest.raises(RuntimeError, match="health gate failed"):
            retriever_health_check(url)
    with fake_health_server("booting", 21_015_324) as url:
        with pytest.raises(RuntimeError, match="health gate failed"):
            retriever_health_check(url)


def test_action_quality():
    valid = action_quality('<search>q</search>', True)
    assert valid == {
        "has_search_tag": True,
        "has_answer_tag": False,
        "mixed_tags": False,
        "duplicate_tags": False,
        "projected_valid": True,
    }
    mixed = action_quality('<search>q</search> then <answer>a</answer>', False)
    assert mixed["mixed_tags"] is True
    assert mixed["projected_valid"] is False
    duplicated = action_quality('<answer>a</answer><answer>b</answer>', False)
    assert duplicated["duplicate_tags"] is True


def test_offline_rescore_and_metrics():
    episodes = [
        {
            "question": "q1", "answers": ["CBS"], "source": "nq", "reward": 1.0, "won": True,
            "steps": [
                {
                    "raw_action": "<search>what channel</search>",
                    "action_quality": action_quality("<search>what channel</search>", True),
                    "executed_search": True,
                    "info": {"retrieval": {"status": "success"}},
                },
                {
                    "raw_action": "<answer>CBS</answer>",
                    "action_quality": action_quality("<answer>CBS</answer>", True),
                    "executed_search": False,
                    "info": {},
                },
            ],
            "offline": offline_rescore(
                [{"raw_action": "<search>what channel</search>"}, {"raw_action": "<answer>CBS</answer>"}],
                ["CBS"],
            ),
        },
        {
            "question": "q2", "answers": ["Paris"], "source": "hotpotqa", "reward": 0.0, "won": False,
            "steps": [
                {
                    "raw_action": "<search>bad query</search><answer>Rome</answer>",
                    "action_quality": action_quality("<search>bad query</search><answer>Rome</answer>", False),
                    "executed_search": True,
                    "info": {"retrieval": {"status": "invalid_query"}},
                },
            ],
            "offline": offline_rescore(
                [{"raw_action": "<search>bad query</search><answer>Rome</answer>"}], ["Paris"]
            ),
        },
        {
            "question": "q3", "answers": ["X"], "source": "triviaqa", "reward": 0.0, "won": False,
            "steps": [
                {
                    "raw_action": "<answer>Y</answer>",
                    "action_quality": action_quality("<answer>Y</answer>", True),
                    "executed_search": False,
                    "info": {},
                },
            ],
            "offline": offline_rescore([{"raw_action": "<answer>Y</answer>"}], ["X"]),
        },
    ]
    metrics = aggregate_metrics(episodes)
    assert metrics["overall"]["n"] == 3
    assert metrics["overall"]["em"] == 1
    assert metrics["overall"]["success"] == 1
    assert metrics["per_source"]["nq"]["em_rate"] == 1.0
    assert metrics["per_source"]["hotpotqa"]["em_rate"] == 0.0
    assert metrics["action_stats"]["invalid_actions"] == 1
    assert metrics["action_stats"]["mixed_tag_steps"] == 1
    assert metrics["retrieval"]["executed_searches"] == 2
    assert metrics["retrieval"]["statuses"] == {"success": 1, "invalid_query": 1}
    assert metrics["retrieval"]["invalid_query_rate"] == 0.5
    # Env reward and offline rescore agree on all three episodes.
    assert metrics["offline_rescore"]["matches"] == 3
    assert metrics["offline_rescore"]["mismatches"] == 0


def test_offline_rescore_extracts_last_answer_only():
    result = offline_rescore(
        [
            {"raw_action": "<answer>wrong</answer>"},
            {"raw_action": "doc text that must not matter"},
            {"raw_action": "<answer>right</answer>"},
        ],
        ["right"],
    )
    assert result["score"] == 1.0
    assert result["final_answer"] == "right"


def test_atomic_write_jsonl(tmp_path: Path):
    out = tmp_path / "episodes.jsonl"
    atomic_write_jsonl(out, [{"a": 1}, {"a": np.int64(2)}])
    assert not (tmp_path / "episodes.jsonl.partial").exists()
    assert [json.loads(line)["a"] for line in out.read_text().splitlines()] == [1, 2]


def test_managed_environment_gate(monkeypatch):
    import torch

    from run_p3_eval_heldout import validate_managed_environment

    monkeypatch.delenv("PROJECT3_RUN_ID", raising=False)
    monkeypatch.delenv("PROJECT3_RUN_DIR", raising=False)
    with pytest.raises(RuntimeError, match="run_managed.sh"):
        validate_managed_environment()

    monkeypatch.setenv("PROJECT3_RUN_ID", "eval-run")
    monkeypatch.setenv("PROJECT3_RUN_DIR", "/tmp/eval")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    with pytest.raises(RuntimeError, match="GPU 0"):
        validate_managed_environment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert validate_managed_environment()["run_id"] == "eval-run"


def test_eval_wrapper_requires_managed_run():
    env = {
        k: v for k, v in os.environ.items()
        if k not in {"PROJECT3_RUN_ID", "PROJECT3_RUN_DIR", "CUDA_VISIBLE_DEVICES", "PROJECT3_DATA_ROOT"}
    }
    env.setdefault("PROJECT3_DATA_ROOT", "/media/imc/data")
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "run_p3_eval_heldout.sh")],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 13, result.stdout + result.stderr
    assert "run_managed.sh" in result.stderr
