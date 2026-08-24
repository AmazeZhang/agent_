from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_training_curves.py"
SPEC = importlib.util.spec_from_file_location("generate_training_curves", SCRIPT)
assert SPEC and SPEC.loader
CURVES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURVES)


def metric_line(step: int, reward: float, grad: float) -> str:
    return (
        f"\x1b[36m(TaskRunner pid=42)\x1b[0m step:{step}"
        f" - actor/pg_loss:{-0.01 * step:.3f}"
        f" - actor/kl_loss:{0.002 * step:.3f}"
        f" - actor/entropy_loss:{0.8 - 0.1 * step:.3f}"
        f" - actor/grad_norm:{grad:.3f}"
        f" - training/global_step:{step:.3f}"
        f" - critic/score/mean:{reward:.3f}"
        f" - episode/reward/mean:{reward:.3f}"
        f" - episode/success_rate:{reward:.3f}"
        f" - episode/tool_call_count/mean:{1 + step / 10:.3f}"
        f" - episode/length/mean:{2 + step / 10:.3f}"
        f" - timing_s/gen:{20 + step:.3f}"
        f" - timing_s/update_actor:{60 + step:.3f}"
        f" - timing_s/step:{100 + step:.3f}"
        f" - perf/throughput:{70 - step:.3f}\n"
    )


def audit_record(traj: str, env_step: int, *, terminal: bool, searched: bool) -> dict:
    search = (
        {
            "version": "v2",
            "status": "success",
            "terminal": False,
            "evidence_hit": True,
            "evidence_credit": env_step == 0,
            "invalid_or_error": False,
            "redundant_search": env_step > 0,
        }
        if searched
        else {"version": "v2", "terminal": terminal}
    )
    return {
        "trajectory_advantage": 0.5 if traj == "a" else -0.5,
        "metadata": {
            "traj_uid": traj,
            "env_step": env_step,
            "is_padding": False,
            "search_v1": search,
            "search_v1_episode": {
                "answer_reward_c": 100 if traj == "a" else 0,
                "format_reward_c": 0,
                "evidence_hit_reward_c": 15,
                "searched_correct_bonus_c": 30 if traj == "a" else 0,
                "invalid_penalty_c": 0,
                "redundant_penalty_c": -45 if env_step > 0 else 0,
                "answer_leak_penalty_c": 0,
            },
        },
    }


def test_generate_console_and_audit_artifacts_atomically(tmp_path: Path) -> None:
    run = tmp_path / "run-a"
    rollout = run / "rollouts"
    rollout.mkdir(parents=True)
    (run / "metadata.env").write_text("run_id=run-a\nphysical_gpu_ids=1,2\n", encoding="utf-8")
    (run / "stdout.log").write_text(metric_line(1, 0.2, 1.1) + metric_line(2, 0.4, 0.9), encoding="utf-8")
    records = [
        audit_record("a", 0, terminal=False, searched=True),
        audit_record("a", 1, terminal=True, searched=False),
        audit_record("b", 0, terminal=False, searched=True),
        audit_record("b", 1, terminal=False, searched=True),
        audit_record("b", 3, terminal=True, searched=False),
    ]
    (rollout / "1.audit.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    output = CURVES.generate(run)
    assert output == run / "training_curves"
    assert not list(run.glob(".training_curves.partial-*"))
    expected = {
        "index.html",
        "metrics.csv",
        "search_behavior.csv",
        "search_behavior.svg",
        "summary.json",
        "training_overview.svg",
        "training_system.svg",
    }
    assert {path.name for path in output.iterdir()} == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["metric_steps"] == [1, 2]
    assert summary["audit_steps"] == [1]
    assert summary["generation"]["gpu_required"] is False
    with (output / "search_behavior.csv").open(newline="") as handle:
        behavior = next(csv.DictReader(handle))
    assert behavior["search_trajectory_rate"] == "1"
    assert behavior["true_redundant_rate"] == "0.333333333333"
    assert behavior["reached_step4_rate"] == "0.5"
    overview = (output / "training_overview.svg").read_text()
    assert "Outcome and reward" in overview
    assert "<polyline" in overview

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        CURVES.generate(run)


def test_duplicate_step_last_line_wins_and_nonfinite_is_reported(tmp_path: Path) -> None:
    log = tmp_path / "stdout.log"
    log.write_text(
        metric_line(1, 0.1, 1.0)
        + "step:1 - training/global_step:1.000 - critic/score/mean:0.900 - actor/grad_norm:nan\n",
        encoding="utf-8",
    )
    rows, diagnostic = CURVES.parse_console_metrics(log)
    assert rows == [{"step": 1.0, "training/global_step": 1.0, "critic/score/mean": 0.9}]
    assert diagnostic["duplicate_steps"] == [1]
    assert diagnostic["nonfinite_metrics"][0]["metric"] == "actor/grad_norm"


def test_no_training_metrics_is_a_clean_skip(tmp_path: Path) -> None:
    run = tmp_path / "eval-run"
    run.mkdir()
    (run / "stdout.log").write_text("evaluation complete\n", encoding="utf-8")
    assert CURVES.generate(run) is None
    assert not (run / "training_curves").exists()
