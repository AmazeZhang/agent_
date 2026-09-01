from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_rl_smoke_curves.py"
SPEC = importlib.util.spec_from_file_location("generate_rl_smoke_curves", SCRIPT)
assert SPEC and SPEC.loader
CURVES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURVES)


def _write_replay(run: Path) -> None:
    output = run / "output"
    output.mkdir(parents=True)
    state = {
        "global_step": 1,
        "weighted_loss": 0.00005,
        "grad_norm": 0.004,
        "learning_rate": 0.000001,
        "active_trajectory_count": 4,
    }
    (output / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    (run / "stdout.log").write_text(json.dumps(state) + "\n", encoding="utf-8")


def _event(step: int, rollout: int, reward: float, fatal: str | None = None) -> dict:
    return {
        "global_step": step,
        "rollout": rollout,
        "reward": reward,
        "full_success": False,
        "fatal": fatal,
        "tools": ["image_search", "text_search"],
    }


def _write_online(run: Path) -> None:
    checkpoint = run / "output" / "checkpoint-2"
    checkpoint.mkdir(parents=True)
    state = {
        "global_step": 2,
        "weighted_loss": 0.0003,
        "grad_norm": 0.02,
        "learning_rate": 0.000001,
        "gate": {"passed": True},
        "group_summary": {"reward_mean": 0.1, "reward_population_variance": 0.01},
    }
    (checkpoint / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    events = [_event(2, 1, 0.0), _event(2, 2, 0.2), _event(3, 1, 0.0, "maximum-turns-exceeded"), _event(3, 2, 0.1)]
    (run / "stdout.log").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_generate_preserves_failed_rollout_step_without_fake_loss(tmp_path: Path) -> None:
    replay, online, output = tmp_path / "replay", tmp_path / "online", tmp_path / "artifacts"
    _write_replay(replay)
    _write_online(online)

    rows = CURVES.generate(replay, online, output)

    assert [row["step"] for row in rows] == [1, 2, 3]
    assert rows[0]["stage"] == "replay"
    assert rows[1]["optimizer_update"] == 1
    assert rows[2]["optimizer_update"] == 0
    assert rows[2]["weighted_loss"] == ""
    assert rows[2]["fatal_fraction"] == 0.5
    with (output / "p4_rl_smoke_metrics.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[2]["weighted_loss"] == ""
    assert csv_rows[2]["note"] == "rollouts recorded; optimizer gate failed"
    assert len((output / "p4_rl_smoke_events.jsonl").read_text().splitlines()) == 4
    svg = (output / "p4_rl_smoke_training_status.svg").read_text()
    assert "stage-separated" in svg
    assert "gate failed" in svg
    assert "KL and entropy were not logged" in svg
    manifest = json.loads((output / "p4_rl_smoke_source_manifest.json").read_text())
    assert manifest["rows"] == 3
    assert manifest["normalized_rollout_events"] == 4
    assert [source["label"] for source in manifest["sources"]] == [
        "replay/trainer_state.json",
        "replay/stdout.log",
        "fresh-online/stdout.log",
        "fresh-online/checkpoint-2/trainer_state.json",
    ]
