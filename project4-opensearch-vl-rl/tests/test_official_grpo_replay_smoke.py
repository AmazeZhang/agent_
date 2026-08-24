"""CPU-only integrity tests for the one-step official-provider GRPO replay."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_official_grpo_replay_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_official_grpo_replay_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OfficialGrpoReplaySmokeTest(unittest.TestCase):
    def frozen_inputs(self, root: Path) -> tuple[Path, Path, dict]:
        adapter = root / "adapter"
        dataset = root / "dataset"
        adapter.mkdir()
        dataset.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (dataset / "manifest.json").write_bytes(b"manifest")
        (dataset / "tasks.jsonl").write_bytes(b"tasks")
        report = {
            "mode": "stochastic-rollout-only-no-optimizer-no-api",
            "adapter_sha256": MODULE.sha256_file(
                adapter / "adapter_model.safetensors"
            ),
            "dataset_manifest_sha256": MODULE.sha256_file(dataset / "manifest.json"),
            "tasks_sha256": MODULE.sha256_file(dataset / "tasks.jsonl"),
            "tool_protocol": "official-local-v1",
            "tool_observation_schema": "official-provider-v1",
            "reward_version": "evidence-fidelity-v2",
            "gate": {"passed": True},
            "groups": [
                {
                    "task_id": "task-1",
                    "items": [{"result": {}}, {"result": {}}],
                    "summary": {"advantages": {"fatal_clamped": [-0.5, 0.5]}},
                }
            ],
        }
        return adapter, dataset, report

    def test_report_requires_matching_frozen_inputs_and_signal(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            adapter, dataset, report = self.frozen_inputs(Path(raw_root))
            group = MODULE.validate_rollout_report(
                report, adapter=adapter, dataset_root=dataset, task_id="task-1"
            )
            self.assertEqual(group["task_id"], "task-1")
            report["tasks_sha256"] = "tampered"
            with self.assertRaisesRegex(ValueError, "tasks_sha256"):
                MODULE.validate_rollout_report(
                    report, adapter=adapter, dataset_root=dataset, task_id="task-1"
                )

    def test_zero_advantage_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            adapter, dataset, report = self.frozen_inputs(Path(raw_root))
            report["groups"][0]["summary"]["advantages"]["fatal_clamped"] = [
                0.0,
                0.0,
            ]
            with self.assertRaisesRegex(ValueError, "no optimizer signal"):
                MODULE.validate_rollout_report(
                    report, adapter=adapter, dataset_root=dataset, task_id="task-1"
                )

    def test_turn_replay_appends_only_recorded_observation(self) -> None:
        messages: list[dict] = []
        MODULE.append_turn_messages(
            messages, {"assistant": "call", "observation": "result"}
        )
        self.assertEqual([item["role"] for item in messages], ["assistant", "user"])
        self.assertIn("result", messages[1]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
