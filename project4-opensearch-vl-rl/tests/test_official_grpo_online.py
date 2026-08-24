"""CPU-only fail-closed tests for the fresh-rollout online GRPO loop."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_official_grpo_online.py"
SPEC = importlib.util.spec_from_file_location("run_official_grpo_online", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OfficialGrpoOnlineTest(unittest.TestCase):
    def test_resume_state_and_optimizer_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            adapter = Path(raw_root) / "adapter"
            adapter.mkdir()
            optimizer = adapter / "optimizer.pt"
            optimizer.write_bytes(b"optimizer")
            (adapter / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "algorithm": "single-epoch-on-policy-grpo-replay-fatal-clamped",
                        "global_step": 1,
                    }
                )
            )
            state = MODULE.validate_resume_source(adapter, optimizer, 1)
            self.assertEqual(state["global_step"], 1)
            with self.assertRaisesRegex(ValueError, "global_step"):
                MODULE.validate_resume_source(adapter, optimizer, 2)
            outside = Path(raw_root) / "optimizer.pt"
            outside.write_bytes(b"optimizer")
            with self.assertRaisesRegex(ValueError, "belong"):
                MODULE.validate_resume_source(adapter, outside, 1)

    def test_online_group_requires_variance_format_and_fatal_gate(self) -> None:
        def item(reward: float) -> dict:
            return {
                "result": {"score": {"format_valid": True, "full_success": False}},
                "reward": {
                    "reward": reward,
                    "is_fatal": False,
                    "r_query": 0.0,
                    "r_exact_success": 0.0,
                },
            }

        variable = [item(0.1), item(0.2), item(0.1), item(0.2)]
        group = {
            "summary": MODULE.summarize_group(variable),
        }
        self.assertTrue(MODULE.validate_online_group(group)["passed"])
        constant = [item(0.1)] * 4
        group["summary"] = MODULE.summarize_group(constant)
        with self.assertRaisesRegex(ValueError, "optimizer gate"):
            MODULE.validate_online_group(group)


if __name__ == "__main__":
    unittest.main()
