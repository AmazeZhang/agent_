"""CPU-only tests for the official crop rollout probe selection and metrics."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/run_official_crop_rollout_probe.py"
SPEC = importlib.util.spec_from_file_location("run_official_crop_rollout_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OfficialCropRolloutProbeTests(unittest.TestCase):
    def test_policy_kind_distinguishes_full_model_and_adapter(self) -> None:
        self.assertEqual(MODULE.policy_kind(None), "local-full-model")
        self.assertEqual(
            MODULE.policy_kind(Path("adapter")), "local-base-plus-lora"
        )

    def test_relaxed_budget_accepts_official_scale_single_row(self) -> None:
        self.assertEqual(MODULE.validate_probe_budget(1024, 20, [71]), (71,))

    def test_relaxed_budget_rejects_unknown_duplicate_or_excessive_rows(self) -> None:
        invalid = [
            (2048, 20, [71]),
            (1024, 21, [71]),
            (1024, 20, [71, 71]),
            (1024, 20, [72]),
        ]
        for max_tokens, turns, rows in invalid:
            with self.subTest(
                max_tokens=max_tokens, turns=turns, rows=rows
            ), self.assertRaises(ValueError):
                MODULE.validate_probe_budget(max_tokens, turns, rows)

    def test_summary_requires_real_img2_search(self) -> None:
        expected = {
            "name": "crop",
            "arguments": {"image": "img_1", "x": 1, "y": 2, "width": 3, "height": 4},
        }
        result = {
            "fatal": None,
            "turns": [
                {
                    "tool_call": expected,
                    "observation": "Image cropped successfully. New image ID: img_2.",
                    "image_search_cache": False,
                },
                {
                    "tool_call": {"name": "image_search", "arguments": {"url": "img_2"}},
                    "observation": "result",
                    "image_search_cache": False,
                },
            ],
        }
        summary = MODULE.summarize_probe(result, expected)
        self.assertTrue(summary["first_tool_crop"])
        self.assertTrue(summary["first_call_exact_official_expert"])
        self.assertTrue(summary["crop_succeeded"])
        self.assertTrue(summary["followup_uses_img2"])
        self.assertTrue(summary["live_image_search_img2"])
        self.assertEqual(summary["turn_count"], 2)
        self.assertTrue(summary["terminated_with_response"])

    def test_summary_does_not_treat_img1_cache_as_crop_chain(self) -> None:
        expected = {"name": "crop", "arguments": {}}
        result = {
            "fatal": None,
            "turns": [
                {"tool_call": expected, "observation": "crop failed"},
                {
                    "tool_call": {"name": "image_search", "arguments": {"url": "img_1"}},
                    "image_search_cache": True,
                },
            ],
        }
        summary = MODULE.summarize_probe(result, expected)
        self.assertFalse(summary["crop_succeeded"])
        self.assertFalse(summary["followup_uses_img2"])
        self.assertFalse(summary["live_image_search_img2"])


if __name__ == "__main__":
    unittest.main()
