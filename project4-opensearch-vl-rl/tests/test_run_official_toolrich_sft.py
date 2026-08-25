"""CPU-only contract tests for targeted official tool-rich SFT."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_official_toolrich_sft import training_config


class OfficialToolrichSftTests(unittest.TestCase):
    def test_config_continues_existing_adapter_with_frozen_qlora(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            config = training_config(root / "run", 1, adapter, 0)
        self.assertEqual(config["adapter_name_or_path"], str(adapter))
        self.assertFalse(config["create_new_adapter"])
        self.assertEqual(config["quantization_method"], "bnb")
        self.assertEqual(config["quantization_type"], "nf4")
        self.assertTrue(config["double_quantization"])
        self.assertEqual(config["max_samples"], 97)
        self.assertNotIn("resume_from_checkpoint", config)

    def test_resume_config_points_trainer_and_adapter_to_same_checkpoint(self) -> None:
        checkpoint = Path("/runs/checkpoint-1")
        config = training_config(Path("/new-run"), 5, checkpoint, 1)
        self.assertEqual(config["adapter_name_or_path"], str(checkpoint))
        self.assertEqual(config["resume_from_checkpoint"], str(checkpoint))


if __name__ == "__main__":
    unittest.main()
