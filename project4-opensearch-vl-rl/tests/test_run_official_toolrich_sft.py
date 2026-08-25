"""CPU-only contract tests for targeted official tool-rich SFT."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.run_official_toolrich_sft import training_config


class OfficialToolrichSftTests(unittest.TestCase):
    def test_direct_cli_import_reaches_managed_run_guard(self) -> None:
        project = Path(__file__).parents[1]
        completed = subprocess.run(
            [
                "/media/imc/data/yzy/agent/project4-opensearch-vl-rl/envs/sft-py311/bin/python",
                str(project / "scripts/run_official_toolrich_sft.py"),
                "--max-steps",
                "1",
            ],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must run inside scripts/run_managed.sh", completed.stderr)
        self.assertNotIn("ModuleNotFoundError", completed.stderr)

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
