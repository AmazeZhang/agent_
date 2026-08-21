import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).parents[1] / "scripts/run_wit_agentic_sft.py"
SPEC = importlib.util.spec_from_file_location("run_wit_agentic_sft", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunWitAgenticSftTest(unittest.TestCase):
    def test_dataset_manifest_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "status": "challenge-ready",
                "purpose": "local-agentic-sft-rl-challenge",
                "image_observation_contains_text_summary": False,
                "image_runtime_handle": "img_1",
                "final_response_format": "Title: <exact title>\\nEvidence: <first sentence-or-no-match>",
                "evidence_extraction": "first_terminal_punctuation_or_360_characters",
                "split_unit": "entity_id-or-synthetic-probe-id",
                "maximum_agent_turns": 4,
                "split_counts": {"dev": 20, "test": 20, "train": 80},
                "task_type_counts": {
                    "candidate-conflict": 48,
                    "clean": 12,
                    "no-match": 24,
                    "transient-tool-failure": 36,
                },
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(MODULE.validate_dataset(root), manifest)
            manifest["image_observation_contains_text_summary"] = True
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "image_observation"):
                MODULE.validate_dataset(root)

    def test_managed_run_excludes_gpu_zero_and_five(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            run_dir = run_root / "run-1"
            run_dir.mkdir()
            environment = {
                "PROJECT4_RUN_ID": "run-1",
                "PROJECT4_RUN_TOKEN": "token",
                "PROJECT4_RUN_DIR": str(run_dir),
                "CUDA_VISIBLE_DEVICES": "1",
            }
            with patch.object(MODULE, "RUN_ROOT", run_root):
                self.assertEqual(MODULE.require_managed_run(environment)[1], "1")
                environment["CUDA_VISIBLE_DEVICES"] = "5"
                with self.assertRaisesRegex(RuntimeError, "GPU0/GPU5"):
                    MODULE.require_managed_run(environment)


if __name__ == "__main__":
    unittest.main()
