"""CPU-only tests for the non-overwriting official-provider QVA projection."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts/build_official_provider_rl_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_official_provider_rl_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildOfficialProviderDatasetTest(unittest.TestCase):
    def test_tasks_are_byte_identical_and_output_is_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            project_data = Path(raw_root)
            source = project_data / "datasets/processed/source"
            output = project_data / "datasets/processed/output"
            (source / "images").mkdir(parents=True)
            (source / "images/query.jpg").write_bytes(b"image")
            tasks = b'{"task_id":"one"}\n'
            (source / "tasks.jsonl").write_bytes(tasks)
            (source / "dataset_info.json").write_text("{}\n", encoding="utf-8")
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "tool_protocol": "official-local-v1",
                        "sft_sha256": {"train": "stale"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            expected = MODULE.sha256_file(source / "tasks.jsonl")
            with mock.patch.object(MODULE, "PROJECT_DATA", project_data), mock.patch.object(
                MODULE, "EXPECTED_SOURCE_TASKS_SHA256", expected
            ):
                MODULE.build(source, output)
                self.assertEqual((output / "tasks.jsonl").read_bytes(), tasks)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["rows_modified"], 0)
                self.assertEqual(
                    manifest["tool_observation_schema"], "official-provider-v1"
                )
                self.assertNotIn("sft_sha256", manifest)
                with self.assertRaises(FileExistsError):
                    MODULE.build(source, output)


if __name__ == "__main__":
    unittest.main()
