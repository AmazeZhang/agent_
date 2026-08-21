"""CPU-only tests for deterministic stratified RL audit selection."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_stratified_rl_audit",
    PROJECT_ROOT / "scripts/select_stratified_rl_audit.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load select_stratified_rl_audit.py")
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


class StratifiedSelectionTest(unittest.TestCase):
    def write_rows(self, path: Path, answer_prefix: str) -> None:
        rows = [
            {"dataset": "large", "answer": f"{answer_prefix}-{index}"}
            for index in range(8)
        ]
        rows.extend(
            {"dataset": "small", "answer": f"{answer_prefix}-{index}"}
            for index in range(2)
        )
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def test_proportional_and_answer_independent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-stratified.") as temporary:
            root = Path(temporary)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            self.write_rows(first, "gold-a")
            self.write_rows(second, "completely-different-gold-b")
            manifest_a = SELECTOR.build_manifest(first, 5, "fixed-seed")
            manifest_b = SELECTOR.build_manifest(second, 5, "fixed-seed")

            self.assertEqual(
                manifest_a["selected_dataset_counts"], {"large": 4, "small": 1}
            )
            self.assertEqual(manifest_a["samples"], manifest_b["samples"])
            self.assertFalse(manifest_a["selection"]["uses_answer_for_selection"])

    def test_rejects_invalid_sample_size(self) -> None:
        with self.assertRaises(ValueError):
            SELECTOR.proportional_quotas({"a": 2}, 3)


if __name__ == "__main__":
    unittest.main()
