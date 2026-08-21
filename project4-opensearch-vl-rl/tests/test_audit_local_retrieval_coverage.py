"""CPU-only tests for answer-independent retrieval coverage reporting."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_local_retrieval_coverage",
    PROJECT_ROOT / "scripts/audit_local_retrieval_coverage.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load audit_local_retrieval_coverage.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AuditLocalRetrievalCoverageTest(unittest.TestCase):
    def test_loads_rows_without_answers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-coverage.") as temporary:
            root = Path(temporary)
            dataset = root / "data.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "dataset": "wiki_en",
                        "question": "Which entity?",
                        "answer": "secret gold answer",
                        "images": ["images/a.jpg"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            selection = {
                "source": {"sha256": AUDIT.sha256_file(dataset)},
                "selection": {"uses_answer_for_selection": False},
                "samples": [
                    {"sample_id": "sample", "row_index": 0, "dataset": "wiki_en"}
                ],
            }
            rows = AUDIT.load_selected_rows(dataset, selection)
            self.assertEqual(rows[0]["question"], "Which entity?")
            self.assertNotIn("answer", rows[0])
            self.assertNotIn("secret gold answer", str(rows))

    def test_confidence_summary_is_explicit_proxy(self) -> None:
        summary = AUDIT.summarise_scores([0.8, 0.65, 0.4])
        self.assertEqual(
            summary["confidence_proxy_counts"], {"high": 1, "medium": 1, "low": 1}
        )


if __name__ == "__main__":
    unittest.main()
