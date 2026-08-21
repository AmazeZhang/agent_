import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "audit_rl_dataset.py"


class AuditRLDatasetTest(unittest.TestCase):
    def test_refuses_overwrite_and_reports_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-audit-test-") as temp_dir:
            tmp_path = Path(temp_dir)
            source = tmp_path / "sample.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question": "What is shown?",
                                "answer": "A bridge \\boxed{bridge}",
                                "images": ["images/a.jpg"],
                                "dataset": "wiki_en",
                            }
                        ),
                        json.dumps(
                            {
                                "question": "这是什么？",
                                "answer": "桥",
                                "images": ["images/a.jpg"],
                                "dataset": "wiki_zh",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = tmp_path / "report.json"

            first = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(first.stdout)["rows"], 2)
            self.assertEqual(report["dataset_counts"], {"wiki_en": 1, "wiki_zh": 1})
            self.assertEqual(
                report["question_language_heuristic"],
                {"contains_cjk": 1, "no_cjk": 1},
            )
            self.assertEqual(report["unique_image_references"], 1)
            self.assertEqual(report["duplicate_image_reference_rows"], 1)
            self.assertEqual(report["answers_containing_boxed"], 1)

            second = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("FileExistsError", second.stderr)


if __name__ == "__main__":
    unittest.main()
