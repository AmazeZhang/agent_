"""CPU-only tests for RL image reference and decode auditing."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_rl_images", PROJECT_ROOT / "scripts/audit_rl_images.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load audit_rl_images.py")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class AuditRlImagesTest(unittest.TestCase):
    def test_audits_referenced_images(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-audit-images.") as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            Image.new("RGB", (12, 8), "red").save(root / "images/a.jpg")
            Image.new("RGB", (4, 16), "blue").save(root / "images/b.png")
            jsonl = root / "data.jsonl"
            with jsonl.open("w", encoding="utf-8") as handle:
                for reference in ("images/a.jpg", "images/b.png"):
                    handle.write(json.dumps({"images": [reference]}) + "\n")

            report = AUDITOR.audit_images(jsonl, root)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["formats"], {"JPEG": 1, "PNG": 1})
            self.assertEqual(report["dimensions"]["minimum_width"], 4)
            self.assertEqual(report["dimensions"]["maximum_height"], 16)

    def test_rejects_escape_corruption_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-bad-images.") as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "images/bad.jpg").write_bytes(b"not an image")
            outside = root / "outside.jpg"
            Image.new("RGB", (2, 2), "white").save(outside)
            (root / "images/link.jpg").symlink_to(outside)

            for reference in ("../outside.jpg", "images/bad.jpg", "images/link.jpg"):
                jsonl = root / f"case-{reference.replace('/', '_')}.jsonl"
                jsonl.write_text(json.dumps({"images": [reference]}) + "\n")
                with self.subTest(reference=reference), self.assertRaises(
                    (ValueError, FileNotFoundError)
                ):
                    AUDITOR.audit_images(jsonl, root)


if __name__ == "__main__":
    unittest.main()
