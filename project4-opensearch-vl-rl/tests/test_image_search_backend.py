"""CPU-only safety and compatibility tests for local image_search."""

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval.image_search_backend import (  # noqa: E402
    resolve_local_image,
    validate_encoder_revision,
)


class ImageSearchBackendTest(unittest.TestCase):
    def test_requires_exact_encoder_revision(self) -> None:
        digest = "a" * 64
        validate_encoder_revision({"corpus_revision": f"data+{digest}"}, digest)
        with self.assertRaises(ValueError):
            validate_encoder_revision({"corpus_revision": "data+different"}, digest)

    def test_local_path_guard_rejects_escape_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-image-backend.") as temporary:
            base = Path(temporary)
            root = base / "allowed"
            root.mkdir()
            valid = root / "valid.png"
            outside = base / "outside.png"
            Image.new("RGB", (3, 3), "red").save(valid)
            Image.new("RGB", (3, 3), "blue").save(outside)
            self.assertEqual(resolve_local_image(valid, root), valid.resolve())
            self.assertEqual(resolve_local_image(Path("valid.png"), root), valid.resolve())
            with self.assertRaises(ValueError):
                resolve_local_image(outside, root)
            link = root / "link.png"
            link.symlink_to(outside)
            with self.assertRaises(ValueError):
                resolve_local_image(link, root)


if __name__ == "__main__":
    unittest.main()
