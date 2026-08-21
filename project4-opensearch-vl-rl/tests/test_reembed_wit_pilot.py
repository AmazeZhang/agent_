"""CPU-only guards for the managed WIT re-embedding task."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reembed_wit_pilot", PROJECT_ROOT / "scripts/reembed_wit_pilot.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load reembed_wit_pilot.py")
REEMBED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REEMBED)


class ReembedWitPilotTest(unittest.TestCase):
    def test_requires_complete_managed_identity(self) -> None:
        with self.assertRaises(RuntimeError):
            REEMBED.require_managed_environment({})
        REEMBED.require_managed_environment(
            {
                "PROJECT4_RUN_ID": "test",
                "PROJECT4_RUN_DIR": "/tmp/run",
                "PROJECT4_RUN_TOKEN": "token",
            }
        )

    def test_decodes_embedded_image_and_rejects_missing_bytes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-reembed-image.") as temporary:
            image_path = Path(temporary) / "image.png"
            Image.new("RGB", (7, 5), "red").save(image_path)
            decoded = REEMBED.decode_image({"bytes": image_path.read_bytes()})
            self.assertEqual(decoded.size, (7, 5))
            with self.assertRaises(ValueError):
                REEMBED.decode_image({"bytes": None})


if __name__ == "__main__":
    unittest.main()
