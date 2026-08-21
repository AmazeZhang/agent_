"""CPU-only tests for guarded ZIP audit and extraction."""

import importlib.util
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "safe_extract_zip", PROJECT_ROOT / "scripts/safe_extract_zip.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load safe_extract_zip.py")
SAFE_ZIP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAFE_ZIP
SPEC.loader.exec_module(SAFE_ZIP)


class SafeZipTest(unittest.TestCase):
    def audit(self, archive: Path):
        return SAFE_ZIP.audit_archive(
            archive,
            max_files=10,
            max_uncompressed_bytes=1 << 20,
            max_ratio=100.0,
        )

    def test_audit_and_extract_valid_archive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-safe-zip.") as temporary:
            root = Path(temporary)
            archive = root / "valid.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("images/a.txt", b"alpha")
                handle.writestr("images/b.txt", b"beta")

            audit = self.audit(archive)
            self.assertEqual(audit.files, 2)
            output = root / "output"
            SAFE_ZIP.extract_archive(archive, output)
            self.assertEqual((output / "images/a.txt").read_bytes(), b"alpha")
            with self.assertRaises(FileExistsError):
                SAFE_ZIP.extract_archive(archive, output)

    def test_rejects_traversal_and_backslash(self) -> None:
        for index, name in enumerate(("../escape", "/absolute", "dir\\escape")):
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix=f"p4-bad-zip-{index}."
            ) as temporary:
                archive = Path(temporary) / "bad.zip"
                with zipfile.ZipFile(archive, "w") as handle:
                    handle.writestr(name, b"unsafe")
                with self.assertRaises(ValueError):
                    self.audit(archive)

    def test_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-link-zip.") as temporary:
            archive = Path(temporary) / "link.zip"
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr(info, "target")
            with self.assertRaises(ValueError):
                self.audit(archive)


if __name__ == "__main__":
    unittest.main()
