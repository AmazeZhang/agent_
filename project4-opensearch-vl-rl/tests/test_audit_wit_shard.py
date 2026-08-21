"""CPU-only test for bounded WIT Parquet schema auditing."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_wit_shard", PROJECT_ROOT / "scripts/audit_wit_shard.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load audit_wit_shard.py")
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class AuditWitShardTest(unittest.TestCase):
    def test_reports_schema_without_exporting_values(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-wit-audit.") as temporary:
            shard = Path(temporary) / "sample.parquet"
            table = pa.table(
                {
                    "image": pa.array(
                        [{"bytes": b"image-content", "path": None}],
                        type=pa.struct(
                            [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
                        ),
                    ),
                    "embedding": pa.array([[1.0, 0.0, -1.0]], type=pa.list_(pa.float32())),
                    "page_title": ["Example entity"],
                }
            )
            pq.write_table(table, shard, compression="zstd")

            report = AUDITOR.audit_shard(shard, 1)
            self.assertEqual(report["parquet"]["rows"], 1)
            self.assertEqual(report["samples"][0]["embedding"]["length"], 3)
            image = report["samples"][0]["image"]["fields"]["bytes"]
            self.assertEqual(image["length"], len(b"image-content"))
            self.assertNotIn("image-content", str(report))
            with self.assertRaises(ValueError):
                AUDITOR.audit_shard(shard, 11)


if __name__ == "__main__":
    unittest.main()
