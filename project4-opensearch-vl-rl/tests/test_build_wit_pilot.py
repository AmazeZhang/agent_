"""CPU-only end-to-end test for paired WIT pilot indexes."""

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "build_wit_pilot", PROJECT_ROOT / "scripts/build_wit_pilot.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load build_wit_pilot.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

from local_retrieval import ExactVisualIndex, LocalTextIndex  # noqa: E402


class BuildWitPilotTest(unittest.TestCase):
    def write_shard(self, path: Path) -> None:
        features_type = pa.struct(
            [
                pa.field("language", pa.list_(pa.string())),
                pa.field("page_url", pa.list_(pa.string())),
                pa.field("page_title", pa.list_(pa.string())),
                pa.field("context_page_description", pa.list_(pa.string())),
                pa.field("context_section_description", pa.list_(pa.string())),
                pa.field("caption_title_and_reference_description", pa.list_(pa.string())),
                pa.field("caption_reference_description", pa.list_(pa.string())),
                pa.field("caption_alt_text_description", pa.list_(pa.string())),
            ]
        )
        table = pa.table(
            {
                "embedding": pa.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    type=pa.list_(pa.float64(), 3),
                ),
                "image_url": ["https://images/bridge.jpg", "https://images/art.jpg"],
                "metadata_url": ["https://meta/bridge", "https://meta/art"],
                "caption_attribution_description": ["a bridge", "an artwork"],
                "wit_features": pa.array(
                    [
                        {
                            "language": ["fr", "en"],
                            "page_url": ["https://fr/bridge", "https://en/bridge"],
                            "page_title": ["Pont", "Example Bridge"],
                            "context_page_description": ["Pont ancien", "Opened in 1894"],
                            "context_section_description": [None, None],
                            "caption_title_and_reference_description": [None, None],
                            "caption_reference_description": [None, None],
                            "caption_alt_text_description": [None, None],
                        },
                        {
                            "language": ["en"],
                            "page_url": ["https://en/art"],
                            "page_title": ["Blue Landscape"],
                            "context_page_description": ["Painted in 1902"],
                            "context_section_description": [None],
                            "caption_title_and_reference_description": [None],
                            "caption_reference_description": [None],
                            "caption_alt_text_description": [None],
                        },
                    ],
                    type=features_type,
                ),
            }
        )
        pq.write_table(table, path)

    def test_builds_searchable_paired_indexes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-wit-build.") as temporary:
            root = Path(temporary)
            shard = root / "shard.parquet"
            output = root / "pilot"
            self.write_shard(shard)
            digest = hashlib.sha256(shard.read_bytes()).hexdigest()
            BUILDER.build_pilot(
                shard,
                output,
                corpus_revision="fixed-revision",
                source_sha256=digest,
                expected_dimension=3,
                batch_size=1,
            )
            visual = ExactVisualIndex(output / "visual")
            self.assertEqual(visual.search([1, 0, 0], top_k=1)[0]["title"], "Example Bridge")
            with LocalTextIndex(output / "text.sqlite") as text:
                self.assertEqual(text.search("opened 1894")[0]["title"], "Example Bridge")
            with self.assertRaises(FileExistsError):
                BUILDER.build_pilot(
                    shard,
                    output,
                    corpus_revision="fixed-revision",
                    source_sha256=digest,
                    expected_dimension=3,
                    batch_size=1,
                )


if __name__ == "__main__":
    unittest.main()
