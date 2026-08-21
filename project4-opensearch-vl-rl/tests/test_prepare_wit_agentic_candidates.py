import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_wit_agentic_candidates.py"
SPEC = importlib.util.spec_from_file_location("prepare_wit_agentic_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareWitAgenticCandidatesTest(unittest.TestCase):
    def test_rank_and_split_are_deterministic_and_entity_disjoint(self) -> None:
        candidates = [
            {
                "row_index": index,
                "entity_id": f"entity-{index}",
                "rank": MODULE.stable_rank("seed", "revision", index),
            }
            for index in range(12)
        ]
        counts = {"train": 6, "dev": 3, "test": 2}
        first = MODULE.assign_splits(candidates, counts)
        second = MODULE.assign_splits(list(reversed(candidates)), counts)
        self.assertEqual(first, second)
        by_split = {
            split: {item["entity_id"] for name, item in first if name == split}
            for split in MODULE.SPLIT_ORDER
        }
        self.assertTrue(by_split["train"].isdisjoint(by_split["dev"] | by_split["test"]))
        self.assertTrue(by_split["dev"].isdisjoint(by_split["test"]))

    def test_query_transform_is_non_identity(self) -> None:
        source = Image.new("RGB", (100, 80), "blue")
        payload = io.BytesIO()
        source.save(payload, format="PNG")
        transformed = MODULE.transform_query(payload.getvalue())
        self.assertEqual(transformed.size, (90, 72))
        self.assertNotEqual(transformed.size, source.size)

    def test_output_root_and_overwrite_are_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "output must be below"):
                MODULE.prepare(
                    Path(directory) / "missing.parquet",
                    outside,
                    revision="revision",
                    source_sha256="0" * 64,
                    split_counts={"train": 1, "dev": 1, "test": 1},
                    seed="seed",
                )

    def test_evidence_filter_requires_english_wikipedia_summary(self) -> None:
        valid = {
            "title": "Example",
            "source": "https://en.wikipedia.org/wiki/Example",
            "summary": "Example is a sufficiently long English summary used for a deterministic unit test record.",
            "language": "en",
        }
        self.assertTrue(MODULE.usable_evidence(valid))
        self.assertFalse(MODULE.usable_evidence({**valid, "language": "de"}))
        self.assertFalse(MODULE.usable_evidence({**valid, "summary": "short"}))


if __name__ == "__main__":
    unittest.main()
