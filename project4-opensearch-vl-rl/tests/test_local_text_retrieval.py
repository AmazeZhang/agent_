"""CPU-only contract tests for local Wikipedia text retrieval."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import (  # noqa: E402
    LocalTextIndex,
    build_text_index,
    text_tool_observation,
)


class LocalTextRetrievalTest(unittest.TestCase):
    def documents(self):
        return [
            {
                "entity_id": "Q-BRIDGE",
                "title": "Example Bridge",
                "source": "https://example/wiki/bridge",
                "text": "The Example Bridge opened in 1894 and crosses the River Test.",
            },
            {
                "entity_id": "Q-PAINTING",
                "title": "Blue Landscape",
                "source": "https://example/wiki/painting",
                "text": "Blue Landscape is an impressionist painting completed in 1902.",
            },
        ]

    def test_search_lookup_and_observation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-text-index.") as temporary:
            output = Path(temporary) / "wiki.sqlite"
            build_text_index(
                output,
                self.documents(),
                corpus="synthetic-wikipedia",
                corpus_revision="test-revision",
            )
            with LocalTextIndex(output) as index:
                results = index.search("Which bridge opened in 1894?", top_k=2)
                self.assertEqual(results[0]["entity_id"], "Q-BRIDGE")
                self.assertEqual(results[0]["corpus_revision"], "test-revision")
                exact = index.lookup("Q-PAINTING")
                self.assertEqual(exact["title"], "Blue Landscape")
                self.assertIsNone(index.lookup("Q-MISSING"))

                observation = text_tool_observation(results)
                payload = json.loads(observation.split("\n", 1)[1])
                self.assertEqual(payload["backend"], "local_text_index")
                self.assertEqual(payload["match_count"], 1)

            with self.assertRaises(FileExistsError):
                build_text_index(
                    output,
                    self.documents(),
                    corpus="test",
                    corpus_revision="test",
                )

    def test_rejects_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-bad-text-index.") as temporary:
            output = Path(temporary) / "bad.sqlite"
            with self.assertRaises(ValueError):
                build_text_index(
                    output,
                    [{"entity_id": "Q1", "title": "missing fields"}],
                    corpus="test",
                    corpus_revision="test",
                )

            valid = Path(temporary) / "valid.sqlite"
            build_text_index(
                valid,
                self.documents(),
                corpus="test",
                corpus_revision="test",
            )
            with LocalTextIndex(valid) as index:
                with self.assertRaises(ValueError):
                    index.search("!!!")
                with self.assertRaises(ValueError):
                    index.search("bridge", top_k=21)


if __name__ == "__main__":
    unittest.main()
