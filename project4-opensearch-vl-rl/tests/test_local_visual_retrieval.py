"""CPU-only contract tests for the local visual retrieval pilot."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import (  # noqa: E402
    ExactVisualIndex,
    build_exact_index,
    entity_tool_observation,
    tool_observation,
)


class LocalVisualRetrievalTest(unittest.TestCase):
    def test_exact_ranking_threshold_and_observation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-visual-index.") as temporary:
            output = Path(temporary) / "index"
            build_exact_index(
                output,
                np.array([[1, 0], [0.8, 0.2], [-1, 0]], dtype=np.float32),
                [
                    {"title": "A", "source": "https://a", "entity_id": "a"},
                    {
                        "title": "B",
                        "source": "https://b",
                        "summary": "candidate b",
                        "entity_id": "b",
                    },
                    {"title": "C", "source": "https://c", "entity_id": "c"},
                ],
                corpus="synthetic-contract-test",
                corpus_revision="test-revision",
            )
            index = ExactVisualIndex(output)
            results = index.search([1, 0], top_k=3, minimum_similarity=0.5)
            self.assertEqual([item["entity_id"] for item in results], ["a", "b"])
            self.assertEqual(results[0]["similarity"], 1.0)

            observation = tool_observation(results)
            payload = json.loads(observation.split("\n", 1)[1])
            self.assertEqual(payload["backend"], "local_visual_index")
            self.assertEqual(payload["match_count"], 2)
            self.assertIn("corpus_revision", payload["results"][0])
            entity_payload = json.loads(
                entity_tool_observation(results).split("\n", 1)[1]
            )
            self.assertEqual(entity_payload["evidence_scope"], "entity-candidates-only")
            self.assertNotIn("summary", entity_payload["results"][0])
            batch_results = index.search_batch(
                [[1, 0], [-1, 0]], top_k=1, minimum_similarity=-1.0
            )
            self.assertEqual(batch_results[0][0]["entity_id"], "a")
            self.assertEqual(batch_results[1][0]["entity_id"], "c")

            with self.assertRaises(FileExistsError):
                build_exact_index(
                    output,
                    np.array([[1, 0]], dtype=np.float32),
                    [{"title": "D", "source": "https://d", "entity_id": "d"}],
                    corpus="test",
                    corpus_revision="test",
                )

    def test_rejects_invalid_vectors_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p4-bad-visual-index.") as temporary:
            root = Path(temporary)
            cases = (
                (np.array([[0, 0]], dtype=np.float32), [{"title": "A", "source": "s", "entity_id": "a"}]),
                (np.array([[1, 0]], dtype=np.float32), [{"title": "A", "source": "s"}]),
                (np.array([[1, 0]], dtype=np.float32), []),
            )
            for index, (vectors, metadata) in enumerate(cases):
                with self.subTest(index=index), self.assertRaises((ValueError, RuntimeError)):
                    build_exact_index(
                        root / f"index-{index}",
                        vectors,
                        metadata,
                        corpus="test",
                        corpus_revision="test",
                    )


if __name__ == "__main__":
    unittest.main()
