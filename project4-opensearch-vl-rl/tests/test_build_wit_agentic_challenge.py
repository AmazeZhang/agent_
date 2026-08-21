import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/build_wit_agentic_challenge.py"
SPEC = importlib.util.spec_from_file_location("build_wit_agentic_challenge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BuildWitAgenticChallengeTest(unittest.TestCase):
    def source_task(self) -> dict:
        return {
            "task_id": "wit-1",
            "split": "train",
            "gold_final": "Title: First\nEvidence: First evidence sentence.",
            "retrieval_results": [
                {
                    "entity_id": "first",
                    "title": "First",
                    "summary": "First evidence sentence. More facts.",
                    "source": "https://example.test/first",
                    "corpus": "test",
                    "corpus_revision": "one",
                    "similarity": 0.9,
                },
                {
                    "entity_id": "second",
                    "title": "Second",
                    "summary": "Second evidence mentions photosynthesis. More facts.",
                    "source": "https://example.test/second",
                    "corpus": "test",
                    "corpus_revision": "one",
                    "similarity": 0.8,
                },
            ],
        }

    def test_conflict_keyword_is_unique_and_not_in_title(self) -> None:
        task = self.source_task()
        self.assertEqual(MODULE.conflict_keyword(task), "photosynthesis")
        record, published = MODULE.conflict_example(task, "images/a.jpg", "photosynthesis")
        self.assertEqual(published["gold_title"], "Second")
        self.assertEqual(
            published["oracle_steps"],
            ["image_search", "text_lookup", "text_lookup", "final"],
        )
        self.assertEqual(record["conversations"][-1]["from"], "gpt")

    def test_transient_and_no_match_have_explicit_oracles(self) -> None:
        _, transient = MODULE.transient_example(self.source_task(), "images/a.jpg")
        self.assertEqual(transient["image_search_failures_before_success"], 1)
        self.assertEqual(transient["oracle_steps"][:2], ["image_search", "image_search"])
        _, no_match = MODULE.no_match_example("dev", 0, "images/no-match.png")
        self.assertTrue(no_match["synthetic_safety_probe"])
        self.assertEqual(no_match["gold_title"], "NO_MATCH")


if __name__ == "__main__":
    unittest.main()
