import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/build_wit_rl_boundary_v6.py"
SPEC = importlib.util.spec_from_file_location("build_wit_rl_boundary_v6", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(entity_id: str, title: str, summary: str) -> dict:
    return {"entity_id": entity_id, "title": title, "summary": summary}


class BuildWitRlBoundaryV6Test(unittest.TestCase):
    def candidates(self) -> list[dict]:
        return [
            result("q1", "First title", "Distractor evidence contains exclusionary material."),
            result("q2", "Second title", "Target evidence has photosynthesis and chloroplasts."),
            result("q3", "Third title", "Another candidate concerns architecture and stations."),
        ]

    def test_boundary_clues_are_unique_and_title_independent(self) -> None:
        clues = MODULE.boundary_clues(self.candidates(), 2)
        self.assertIsNotNone(clues)
        positives, exclusion = clues
        self.assertEqual(set(positives), {"photosynthesis", "chloroplasts"})
        self.assertEqual(exclusion, "exclusionary")

    def test_rank3_and_transient_oracles_fit_five_turns(self) -> None:
        source = {
            "task_id": "wit-one",
            "split": "train",
            "retrieval_results": self.candidates(),
        }
        _, rank3 = MODULE.boundary_example(
            source, "images/a.jpg", self.candidates(), target_rank=3, transient=False
        )
        self.assertEqual(
            rank3["oracle_steps"],
            ["image_search", "text_lookup", "text_lookup", "text_lookup", "final"],
        )
        _, transient = MODULE.boundary_example(
            source, "images/a.jpg", self.candidates(), target_rank=2, transient=True
        )
        self.assertEqual(
            transient["oracle_steps"],
            ["image_search", "image_search", "text_lookup", "text_lookup", "final"],
        )

    def test_no_match_after_retry_is_explicit_synthetic_probe(self) -> None:
        _, task = MODULE.no_match_after_retry_example("dev", 0, "images/no.png")
        self.assertTrue(task["synthetic_safety_probe"])
        self.assertEqual(task["image_search_failures_before_success"], 1)
        self.assertEqual(task["oracle_steps"], ["image_search", "image_search", "final"])


if __name__ == "__main__":
    unittest.main()
