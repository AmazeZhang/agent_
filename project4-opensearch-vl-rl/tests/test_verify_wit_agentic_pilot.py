import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/verify_wit_agentic_pilot.py"
SPEC = importlib.util.spec_from_file_location("verify_wit_agentic_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class VerifyWitAgenticPilotTest(unittest.TestCase):
    def test_first_sentence_is_bounded_and_deterministic(self) -> None:
        self.assertEqual(
            MODULE.first_sentence(
                "This sentence is deliberately longer than forty characters. Another sentence."
            ),
            "This sentence is deliberately longer than forty characters.",
        )

    def test_record_requires_two_distinct_tool_steps(self) -> None:
        task = {
            "task_id": "task-1",
            "split": "train",
            "query_image": "images/task-1.jpg",
            "entity_id": "entity-1",
        }
        visual = [
            {
                "title": "Entity One",
                "source": "https://en.wikipedia.org/wiki/Entity_One",
                "summary": "This must not leak through visual search.",
                "entity_id": "entity-1",
                "similarity": 0.98,
                "corpus": "test",
                "corpus_revision": "revision",
            }
        ]
        text = {
            "title": "Entity One",
            "source": "https://en.wikipedia.org/wiki/Entity_One",
            "summary": "Entity One is a sufficiently long evidence sentence for this deterministic test.",
            "entity_id": "entity-1",
            "corpus": "test",
            "corpus_revision": "revision",
        }
        sft, published = MODULE.make_record(task, visual, text)
        roles = [message["from"] for message in sft["conversations"]]
        self.assertEqual(
            roles,
            ["human", "function", "observation", "function", "observation", "gpt"],
        )
        image_observation = json.loads(
            sft["conversations"][2]["value"].split("\n", 1)[1]
        )
        self.assertNotIn("summary", image_observation["results"][0])
        self.assertEqual(
            published["oracle_steps"], ["image_search", "text_lookup", "final"]
        )


if __name__ == "__main__":
    unittest.main()
