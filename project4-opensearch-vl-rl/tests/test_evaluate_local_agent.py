import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_local_agent.py"
SPEC = importlib.util.spec_from_file_location("evaluate_local_agent", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class EvaluateLocalAgentTest(unittest.TestCase):
    def test_tool_call_parser_is_strict(self) -> None:
        call = MODULE.parse_tool_call(
            '<tool_call>{"name":"image_search","arguments":{"image":"img_1"}}</tool_call>'
        )
        self.assertEqual(call["name"], "image_search")
        self.assertIsNone(MODULE.parse_tool_call("Title: answer"))
        with self.assertRaisesRegex(ValueError, "standalone"):
            MODULE.parse_tool_call(
                'prefix <tool_call>{"name":"image_search","arguments":{}}</tool_call>'
            )

    def test_final_score_requires_both_fields(self) -> None:
        task = {"gold_title": "An Entity", "gold_evidence_sentence": "A fact."}
        score = MODULE.score_final("Title: An Entity\nEvidence: A fact.", task)
        self.assertEqual(
            score,
            {"format_valid": True, "title_exact": True, "evidence_exact": True},
        )
        self.assertFalse(MODULE.score_final("An Entity", task)["format_valid"])
        self.assertTrue(MODULE.is_full_success(None, score))
        self.assertFalse(MODULE.is_full_success("fatal", score))

    def test_every_runtime_message_uses_structured_content(self) -> None:
        source = Path(MODULE.__file__).read_text()
        self.assertNotIn('"content": output}', source)
        self.assertIn('"type": "text", "text": output', source)

    def test_transient_image_failure_is_deterministic(self) -> None:
        call = {"name": "image_search", "arguments": {"image": "img_1"}}
        task = {
            "image_search_failures_before_success": 1,
            "retrieval_results": [
                {
                    "entity_id": "entity-1",
                    "title": "Entity One",
                    "source": "https://example.test/entity-1",
                    "similarity": 0.9,
                    "corpus": "test",
                    "corpus_revision": "one",
                }
            ],
        }
        first, _ = MODULE.execute_call(
            call, task, None, image_search_call_count=1
        )
        second, _ = MODULE.execute_call(
            call, task, None, image_search_call_count=2
        )
        self.assertIn("TRANSIENT_FAILURE", first)
        self.assertIn("entity-1", second)

    def test_official_protocol_uses_url_and_hides_local_fields(self) -> None:
        call = {"name": "image_search", "arguments": {"url": "img_1"}}
        task = {
            "retrieval_results": [
                {
                    "entity_id": "private-entity",
                    "title": "Entity One",
                    "source": "https://example.test/entity-1",
                    "similarity": 0.9,
                }
            ]
        }
        observation, _ = MODULE.execute_call(
            call,
            task,
            None,
            image_search_call_count=1,
            observation_format="official-provider-v1",
            tool_protocol="official-local-v1",
        )
        self.assertIn("Title: Entity One", observation)
        self.assertNotIn("private-entity", observation)
        self.assertNotIn("similarity", observation)

    def test_compact_image_observation_matches_boundary_contract(self) -> None:
        call = {"name": "image_search", "arguments": {"image": "img_1"}}
        task = {
            "retrieval_results": [
                {
                    "entity_id": "entity-1",
                    "title": "Entity One",
                    "source": "https://example.test/entity-1",
                    "similarity": 0.9,
                    "corpus": "test",
                    "corpus_revision": "one",
                }
            ]
        }
        observation, _ = MODULE.execute_call(
            call,
            task,
            None,
            image_search_call_count=1,
            observation_format="boundary-compact-v1",
        )
        self.assertIn('"entity_id": "entity-1"', observation)
        self.assertNotIn("source", observation)
        self.assertNotIn("corpus", observation)

    def test_task_id_selection_preserves_requested_order(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest = {
                "status": "rl-boundary-ready",
                "image_observation_contains_text_summary": False,
                "image_runtime_handle": "img_1",
                "final_response_format": (
                    "Title: <exact title>\\nEvidence: <first sentence-or-no-match>"
                ),
                "evidence_extraction": "first_terminal_punctuation_or_360_characters",
                "maximum_agent_turns": 5,
                "text_lookup_summary_max_characters": 360,
                "image_search_top_k_maximum": 3,
                "tool_observation_schema": "boundary-compact-v1",
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            tasks = [{"task_id": task_id, "split": "dev"} for task_id in ("a", "b")]
            (root / "tasks.jsonl").write_text(
                "".join(json.dumps(task) + "\n" for task in tasks)
            )
            with mock.patch.object(
                MODULE, "validate_dataset_root", return_value=root
            ):
                _, selected = MODULE.load_tasks("dev", 2, root, ["b", "a"])
            self.assertEqual([task["task_id"] for task in selected], ["b", "a"])

    def test_metrics_are_aggregated_without_score_normalisation(self) -> None:
        results = [
            {
                "score": {
                    "format_valid": True,
                    "title_exact": True,
                    "evidence_exact": True,
                    "full_success": True,
                },
                "oracle_path_exact": True,
                "fatal": None,
            },
            {
                "score": {
                    "format_valid": False,
                    "title_exact": False,
                    "evidence_exact": False,
                    "full_success": False,
                },
                "oracle_path_exact": False,
                "fatal": "failure",
            },
        ]
        metrics = MODULE.aggregate_metrics(results)
        self.assertEqual(metrics["full_success"], 0.5)
        self.assertEqual(metrics["fatal_rate"], 0.5)
        with self.assertRaises(ValueError):
            MODULE.aggregate_metrics([])


if __name__ == "__main__":
    unittest.main()
