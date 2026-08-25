import copy
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
    def test_relaxed_generation_budget_is_bounded(self) -> None:
        self.assertEqual(MODULE.validate_generation_budget(1024, 20), (1024, 20))
        for tokens, turns in ((31, 20), (1025, 20), (256, 1), (256, 21)):
            with self.subTest(tokens=tokens, turns=turns), self.assertRaises(
                ValueError
            ):
                MODULE.validate_generation_budget(tokens, turns)

    def test_cli_accepts_an_explicit_full_model_root(self) -> None:
        with mock.patch(
            "sys.argv",
            [
                "evaluate_local_agent.py",
                "--model-root",
                "/tmp/official-model",
                "--output",
                "x",
            ],
        ):
            self.assertEqual(
                MODULE.parse_args().model_root, Path("/tmp/official-model")
            )

    def test_train_split_is_available_for_pre_rl_rollout_only_gate(self) -> None:
        with mock.patch(
            "sys.argv", ["evaluate_local_agent.py", "--split", "train", "--output", "x"]
        ):
            self.assertEqual(MODULE.parse_args().split, "train")

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

    def test_official_parser_allows_only_one_well_formed_think_prefix(self) -> None:
        text = (
            "<think>Inspect the image first.</think>\n"
            '<tool_call>{"name":"crop","arguments":{"image":"img_1",'
            '"x":0,"y":0,"width":10,"height":10}}</tool_call>'
        )
        call = MODULE.parse_tool_call(text, allow_official_think_prefix=True)
        self.assertEqual(call["name"], "crop")
        with self.assertRaisesRegex(ValueError, "standalone"):
            MODULE.parse_tool_call(
                "unsafe-prefix " + text, allow_official_think_prefix=True
            )

    def test_official_final_score_allows_think_prefix_when_declared(self) -> None:
        task = {
            "gold_title": "Alpha",
            "gold_evidence_sentence": "Alpha fact.",
            "final_response_wrapper": "response-v1",
            "allow_official_think_prefix": True,
        }
        score = MODULE.score_final(
            "<think>Done.</think><response>Title: Alpha\nEvidence: Alpha fact.</response>",
            task,
        )
        self.assertTrue(score["format_valid"])

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

    def test_multimodal_rollout_injects_crop_pixels_and_searches_img2(self) -> None:
        from PIL import Image
        from local_retrieval import LocalTextIndex, build_text_index

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            Image.new("RGB", (100, 80), "white").save(root / "images/source.png")
            index_path = root / "text.sqlite"
            build_text_index(
                index_path,
                [{"entity_id": "a", "title": "Alpha", "source": "wiki", "text": "Alpha fact."}],
                corpus="test",
                corpus_revision="1",
            )
            outputs = iter(
                [
                    '<tool_call>{"name":"crop","arguments":{"image":"img_1","x":10,"y":15,"width":30,"height":20}}</tool_call>',
                    '<tool_call>{"name":"image_search","arguments":{"url":"img_2"}}</tool_call>',
                    '<response>Title: Alpha\nEvidence: Alpha fact.</response>',
                ]
            )
            message_snapshots = []

            def fake_generate(_model, _processor, messages, **_kwargs):
                message_snapshots.append(copy.deepcopy(messages))
                return next(outputs), 10, 5

            seen = []

            def visual_lookup(image, top_k):
                seen.append((image.size, top_k))
                return [{"title": "Alpha", "source": "wiki", "entity_id": "hidden"}]

            task = {
                "task_id": "crop-probe",
                "task_type": "crop-live-search",
                "split": "dev",
                "query_image": "images/source.png",
                "user_prompt": "Crop then search.",
                "gold_title": "Alpha",
                "gold_evidence_sentence": "Alpha fact.",
                "final_response_wrapper": "response-v1",
                "oracle_steps": ["crop", "image_search", "final"],
            }
            with LocalTextIndex(index_path) as text_index, mock.patch.object(
                MODULE, "generate_turn", side_effect=fake_generate
            ):
                result = MODULE.evaluate_task(
                    None,
                    None,
                    task,
                    text_index,
                    max_new_tokens=64,
                    dataset_root=root,
                    observation_format="official-provider-v1",
                    maximum_turns=3,
                    tool_protocol="official-local-multimodal-v1",
                    visual_lookup=visual_lookup,
                )
            crop_response = message_snapshots[1][-1]["content"]
            self.assertEqual(crop_response[0]["type"], "image")
            self.assertEqual(crop_response[0]["image"].size, (30, 20))
            self.assertIn("New image ID: img_2", crop_response[1]["text"])
            self.assertEqual(seen, [((30, 20), 5)])
            self.assertEqual(result["tool_names"], ["crop", "image_search"])
            self.assertIsNone(result["fatal"])

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
