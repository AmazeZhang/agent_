"""CPU-only contract tests for the official-style frozen-Wiki protocol."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import LocalTextIndex, build_text_index

SPEC = importlib.util.spec_from_file_location("evaluate_local_agent", PROJECT_ROOT / "scripts/evaluate_local_agent.py")
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class ProtocolV8Tests(unittest.TestCase):
    def test_response_wrapper_is_required_when_declared(self) -> None:
        task = {"final_response_wrapper": "response-v1", "gold_title": "Alpha", "gold_evidence_sentence": "Alpha fact."}
        self.assertFalse(EVALUATOR.score_final("Title: Alpha\nEvidence: Alpha fact.", task)["format_valid"])
        self.assertTrue(EVALUATOR.score_final("<response>Title: Alpha\nEvidence: Alpha fact.</response>", task)["format_valid"])

    def test_text_search_uses_query_not_entity_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index_path = Path(tmp) / "wiki.sqlite"
            build_text_index(index_path, [{"entity_id": "a", "title": "Alpha subject", "source": "wiki", "text": "Alpha subject has an evidence sentence."}], corpus="test", corpus_revision="1")
            with LocalTextIndex(index_path) as index:
                observation, _ = EVALUATOR.execute_call({"name": "text_search", "arguments": {"q": "Alpha subject", "top_k": 1}}, {}, index, image_search_call_count=0, observation_format="official-provider-v1", tool_protocol="official-local-v1")
                self.assertIn("Alpha subject", observation)
                self.assertNotIn("entity_id", observation)
                error, _ = EVALUATOR.execute_call({"name": "text_search", "arguments": {"entity_id": "a"}}, {}, index, image_search_call_count=0, tool_protocol="official-local-v1")
                self.assertIn("text_search failed", error)

    def test_protocol_tool_whitelist_excludes_legacy_lookup(self) -> None:
        names = [item["function"]["name"] for item in EVALUATOR.tools_for_protocol("official-local-v1")]
        self.assertIn("text_search", names)
        image_schema = next(item for item in EVALUATOR.tools_for_protocol("official-local-v1") if item["function"]["name"] == "image_search")
        self.assertEqual(image_schema["function"]["parameters"]["required"], ["url"])
        self.assertNotIn("text_lookup", names)

    def test_multimodal_protocol_exposes_complete_official_toolbox(self) -> None:
        names = [
            item["function"]["name"]
            for item in EVALUATOR.tools_for_protocol("official-local-multimodal-v1")
        ]
        self.assertEqual(
            names,
            [
                "crop",
                "layout_parsing",
                "web_search",
                "image_search",
                "text_search",
                "perspective_correct",
                "super_resolution",
                "sharpen",
            ],
        )


if __name__ == "__main__":
    unittest.main()
