import importlib.util
import unittest
from pathlib import Path

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

    def test_every_runtime_message_uses_structured_content(self) -> None:
        source = Path(MODULE.__file__).read_text()
        self.assertNotIn('"content": output}', source)
        self.assertIn('"type": "text", "text": output', source)


if __name__ == "__main__":
    unittest.main()
