import unittest

from local_rl import (
    compute_group_advantages,
    score_evidence_fidelity_trajectory,
    score_trajectory,
)


class LocalRlRewardTest(unittest.TestCase):
    def test_extra_valid_lookup_lowers_query_efficiency_not_accuracy(self) -> None:
        task = {
            "task_id": "one",
            "task_type": "candidate-conflict",
            "gold_title": "Target",
            "oracle_steps": ["image_search", "text_lookup", "text_lookup", "final"],
            "retrieval_results": [
                {"title": "Distractor", "entity_id": "q1"},
                {"title": "Target", "entity_id": "q2"},
            ],
        }
        calls = [
            {"name": "image_search", "arguments": {"image": "img_1"}},
            {"name": "text_lookup", "arguments": {"entity_id": "q1"}},
            {"name": "text_lookup", "arguments": {"entity_id": "q2"}},
            {"name": "text_lookup", "arguments": {"entity_id": "q3"}},
        ]
        result = {
            "task_id": "one",
            "task_type": "candidate-conflict",
            "fatal": None,
            "score": {"format_valid": True, "full_success": True},
            "turns": [{"tool_call": call} for call in calls] + [{}],
        }
        reward = score_trajectory(result, task)
        self.assertEqual(reward["r_accuracy"], 1.0)
        self.assertEqual(reward["r_query"], 0.75)
        self.assertAlmostEqual(reward["reward"], 0.95)

    def test_format_is_a_multiplicative_gate(self) -> None:
        task = {
            "task_id": "no",
            "task_type": "no-match",
            "oracle_steps": ["image_search", "final"],
            "retrieval_results": [],
        }
        result = {
            "task_id": "no",
            "task_type": "no-match",
            "fatal": "invalid-tool-format",
            "score": {"format_valid": False, "full_success": False},
            "turns": [{}],
        }
        reward = score_trajectory(result, task)
        self.assertEqual(reward["reward"], 0.0)
        self.assertTrue(reward["hard_mask"])

    def test_fatal_advantage_is_clamped_one_sided(self) -> None:
        advantages = compute_group_advantages([1.0, 0.0], [False, True])
        self.assertEqual(advantages["raw"], [0.5, -0.5])
        self.assertEqual(advantages["fatal_clamped"], [0.5, 0.0])

    def test_evidence_fidelity_reward_grades_partial_answer(self) -> None:
        task = {
            "task_id": "boundary",
            "task_type": "dual-clue-rank2",
            "gold_entity_id": "q2",
            "gold_title": "Target",
            "gold_evidence_sentence": "Target evidence has alpha beta gamma.",
            "oracle_steps": ["image_search", "text_lookup", "text_lookup", "final"],
            "retrieval_results": [],
            "image_search_failures_before_success": 0,
        }
        calls = [
            {
                "tool_call": {"name": "image_search", "arguments": {"image": "img_1"}},
                "observation": "candidates",
            },
            {
                "tool_call": {"name": "text_lookup", "arguments": {"entity_id": "q2"}},
                "observation": "Target evidence has alpha beta gamma. More facts.",
            },
        ]
        result = {
            "task_id": "boundary",
            "task_type": "dual-clue-rank2",
            "fatal": None,
            "final": "Title: Target\nEvidence: Target evidence has alpha beta.",
            "score": {
                "format_valid": True,
                "title_exact": True,
                "evidence_exact": False,
                "full_success": False,
            },
            "turns": calls + [{}],
        }
        reward = score_evidence_fidelity_trajectory(result, task)
        self.assertEqual(reward["r_title"], 1.0)
        self.assertGreater(reward["r_evidence_f1"], 0.0)
        self.assertLess(reward["r_answer"], 1.0)
        self.assertEqual(reward["r_query"], 1.0)
        self.assertGreater(reward["reward"], 0.2)
        self.assertLess(reward["reward"], 1.0)

    def test_query_reward_requires_gold_evidence_in_observation(self) -> None:
        task = {
            "task_id": "missing",
            "task_type": "dual-clue-rank2",
            "gold_entity_id": "q2",
            "gold_title": "Target",
            "gold_evidence_sentence": "Gold evidence.",
            "oracle_steps": ["image_search", "text_lookup", "final"],
            "retrieval_results": [],
        }
        result = {
            "task_id": "missing",
            "task_type": "dual-clue-rank2",
            "fatal": None,
            "final": "Title: Wrong\nEvidence: Wrong.",
            "score": {
                "format_valid": True,
                "title_exact": False,
                "evidence_exact": False,
                "full_success": False,
            },
            "turns": [
                {
                    "tool_call": {"name": "text_lookup", "arguments": {"entity_id": "q2"}},
                    "observation": "Unrelated evidence.",
                },
                {},
            ],
        }
        reward = score_evidence_fidelity_trajectory(result, task)
        self.assertFalse(reward["evidence_path_valid"])
        self.assertEqual(reward["r_query"], 0.0)

    def test_official_text_search_observation_can_satisfy_evidence_path(self) -> None:
        task = {
            "task_id": "official-search",
            "task_type": "dual-clue-rank3",
            "gold_entity_id": "q3",
            "gold_title": "Target",
            "gold_evidence_sentence": "Gold evidence appears in this passage.",
            "oracle_steps": ["image_search", "text_search", "final"],
            "retrieval_results": [],
        }
        result = {
            "task_id": "official-search",
            "task_type": "dual-clue-rank3",
            "fatal": None,
            "final": "<response>Title: Wrong\nEvidence: Wrong.</response>",
            "score": {
                "format_valid": True,
                "title_exact": False,
                "evidence_exact": False,
                "full_success": False,
            },
            "turns": [
                {
                    "tool_call": {
                        "name": "image_search",
                        "arguments": {"url": "img_1"},
                    },
                    "observation": "candidates",
                },
                {
                    "tool_call": {
                        "name": "text_search",
                        "arguments": {"q": "target clues"},
                    },
                    "observation": (
                        "Tool execution result:\nTitle: Target\n"
                        "Summary: Gold evidence appears in this passage."
                    ),
                },
                {},
            ],
        }
        reward = score_evidence_fidelity_trajectory(result, task)
        self.assertTrue(reward["evidence_path_valid"])
        self.assertEqual(reward["r_query"], 1.0)
        self.assertAlmostEqual(reward["reward"], 0.2)

    def test_no_match_after_retry_requires_exact_oracle_tools(self) -> None:
        task = {
            "task_id": "no-retry",
            "task_type": "no-match-after-retry",
            "gold_entity_id": None,
            "gold_title": "NO_MATCH",
            "gold_evidence_sentence": "No local evidence found.",
            "oracle_steps": ["image_search", "image_search", "final"],
            "retrieval_results": [],
        }
        result = {
            "task_id": "no-retry",
            "task_type": "no-match-after-retry",
            "fatal": None,
            "final": "Title: NO_MATCH\nEvidence: No local evidence found.",
            "score": {
                "format_valid": True,
                "title_exact": True,
                "evidence_exact": True,
                "full_success": True,
            },
            "turns": [
                {
                    "tool_call": {"name": "image_search", "arguments": {"image": "img_1"}}
                },
                {},
            ],
        }
        reward = score_evidence_fidelity_trajectory(result, task)
        self.assertEqual(reward["r_answer"], 1.0)
        self.assertEqual(reward["r_query"], 0.0)


if __name__ == "__main__":
    unittest.main()
