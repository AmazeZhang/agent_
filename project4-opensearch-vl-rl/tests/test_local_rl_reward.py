import unittest

from local_rl import compute_group_advantages, score_trajectory


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


if __name__ == "__main__":
    unittest.main()
