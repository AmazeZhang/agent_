import unittest

from scripts.audit_stochastic_rollout_groups import summarize_group


def _item(reward: float, *, fatal: bool = False, success: bool = False) -> dict:
    return {
        "result": {
            "score": {"full_success": success, "format_valid": not fatal},
        },
        "reward": {
            "reward": reward,
            "is_fatal": fatal,
            "r_query": 1.0 if reward else 0.0,
            "r_accuracy": 1.0 if success else 0.0,
        },
    }


class StochasticRolloutAuditTest(unittest.TestCase):
    def test_group_summary_preserves_real_variance_and_query_only_count(self) -> None:
        summary = summarize_group(
            [_item(1.0, success=True), _item(0.2), _item(0.0, fatal=True)]
        )
        self.assertGreater(summary["reward_population_variance"], 0.0)
        self.assertEqual(summary["unique_rewards"], [0.0, 0.2, 1.0])
        self.assertEqual(summary["query_only_reward_count"], 1)
        self.assertEqual(summary["fatal_count"], 1)
        self.assertEqual(summary["advantages"]["fatal_clamped"][-1], 0.0)


if __name__ == "__main__":
    unittest.main()
