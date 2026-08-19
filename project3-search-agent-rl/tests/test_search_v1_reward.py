from __future__ import annotations

import unittest

import numpy as np
import torch

from searchr1_repro.search_v1_reward import (
    ANSWER_LEAK_C,
    ANSWER_REWARD_C,
    EVIDENCE_HIT_C,
    FORMAT_SCORE_C,
    INVALID_C,
    REDUNDANT_C,
    SCE_C,
    answer_leak_in_query,
    episode_totals,
    evidence_hit_in_docs,
    norm_text,
    reward_float,
    search_step_components,
    terminal_step_components,
    valid_aliases,
)
from searchr1_repro.training_audit import build_rollout_audit_records

# Ground-truth used across the fixtures (aliases: "paris", "city of lights",
# "france"), question WITHOUT the answer embedded.
GT = {"target": ["Paris", "the City of Lights"]}
ALIASES = valid_aliases(GT["target"])
QUESTION = "Which European capital is famous for the Eiffel Tower?"
# Real Retriever-returned document bodies that hit the alias.
DOCS_HIT = "Paris is the capital of France and the most populous city of France."
# Retrieved document bodies that do NOT contain any alias.
DOCS_MISS = "London is the capital of the United Kingdom, located on the Thames."


def search_step(query, *, status="success", docs=DOCS_HIT, prior=0, question=QUESTION):
    """Helper: one search step's v1 components (as the env computes them)."""
    return search_step_components(
        query=query,
        status=status,
        doc_text=docs,
        gt_aliases=ALIASES,
        question=question,
        prior_search_count=prior,
    )


def terminal(em, *, had_evidence, r_answer_total=None):
    if r_answer_total is None:
        r_answer_total = 1.0 if em else 0.1
    return terminal_step_components(r_answer_total=r_answer_total, em=em, had_effective_evidence=had_evidence)


class V1DirectAndEvidenceTest(unittest.TestCase):
    # (1) 不搜索直接答对 -> 1.0
    def test_direct_answer_without_search_is_exactly_1_0(self):
        totals = episode_totals([], terminal(em=True, had_evidence=False))
        self.assertEqual(totals["answer_reward_c"], ANSWER_REWARD_C)
        self.assertEqual(totals["format_reward_c"], 0)
        self.assertEqual(totals["evidence_hit_reward_c"], 0)
        self.assertEqual(totals["searched_correct_bonus_c"], 0)
        self.assertEqual(totals["total_reward_c"], ANSWER_REWARD_C)
        self.assertEqual(reward_float(totals["total_reward_c"]), 1.0)

    # (2) 有效证据搜索后答对 -> 1.45
    def test_effective_evidence_search_then_correct_is_1_45(self):
        step = search_step("capital of france")
        self.assertTrue(step["evidence_effective"])
        self.assertEqual(step["step_shaping_c"], EVIDENCE_HIT_C)
        totals = episode_totals([step], terminal(em=True, had_evidence=True))
        self.assertEqual(totals["total_reward_c"], ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C)
        self.assertEqual(reward_float(totals["total_reward_c"]), 1.45)

    # (4) 有证据但答错 -> 0.25 (format 0.1 + evidence 0.15; 无 sce)
    def test_evidence_but_wrong_answer_is_0_25(self):
        step = search_step("capital of france")
        totals = episode_totals([step], terminal(em=False, had_evidence=True))
        self.assertEqual(totals["total_reward_c"], FORMAT_SCORE_C + EVIDENCE_HIT_C)
        self.assertEqual(reward_float(totals["total_reward_c"]), 0.25)

    # sce requires EM AND effective evidence: wrong answer never earns sce even
    # with evidence; EM without evidence never earns sce (checked in test 1).
    def test_sce_requires_both_em_and_evidence(self):
        totals = episode_totals([search_step("capital of france")], terminal(em=False, had_evidence=True))
        self.assertEqual(totals["searched_correct_bonus_c"], 0)


class V1IrrelevantAndLeakGatesTest(unittest.TestCase):
    # (3) T4 <= T1 hard assertion: irrelevant search + memory correct never
    # above the direct-answer 1.0.
    def test_irrelevant_search_then_memory_correct_not_above_direct(self):
        step = search_step("london attractions", docs=DOCS_MISS)
        self.assertFalse(step["evidence_effective"])
        self.assertEqual(step["step_shaping_c"], 0)  # alpha=0: irrelevant search earns nothing
        totals = episode_totals([step], terminal(em=True, had_evidence=False))
        direct = episode_totals([], terminal(em=True, had_evidence=False))
        self.assertLessEqual(totals["total_reward_c"], direct["total_reward_c"])  # T4 <= T1

    # (6) redundant search never beats direct answer. Semantics (2026-08-19
    # clarification): redundant (2nd+) search steps earn NO evidence credit --
    # without it a 2-evidence-search episode would score 1.15 > 1.0 and violate
    # the hard gate. evidence_effective stays True so sce still settles.
    # Diag T6 ("redundant spam x2, evidence, correct") reproduces 1.00 == T1.
    def test_redundant_search_never_beats_direct(self):
        first = search_step("capital of france", prior=0)
        second = search_step("capital of france again", prior=1)  # 2nd search
        self.assertTrue(first["redundant_search"] is False)
        self.assertTrue(second["redundant_search"])
        self.assertTrue(first["evidence_credit"])
        self.assertTrue(second["evidence_effective"])   # sce linkage intact
        self.assertFalse(second["evidence_credit"])     # no +15 on redundant step
        self.assertEqual(second["step_shaping_c"], REDUNDANT_C)
        totals = episode_totals([first, second], terminal(em=True, had_evidence=True))
        direct = episode_totals([], terminal(em=True, had_evidence=False))
        self.assertEqual(totals["total_reward_c"], ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C + REDUNDANT_C)
        self.assertEqual(reward_float(totals["total_reward_c"]), 1.0)  # == T1, diag parity
        self.assertLessEqual(totals["total_reward_c"], direct["total_reward_c"])  # redundant <= direct

    # (7) T7 <= T1 hard assertion + answer-leak audit fields: leak step's
    # evidence is zeroed and it pays -0.20; a leak never beats direct.
    def test_answer_leak_query_never_beats_direct(self):
        leak_query = "Paris population"  # alias in query, not in question
        step = search_step(leak_query, docs=DOCS_HIT)
        self.assertTrue(step["answer_leak"])
        self.assertEqual(step["answer_leak_alias"], "paris")  # audit field recorded
        self.assertFalse(step["evidence_effective"])  # leak zeroes evidence
        self.assertEqual(step["step_shaping_c"], ANSWER_LEAK_C)  # no +15, just -20
        totals = episode_totals([step], terminal(em=True, had_evidence=False))
        direct = episode_totals([], terminal(em=True, had_evidence=False))
        self.assertEqual(totals["total_reward_c"], ANSWER_REWARD_C + ANSWER_LEAK_C)
        self.assertLessEqual(totals["total_reward_c"], direct["total_reward_c"])  # T7 <= T1
        self.assertEqual(reward_float(totals["total_reward_c"]), 0.80)

    # (8) question 本身包含 alias 不得误判: leak=False, evidence stays effective
    def test_question_embedded_alias_not_judged_leak(self):
        question = "Where is the Eiffel Tower in Paris, France?"
        step = search_step("Paris France capital", question=question)
        self.assertFalse(step["answer_leak"])
        self.assertEqual(step["answer_leak_alias"], None)
        self.assertTrue(step["evidence_effective"])
        totals = episode_totals([step], terminal(em=True, had_evidence=True))
        self.assertEqual(totals["total_reward_c"], ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C)

    # too-short / empty aliases are excluded by valid_aliases (rule written in
    # the tests, per Phase 4B requirement 3)
    def test_short_aliases_excluded_from_leak_rule(self):
        aliases = valid_aliases(["P", "", "pa", "paris"])
        self.assertEqual(aliases, [["pa"], ["paris"]])
        self.assertEqual(answer_leak_in_query("P xyz", aliases, QUESTION)["leak"], False)


class V1AliasTokenBoundaryTest(unittest.TestCase):
    """Phase 4B.1 item 4: token-boundary alias matching.

    Two-character aliases hit ONLY as standalone tokens -- never inside a
    longer word ("it" in "britain", "us" in "museum"). Multi-word aliases are
    contiguous token phrases with whitespace/punctuation folded. Question
    exclusion is per-phrase on the same token basis.
    """

    # negative: "it" never matches inside "britain"
    def test_two_char_alias_it_never_hits_inside_britain(self):
        self.assertFalse(evidence_hit_in_docs("Britain is an island in the Atlantic", [["it"]]))
        # same for the leak rule: the query contains "britain", not the token "it"
        self.assertEqual(answer_leak_in_query("britain capital", [["it"]], QUESTION)["leak"], False)

    # negative: "us" never matches inside "museum" (or any other longer word)
    def test_two_char_alias_us_never_hits_inside_museum(self):
        self.assertFalse(evidence_hit_in_docs("The museum opens at nine", [["us"]]))
        self.assertEqual(answer_leak_in_query("museum hours", [["us"]], QUESTION)["leak"], False)
        # "usa" is a different token than "us"
        self.assertFalse(evidence_hit_in_docs("USA is a country", [["us"]]))

    # positive: "US" as a standalone token DOES hit (case-insensitive)
    def test_us_standalone_token_hits(self):
        self.assertTrue(evidence_hit_in_docs("US economy grew last quarter", [["us"]]))
        leak = answer_leak_in_query("US GDP report", [["us"]], QUESTION)
        self.assertEqual(leak, {"leak": True, "alias": "us"})
        # the same token already present in the question is NOT a new leak
        self.assertEqual(
            answer_leak_in_query("US GDP report", [["us"]], "What was the US GDP in 2023?")["leak"], False
        )

    # multi-word aliases: contiguous token phrase, punctuation/whitespace folded
    def test_multi_word_alias_is_contiguous_token_phrase(self):
        alias = [["the", "city", "of", "lights"]]
        self.assertTrue(evidence_hit_in_docs("the city of lights shines", alias))
        self.assertTrue(evidence_hit_in_docs("The City—of—Lights is a nickname.", alias))
        self.assertFalse(evidence_hit_in_docs("the city shines with bright lights", alias))  # not contiguous
        leak = answer_leak_in_query("the city of lights parade", alias, QUESTION)
        self.assertEqual(leak, {"leak": True, "alias": "the city of lights"})
        # alias already present in the question is not a leak
        self.assertEqual(
            answer_leak_in_query("the city of lights parade", alias, "Where is the city of lights?")["leak"],
            False,
        )

    # two-char alias inside a multi-word phrase matches its OWN token only
    def test_two_char_alias_in_phrase_matches_own_token(self):
        self.assertTrue(evidence_hit_in_docs("us states number fifty", [["us", "states"]]))
        self.assertFalse(evidence_hit_in_docs("states of us", [["us", "states"]]))  # reversed, not phrase
        self.assertFalse(evidence_hit_in_docs("museum states", [["us", "states"]]))  # "us" not its own token

    # case-insensitive single-token alias still matches across case
    def test_single_token_alias_case_insensitive(self):
        self.assertTrue(evidence_hit_in_docs("PARIS is the capital", [["paris"]]))


class V1InvalidAndNoSearchTest(unittest.TestCase):
    # (5) invalid query -> -0.20 on the search step
    def test_invalid_query_pays_minus_0_20(self):
        for bad in (None, "", "   ", "x" * 0):
            step = search_step(bad, status="invalid_query")
            self.assertTrue(step["invalid_or_error"])
            self.assertEqual(step["step_shaping_c"], INVALID_C)
        empty = search_step("", status=None)
        self.assertTrue(empty["invalid_or_error"])

    # (9) error observation 不能被算作 evidence: status outside success set
    # makes the step invalid; error text passed as doc_text is never evidence.
    def test_error_observation_is_never_evidence(self):
        error_doc = "Request failed: <information> Paris</information>"  # pathological
        step = search_step("capital of france", status="api_error", docs=error_doc)
        self.assertTrue(step["invalid_or_error"])
        self.assertFalse(step["evidence_effective"])
        self.assertEqual(step["step_shaping_c"], INVALID_C)
        no_result = search_step("zzz", status="no_results", docs=None)
        self.assertFalse(no_result["invalid_or_error"])
        self.assertFalse(no_result["evidence_effective"])  # no doc -> no evidence
        self.assertEqual(no_result["step_shaping_c"], 0)

    # (10) 无搜索 Episode 不得获得任何搜索奖励 (format-only terminal: 0.1)
    def test_no_search_episode_gets_no_shaping(self):
        totals = episode_totals([], terminal(em=False, had_evidence=False))
        self.assertEqual(totals["n_search_steps"], 0)
        self.assertEqual(totals["evidence_hit_reward_c"], 0)
        self.assertEqual(totals["invalid_penalty_c"], 0)
        self.assertEqual(totals["redundant_penalty_c"], 0)
        self.assertEqual(totals["answer_leak_penalty_c"], 0)
        self.assertEqual(totals["total_reward_c"], FORMAT_SCORE_C)
        self.assertEqual(reward_float(totals["total_reward_c"]), 0.1)

    # evidence_hit only checks the REAL document body, never the query text
    # (query chosen alias-free so the leak rule does not fire)
    def test_evidence_checks_docs_not_query(self):
        step = search_step("French capital population", docs=DOCS_MISS)
        self.assertFalse(step["evidence_effective"])
        self.assertEqual(step["step_shaping_c"], 0)


class V1PlacementTest(unittest.TestCase):
    # (11) 多步 reward 只落在对应 step: mirror episode.py placement for a
    # 3-record episode (search1, search2, terminal). Search records carry ONLY
    # their own shaping; R_answer (+format+sce) only on the terminal record.
    def test_step_attribution_placement(self):
        s1 = search_step("capital of france", prior=0)
        s2 = search_step("capital of france again", prior=1)  # alias-free, redundant
        term = terminal(em=True, had_evidence=True)
        # placement exactly as reward_manager/episode.py _apply_search_aware_v1
        placed = [
            int(s1["step_shaping_c"]),
            int(s2["step_shaping_c"]),
            int(term["answer_reward_c"]) + int(term["format_reward_c"]) + int(term["sce_c"]),
        ]
        self.assertEqual(placed[0], EVIDENCE_HIT_C)  # search1: +0.15 only
        self.assertEqual(placed[1], REDUNDANT_C)     # search2 (redundant): no evidence credit, -0.45
        self.assertEqual(placed[2], ANSWER_REWARD_C + SCE_C)  # terminal: 1.0 + 0.30, no search shaping
        self.assertNotIn(ANSWER_REWARD_C, placed[:2])  # R_answer never on search steps
        totals = episode_totals([s1, s2], term)
        self.assertEqual(sum(placed), totals["total_reward_c"])  # 分量和 == 放置和
        # component sum == placed sum (the manager's own assertion, exact cents)
        self.assertEqual(
            totals["answer_reward_c"] + totals["format_reward_c"] + totals["evidence_hit_reward_c"]
            + totals["searched_correct_bonus_c"] + totals["invalid_penalty_c"]
            + totals["redundant_penalty_c"] + totals["answer_leak_penalty_c"],
            totals["total_reward_c"],
        )


class V1PolicyLossMaskTest(unittest.TestCase):
    # (12) Observation token 的 policy loss mask 保持 0: a multi-turn
    # continuation whose prompt embeds the <information> observation must keep
    # the whole prompt (observation included) out of the policy loss.
    def test_observation_tokens_excluded_from_policy_loss(self):
        prompt = "<information>Paris is the capital of France.</information>"
        prompt_ids = [0, 7, 8, 9, 10]  # <information> + doc tokens + close
        response_ids = [31, 32, 33]
        batch = {
            "prompts": torch.tensor([prompt_ids]),
            "responses": torch.tensor([response_ids]),
            "input_ids": torch.tensor([prompt_ids + response_ids]),
            "attention_mask": torch.tensor([[1] * len(prompt_ids) + [1, 1, 1]]),
            "loss_mask": torch.tensor([[0] * len(prompt_ids) + [1, 1, 1]]),
        }
        metadata = {
            "uid": np.array(["q1"], dtype=object),
            "traj_uid": np.array(["t1"], dtype=object),
            "env_step": np.array([1]),
            "retrieval": np.array([{"status": "success", "document_ids": ["7"]}], dtype=object),
            "retrieval_failed": np.array([False]),
            "is_action_valid": np.array([True]),
            "search_v1": np.array([search_step("capital of france")], dtype=object),
        }
        records = build_rollout_audit_records(batch, metadata, multi_turn=True)
        # builder invariant: the audit itself raises if a prompt token ever
        # enters the policy loss; assert the recorded masks show all-zero prompts
        self.assertEqual(records[0]["prompt_policy_loss_tokens"], 0)
        self.assertEqual(records[0]["policy_loss_mask"][: len(prompt_ids)], [0] * len(prompt_ids))
        self.assertEqual(records[0]["policy_loss_tokens"], 3)  # response only


if __name__ == "__main__":
    unittest.main()
