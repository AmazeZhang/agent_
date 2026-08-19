"""P3 Phase 4B.1 item 2: episode audit is per traj_uid, never per uid.

One question (uid) has 5 distinct trajectories (traj_uid, GRPO n=5); merging
them into one "episode" would hide per-trajectory totals. These tests drive
the REAL reward manager (EpisodeRewardManager._apply_search_aware_v1) over a
constructed DataProto:

- 5 traj_uids under one uid produce 5 INDEPENDENT search_v1_episode totals
- component sum == placed sum per trajectory (exact cents)
- one traj_uid under two uids is a fail-closed RuntimeError
- search_v1_group is the informational per-uid rollup (n_trajectories=5)
- missing search_v1 / traj_uid metadata is a fail-closed RuntimeError
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from verl import DataProto

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.reward_manager.episode import EpisodeRewardManager  # noqa: E402
from searchr1_repro.search_v1_reward import (  # noqa: E402
    search_step_components,
    terminal_step_components,
    valid_aliases,
)

ALIASES = valid_aliases(["Paris"])
QUESTION = "Which European capital is famous for the Eiffel Tower?"


def useful_step():
    return search_step_components(query="capital of france", status="success",
                                  doc_text="Paris is the capital of France.", gt_aliases=ALIASES,
                                  question=QUESTION, prior_search_count=0)


def irrelevant_step():
    return search_step_components(query="london attractions", status="success",
                                  doc_text="London is the capital of the UK.", gt_aliases=ALIASES,
                                  question=QUESTION, prior_search_count=0)


def leak_step():
    return search_step_components(query="Paris metro", status="success",
                                  doc_text="Paris is the capital of France.", gt_aliases=ALIASES,
                                  question=QUESTION, prior_search_count=0)


def invalid_step():
    return search_step_components(query=None, status="invalid_query", doc_text=None,
                                  gt_aliases=ALIASES, question=QUESTION, prior_search_count=0)


def terminal_step(em: bool, had_evidence: bool):
    return terminal_step_components(r_answer_total=1.0 if em else 0.1, em=em,
                                    had_effective_evidence=had_evidence)


def record(uid: str, traj_uid: str, sv: dict) -> dict:
    return {
        "prompts": torch.tensor([[1, 2, 3]]),
        "responses": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),  # 3 prompt + 3 response
        "uid": np.array([uid], dtype=object),
        "traj_uid": np.array([traj_uid], dtype=object),
        "data_source": np.array(["confirm256"], dtype=object),
        # used only by the DEFAULT (non-v1) placement path
        "episode_rewards": np.array([1.0]),
        "episode_lengths": np.array([1]),
        "search_v1": np.array([sv], dtype=object),
    }


def build_data(records: list[dict]) -> DataProto:
    return DataProto.from_dict(
        tensors={k: torch.cat([r[k] for r in records], dim=0)
                 for k in ("prompts", "responses", "attention_mask")},
        non_tensors={k: np.concatenate([r[k] for r in records], axis=0)
                     for k in ("uid", "traj_uid", "data_source", "episode_rewards", "episode_lengths", "search_v1")},
    )


def placed_cents(sv: dict) -> int:
    """The score placed at THIS record's last token (mirrors the manager)."""
    if sv.get("terminal"):
        return int(sv.get("answer_reward_c", 0)) + int(sv.get("format_reward_c", 0)) + int(sv.get("sce_c", 0))
    return int(sv.get("step_shaping_c", 0))


class StubTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return str(list(ids))


def make_manager():
    return EpisodeRewardManager(tokenizer=None, num_examine=0, search_aware_step_reward=True)


class TestTrajUidEpisodeAudit:
    def test_five_trajectories_one_uid_five_independent_totals(self):
        # q1: t1 useful+correct=145 / t2 irrelevant+correct=100 / t3 useful+wrong=25
        #     t4 leak+correct=80 / t5 invalid+wrong=-10
        recs = [
            record("q1", "t1", useful_step()), record("q1", "t1", terminal_step(True, True)),
            record("q1", "t2", irrelevant_step()), record("q1", "t2", terminal_step(True, False)),
            record("q1", "t3", useful_step()), record("q1", "t3", terminal_step(False, True)),
            record("q1", "t4", leak_step()), record("q1", "t4", terminal_step(True, False)),
            record("q1", "t5", invalid_step()), record("q1", "t5", terminal_step(False, False)),
        ]
        data = build_data(recs)
        reward = make_manager()(data)

        expected = {"t1": 145, "t2": 100, "t3": 25, "t4": 80, "t5": -10}
        seen: dict[str, list[int]] = {}
        for i in range(len(recs)):
            traj = recs[i]["traj_uid"][0]
            episode = data.non_tensor_batch["search_v1_episode"][i]
            assert episode["uid"] == "q1"
            assert episode["n_records"] == 2
            seen.setdefault(traj, []).append(episode["total_reward_c"])
            # the placed score is THIS record's own contribution (step shaping or
            # terminal R_answer+format+sce), not the trajectory total; float32
            # tensor -> tolerance compare
            assert abs(float(reward[i, 2]) - placed_cents(recs[i]["search_v1"][0]) / 100.0) < 1e-6
        for traj, totals in seen.items():
            assert len(totals) == 2 and totals[0] == totals[1] == expected[traj]
        # 5 DISTINCT per-trajectory totals, never merged into one uid-level episode
        distinct = {totals[0] for totals in seen.values()}
        assert distinct == {145, 100, 25, 80, -10}

        # group rollup is informational: 5 trajectories, sum of totals
        group = data.non_tensor_batch["search_v1_group"][0]
        assert group["n_trajectories"] == 5
        assert group["total_reward_c"] == 145 + 100 + 25 + 80 - 10
        assert group["traj_uids"] == ["t1", "t2", "t3", "t4", "t5"]

    def test_same_traj_uid_under_two_uids_is_fail_closed(self):
        recs = [
            record("q1", "tX", terminal_step(True, False)),
            record("q2", "tX", terminal_step(True, False)),
        ]
        with pytest.raises(RuntimeError, match="multiple uids"):
            make_manager()(build_data(recs))

    def test_missing_search_v1_metadata_is_fail_closed(self):
        recs = [record("q1", "t1", terminal_step(True, False))]
        recs[0]["search_v1"] = np.array([None], dtype=object)
        with pytest.raises(RuntimeError, match="no search_v1 metadata"):
            make_manager()(build_data(recs))

    def test_missing_traj_uid_metadata_is_fail_closed(self):
        recs = [record("q1", "t1", terminal_step(True, False))]
        recs[0]["traj_uid"] = np.array([None], dtype=object)
        with pytest.raises(RuntimeError, match="traj_uid grouping metadata"):
            make_manager()(build_data(recs))

    def test_default_mode_never_touches_v1_path(self):
        # search_aware_step_reward=False (official-loose) -> no traj_uid needed,
        # no search_v1 needed; the default placement path is untouched.
        data = build_data([record("q1", "t1", terminal_step(True, False))])
        del data.non_tensor_batch["search_v1"]
        reward = EpisodeRewardManager(tokenizer=StubTokenizer(), num_examine=0, search_aware_step_reward=False)(data)
        assert float(reward[0, 2]) == 1.0
