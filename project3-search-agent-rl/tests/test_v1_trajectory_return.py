"""P3 Phase 4B.1 item 1: trajectory-return GRPO advantage (patch 0008).

Construction test (same uid "q1", 5 trajectories) exactly as specified:

  T1 direct correct            [1.00]            total 1.00
  T2 useful search + correct   [0.15, 1.30]      total 1.45
  T3 evidence search + wrong   [0.15, 0.10]      total 0.25
  T4 irrelevant search + wrong [0.00, 1.00]      total 1.00
  T5 invalid search + wrong    [-0.20, 0.10]     total -0.10

Hard assertions:
  - T2's two records get the SAME positive advantage (trajectory-level broadcast)
  - T2 advantage > T1 advantage (a useful search trajectory beats direct)
  - T1 == T4 (identical trajectory returns -> identical advantages)
  - T5 is the lowest
  - the group's 5 trajectory advantages sum to ~0 (GRPO centering)
  - record order permutation does not change any trajectory advantage
  - different uids NEVER mix groups (q2's stats don't leak into q1)
  - Observation tokens keep advantage 0 (response_mask = policy loss mask)

Integration: ray_trainer.compute_advantage with search_v1_trajectory_return=True
must take the new branch (different from the default per-record path).
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

from verl.trainer.ppo.core_algos import compute_grpo_trajectory_return_advantage  # noqa: E402
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage  # noqa: E402

RESP_LEN = 4          # [action, action, placed, observation]
PLACED_IDX = 2
SEARCH_MASK = [1, 1, 1, 0]      # observation token excluded from policy loss
TERMINAL_MASK = [1, 1, 1, 1]

# (traj_uid, per-record placed rewards) -- placed at PLACED_IDX of each record
TRAJECTORIES = {
    "t1": [1.00],                # direct correct
    "t2": [0.15, 1.30],          # useful search + correct
    "t3": [0.15, 0.10],          # evidence search + wrong
    "t4": [0.00, 1.00],          # irrelevant search + correct
    "t5": [-0.20, 0.10],         # invalid search + wrong
}


def expected_adv(returns: list[float]) -> dict[str, float]:
    """Reference trajectory advantages for one uid group (numpy, ddof=1 mirrors
    torch.std sample std)."""
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1.0
    return {tid: (r - mean) / (std + 1e-6) for tid, r in zip(TRAJECTORIES, returns)}


def make_group(uid: str, traj_rewards: dict[str, list[float]], *, shuffle: bool = False) -> dict:
    """Build token_level_rewards / response_mask / uid / traj_uid for a group."""
    records = []  # (uid, traj_uid, placed, mask)
    for tid, rewards in traj_rewards.items():
        for j, r in enumerate(rewards):
            is_terminal = j == len(rewards) - 1
            mask = TERMINAL_MASK if is_terminal else SEARCH_MASK
            records.append((uid, tid, r, mask))
    if shuffle:
        rng = np.random.RandomState(7)
        idx = rng.permutation(len(records))
        records = [records[i] for i in idx]
    rewards = torch.zeros(len(records), RESP_LEN)
    masks = torch.zeros(len(records), RESP_LEN, dtype=torch.long)
    uids = np.empty(len(records), dtype=object)
    trajs = np.empty(len(records), dtype=object)
    for i, (u, tid, r, mask) in enumerate(records):
        rewards[i, PLACED_IDX] = r
        masks[i] = torch.tensor(mask)
        uids[i] = u
        trajs[i] = tid
    return {"token_level_rewards": rewards, "response_mask": masks, "uid": uids, "traj_uid": trajs}


def traj_adv_of(advantages: torch.Tensor, group: dict, tid: str) -> float:
    """The trajectory advantage of tid (identical at every valid token)."""
    for i in range(len(group["traj_uid"])):
        if group["traj_uid"][i] == tid:
            row = advantages[i][group["response_mask"][i] > 0]
            assert len(row) > 0
            return float(row[0])
    raise AssertionError(f"traj {tid} not found")


class TestTrajectoryReturnAdvantage:
    def setup_method(self):
        self.returns = [1.00, 1.45, 0.25, 1.00, -0.10]  # t1..t5
        self.exp = expected_adv(self.returns)

    def test_t2_records_share_identical_positive_advantage(self):
        group = make_group("q1", TRAJECTORIES)
        adv, _ = compute_grpo_trajectory_return_advantage(
            group["token_level_rewards"], group["response_mask"], group["uid"], group["traj_uid"])
        # T2's search record and terminal record carry the SAME advantage
        t2_rows = [i for i, t in enumerate(group["traj_uid"]) if t == "t2"]
        vals = {float(adv[i][group["response_mask"][i] > 0].max()) for i in t2_rows}
        assert len(vals) == 1 and next(iter(vals)) > 0
        assert next(iter(vals)) == pytest.approx(self.exp["t2"])

    def test_t2_beats_t1_t4_equals_t1_t5_lowest(self):
        group = make_group("q1", TRAJECTORIES)
        adv, _ = compute_grpo_trajectory_return_advantage(
            group["token_level_rewards"], group["response_mask"], group["uid"], group["traj_uid"])
        a = {tid: traj_adv_of(adv, group, tid) for tid in TRAJECTORIES}
        assert a["t2"] > a["t1"]                          # useful search > direct
        assert a["t1"] == pytest.approx(a["t4"])          # same return -> same adv
        assert a["t5"] == min(a.values())                 # invalid+wrong lowest
        assert a["t1"] > 0 and a["t5"] < 0
        for tid in TRAJECTORIES:
            assert a[tid] == pytest.approx(self.exp[tid])

    def test_group_mean_is_zero(self):
        group = make_group("q1", TRAJECTORIES)
        adv, _ = compute_grpo_trajectory_return_advantage(
            group["token_level_rewards"], group["response_mask"], group["uid"], group["traj_uid"])
        a = [traj_adv_of(adv, group, tid) for tid in TRAJECTORIES]
        assert abs(sum(a)) < 1e-6

    def test_record_order_permutation_is_invariant(self):
        base = make_group("q1", TRAJECTORIES)
        perm = make_group("q1", TRAJECTORIES, shuffle=True)
        adv_base, _ = compute_grpo_trajectory_return_advantage(
            base["token_level_rewards"], base["response_mask"], base["uid"], base["traj_uid"])
        adv_perm, _ = compute_grpo_trajectory_return_advantage(
            perm["token_level_rewards"], perm["response_mask"], perm["uid"], perm["traj_uid"])
        for tid in TRAJECTORIES:
            assert traj_adv_of(adv_base, base, tid) == pytest.approx(traj_adv_of(adv_perm, perm, tid))

    def test_different_uids_never_mix(self):
        # q1 alone
        g1 = make_group("q1", TRAJECTORIES)
        adv1, _ = compute_grpo_trajectory_return_advantage(
            g1["token_level_rewards"], g1["response_mask"], g1["uid"], g1["traj_uid"])
        a1 = {tid: traj_adv_of(adv1, g1, tid) for tid in TRAJECTORIES}

        # q2 with very different returns; q1 values must not change
        q2 = {"u6": [3.0], "u7": [0.5, 0.5], "u8": [0.2], "u9": [0.1, 0.05], "u10": [-1.0, 0.05]}
        g2 = make_group("q1", TRAJECTORIES)
        for k, v in g2.items():
            g2[k] = np.concatenate([np.asarray(g2[k], dtype=object), np.asarray(make_group("q2", q2)[k], dtype=object)]) \
                if isinstance(g2[k], np.ndarray) and g2[k].dtype == object else torch.cat([g2[k], make_group("q2", q2)[k]])
        adv2, _ = compute_grpo_trajectory_return_advantage(
            g2["token_level_rewards"], g2["response_mask"], g2["uid"], g2["traj_uid"])
        for tid in TRAJECTORIES:
            assert traj_adv_of(adv2, g2, tid) == pytest.approx(a1[tid])

    def test_observation_tokens_keep_zero_advantage(self):
        group = make_group("q1", TRAJECTORIES)
        adv, _ = compute_grpo_trajectory_return_advantage(
            group["token_level_rewards"], group["response_mask"], group["uid"], group["traj_uid"])
        for i in range(len(group["traj_uid"])):
            mask = group["response_mask"][i]
            assert torch.all(adv[i][mask == 0] == 0)
            assert torch.all(adv[i][mask > 0] == adv[i][mask > 0][0])  # uniform broadcast

    def test_single_trajectory_group_matches_legacy_behavior(self):
        # one traj per uid: mean 0 / std 1 (mirrors compute_grpo_outcome_advantage)
        g = make_group("q1", {"t1": [0.15, 1.30]})
        adv, _ = compute_grpo_trajectory_return_advantage(
            g["token_level_rewards"], g["response_mask"], g["uid"], g["traj_uid"])
        assert traj_adv_of(adv, g, "t1") == pytest.approx(1.45 / (1 + 1e-6))  # (1.45-0)/(1+eps)

    def test_same_traj_uid_under_two_uids_fails_closed(self):
        rewards = torch.zeros(2, RESP_LEN)
        masks = torch.ones(2, RESP_LEN, dtype=torch.long)
        uids = np.array(["q1", "q2"], dtype=object)
        trajs = np.array(["tX", "tX"], dtype=object)
        rewards[:, PLACED_IDX] = 1.0
        with pytest.raises(ValueError, match="multiple uids"):
            compute_grpo_trajectory_return_advantage(rewards, masks, uids, trajs)


class TestComputeAdvantageIntegration:
    def test_flag_routes_to_trajectory_return_branch(self):
        g = make_group("q1", TRAJECTORIES)
        data = DataProto.from_dict(
            tensors={
                "token_level_rewards": g["token_level_rewards"],
                "response_mask": g["response_mask"],
                "loss_mask": g["response_mask"],  # multi-turn: loss_mask tail is the GRPO mask
            },
            non_tensors={"uid": g["uid"], "traj_uid": g["traj_uid"]},
        )
        adv_v1 = compute_advantage(
            data, adv_estimator=AdvantageEstimator.GRPO, multi_turn=True,
            search_v1_trajectory_return=True,
        )
        for tid in TRAJECTORIES:
            assert traj_adv_of(adv_v1.batch["advantages"], g, tid) == pytest.approx(
                expected_adv([1.00, 1.45, 0.25, 1.00, -0.10])[tid])

    def test_default_path_is_unchanged_per_record_normalization(self):
        g = make_group("q1", TRAJECTORIES)
        data = DataProto.from_dict(
            tensors={
                "token_level_rewards": g["token_level_rewards"],
                "response_mask": g["response_mask"],
                "loss_mask": g["response_mask"],
            },
            non_tensors={"uid": g["uid"], "traj_uid": g["traj_uid"]},
        )
        adv_default = compute_advantage(data, adv_estimator=AdvantageEstimator.GRPO, multi_turn=True)
        # the default per-record path normalizes every record's PARTIAL sum
        # inside the group: a +0.15 search record of t2 gets a DIFFERENT
        # (negative) value than the trajectory-return path gives it
        t2_search_row = next(i for i, t in enumerate(g["traj_uid"]) if t == "t2")
        default_val = float(adv_default.batch["advantages"][t2_search_row][g["response_mask"][t2_search_row] > 0][0])
        v1_val = expected_adv([1.00, 1.45, 0.25, 1.00, -0.10])["t2"]
        assert default_val != pytest.approx(v1_val)
        assert default_val < 0  # the Phase 4B.1 bug: useful search step scored negative
