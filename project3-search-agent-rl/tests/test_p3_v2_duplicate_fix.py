"""P3 Search-aware clean v2: duplicate-record fix CPU gate suite.

Root cause (2026-08-22 smoke `...-fsdp6-b66-n5-s0-20260822a`): the rollout loop
emits EXACTLY one record per (traj_uid, env_step) -- 800 records for 330
trajectories -- but `adjust_batch(..., mode="copy")` copy-padded random real
rows to reach the DP-divisible batch size (800 % 6 = 2 -> 4 copies), so the
training batch contained 4 duplicate (uid, traj_uid, env_step) identities that
double-counted their reward and policy gradient.

Fix (2026-08-23): `adjust_batch` now appends SYNTHETIC padding rows
(is_padding=True, unique "__pad__{k}" uid/traj_uid, env_step=-1, one attended
token at prompt position 0 -> response part all-zero -> zero response_mask ->
zero loss, zero reward -> own size-1 GRPO group -> advantage exactly 0;
loss_mask zeroed for multi-turn batches) plus a fail-closed invariant
`assert_unique_training_records` enforced in ray_trainer BEFORE reward placement
and advantage computation. The single attended prompt token keeps the actor's
unpad from producing an empty sequence (all-zero attention crashed
actor_rollout_compute_log_prob with "cannot reshape tensor of 0 elements" in
the 2026-08-23a smoke); `dp_actor.update_policy` runs zero-response-mask
micro-batches as a COLLECTIVE-UNIFORM zero-loss forward+backward (token-mean
0/0 would be NaN; a per-rank `continue` skip would desync NCCL collective
counts and deadlock -- the deterministic 20260823b/c hang, see D13).

Tests (CPU-only, user requirements 8/9 of the fix directive):
  D1  multi-chunk concat: 330 trajectories (66 uid x 5) / 800 records, chunk
      boundaries splitting trajectories, no cross-chunk duplicate append
  D2  adjust_batch pads to DP divisibility with synthetic rows (800 % 6 = 2
      -> 4 padding rows) and keeps every real record exactly once
  D3  exact divisibility (804 % 6 = 0) -> batch returned unchanged, no padding
  D4  fail-closed invariant raises on an injected duplicate (uid, traj_uid,
      env_step) and names both record indices
  D5  no false deletion / no content-based dedup: byte-identical content from
      two different trajectories keeps BOTH records; "delete" mode removes
      whole rows only
  D6  trajectory returns are NOT double-counted: per-traj return after
      adjust_batch equals the pre-padding per-traj return (old copy-padding
      inflated 4 trajectories; padding rows contribute 0)
  D7  record total == sum of actual steps per trajectory; per-traj record
      counts unchanged by adjust_batch
  D8  padding rows are inert in the v2 reward manager (skipped, no reward
      placed; real rows placed; version:"v2" in episode/group aggregates) and
      in the audit (padding rows skipped, is_padding=False on real rows)
  D9  padding rows get advantage exactly 0.0 from the runtime v2 trajectory
      advantage (own size-1 uid group)

Run (CPU-only):  CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_p3_v2_duplicate_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "verl-agent-v2"))

from verl import DataProto  # noqa: E402

from agent_system.multi_turn_rollout.utils import (  # noqa: E402
    adjust_batch,
    assert_unique_training_records,
)

N_UID = 66
N_TRAJ = 330  # 66 uid x 5 (GRPO n=5), mirroring the smoke run
# 190 trajectories x 2 steps + 140 x 3 steps = 800 records (smoke's total)
STEP_COUNTS = [2] * 190 + [3] * 140
P_LEN = 32  # prompt width
R_LEN = 8  # response width


def _identity_plan() -> list[tuple[str, str, int]]:
    """(uid, traj_uid, env_step) for the 800 records: traj index i belongs to
    uid i % 66 (5 trajectories per uid), steps 0..(n_steps-1)."""
    plan = []
    for i, n_steps in enumerate(STEP_COUNTS):
        uid = f"uid{i % N_UID:08x}"
        tid = f"traj{i:08x}"
        for step in range(n_steps):
            plan.append((uid, tid, step))
    assert len(plan) == 800
    return plan


PLAN = _identity_plan()
TRAJ_N_STEPS = {tid: 0 for _, tid, _ in PLAN}
for _, tid, step in PLAN:
    TRAJ_N_STEPS[tid] = max(TRAJ_N_STEPS[tid], step + 1)


def _is_terminal(uid: str, tid: str, step: int) -> bool:
    return step == TRAJ_N_STEPS.get(tid, 1) - 1


def _placed_cents(uid: str, tid: str, step: int) -> int:
    return 100 if _is_terminal(uid, tid, step) else 15


def _dp6_config() -> SimpleNamespace:
    """Mirror the smoke hydra overrides: world=6, all micro batches=1."""
    return SimpleNamespace(
        trainer=SimpleNamespace(n_gpus_per_node=6, nnodes=1),
        algorithm=SimpleNamespace(use_kl_in_reward=False),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=1,
                                    multi_turn=SimpleNamespace(enable=False)),
            ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=1),
            actor=SimpleNamespace(ppo_micro_batch_size_per_gpu=1, use_kl_loss=False),
        ),
    )


def _make_batch(plan: list[tuple[str, str, int]]) -> DataProto:
    """Schema-complete synthetic training batch, one row per record.

    Mirrors the runtime pre-reward batch: prompts (P_LEN), input_ids
    (P_LEN+R_LEN), attention_mask (all active), responses, uid/traj_uid/
    env_step/search_v1. The per-record v2 placement values are consistent with
    _placed_cents (terminal row: answer_reward_c=100; search row:
    step_shaping_c=15 with evidence credit).
    """
    n = len(plan)
    uids = np.array([p[0] for p in plan], dtype=object)
    trajs = np.array([p[1] for p in plan], dtype=object)
    steps = np.array([p[2] for p in plan], dtype=np.int64)
    search_v1 = np.empty(n, dtype=object)
    for i, (uid, tid, step) in enumerate(plan):
        if _is_terminal(uid, tid, step):
            search_v1[i] = {
                "terminal": True, "answer_reward_c": 100, "format_reward_c": 0,
                "sce_c": 0, "evidence_credit": False, "invalid_or_error": False,
                "redundant_search": False, "answer_leak": False,
            }
        else:
            search_v1[i] = {
                "terminal": False, "step_shaping_c": 15, "evidence_credit": True,
                "invalid_or_error": False, "redundant_search": False,
                "answer_leak": False,
            }
    return DataProto(
        batch=TensorDict(
            source={
                "prompts": torch.randint(0, 100, (n, P_LEN), dtype=torch.long),
                "input_ids": torch.randint(0, 100, (n, P_LEN + R_LEN), dtype=torch.long),
                "attention_mask": torch.ones(n, P_LEN + R_LEN, dtype=torch.long),
                "position_ids": torch.arange(P_LEN + R_LEN).repeat(n, 1),
                "responses": torch.randint(0, 100, (n, R_LEN), dtype=torch.long),
            },
            batch_size=(n,),
        ),
        non_tensor_batch={
            "uid": uids,
            "traj_uid": trajs,
            "env_step": steps,
            "search_v1": search_v1,
        },
        meta_info={"multi_turn": False},
    )


def _split_into_chunks(n_records: int) -> list[tuple[int, int]]:
    """13 chunks; several boundaries fall between two records of the SAME
    trajectory (worst case for cross-chunk duplication)."""
    bounds = [0, 61, 137, 138, 250, 251, 379, 380, 510, 511, 640, 641, 700, 800]
    assert bounds[-1] == n_records
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _identities(batch: DataProto, skip_padding: bool = True) -> set[tuple[str, str, int]]:
    pad = batch.non_tensor_batch.get("is_padding")
    ids = set()
    for i in range(len(batch)):
        if skip_padding and pad is not None and bool(pad[i]):
            continue
        ids.add((str(batch.non_tensor_batch["uid"][i]),
                 str(batch.non_tensor_batch["traj_uid"][i]),
                 int(batch.non_tensor_batch["env_step"][i])))
    return ids


# ---------------------------------------------------------------------------
# D1 multi-chunk concat: no cross-chunk duplicate append
# ---------------------------------------------------------------------------
def test_D1_multi_chunk_concat_no_cross_chunk_duplicates():
    chunks = [_make_batch(PLAN[start:end]) for start, end in _split_into_chunks(800)]
    assembled = DataProto.concat(chunks)
    assert len(assembled) == 800
    # chunk boundaries that split a trajectory (records 60/61, 136/137, ...)
    # must not duplicate that trajectory's steps
    assert len(_identities(assembled)) == 800
    assert _identities(assembled) == set(PLAN)
    # assembly itself introduces no padding flag
    assert "is_padding" not in assembled.non_tensor_batch
    assert_unique_training_records(assembled)  # fail-closed invariant passes


# ---------------------------------------------------------------------------
# D2 adjust_batch pads with synthetic rows, never duplicates real records
# ---------------------------------------------------------------------------
def test_D2_adjust_batch_synthetic_padding_no_duplicates():
    data = _make_batch(PLAN)  # 800 records, 800 % 6 = 2 -> to_add = 4
    out = adjust_batch(_dp6_config(), data, mode="copy")
    assert len(out) == 804
    # exactly 4 synthetic padding rows with unique identities
    pad_rows = [i for i in range(len(out)) if bool(out.non_tensor_batch["is_padding"][i])]
    assert len(pad_rows) == 4
    for i in pad_rows:
        uid = str(out.non_tensor_batch["uid"][i])
        tid = str(out.non_tensor_batch["traj_uid"][i])
        assert uid.startswith("__pad__") and tid.startswith("__pad__")
        assert int(out.non_tensor_batch["env_step"][i]) == -1
        # attention_mask: ONE attended token at prompt position 0 (the actor
        # unpad must not produce an empty sequence); response part all-zero
        # -> response_mask all-zero -> zero loss/reward/adv
        mask = out.batch["attention_mask"][i]
        assert int(mask[0]) == 1
        assert mask[1:].sum().item() == 0
        assert out.non_tensor_batch["search_v1"][i] is None
    assert len(_identities(out, skip_padding=False)) == 800 + 4  # 800 real + 4 unique padding
    assert _identities(out, skip_padding=True) == set(PLAN)  # every real record once
    assert_unique_training_records(out)


# ---------------------------------------------------------------------------
# D3 exact divisibility -> no padding, batch unchanged
# ---------------------------------------------------------------------------
def test_D3_exact_divisibility_no_padding():
    data = _make_batch(PLAN)
    extra = _make_batch([("uid_x", "traj_x0", 0), ("uid_x", "traj_x1", 0),
                         ("uid_x", "traj_x2", 0), ("uid_x", "traj_x3", 0)])
    data2 = DataProto.concat([data, extra])
    assert len(data2) == 804  # 804 % 6 == 0
    out = adjust_batch(_dp6_config(), data2, mode="copy")
    assert out is data2, "exact divisibility must return the batch unchanged"
    assert "is_padding" not in out.non_tensor_batch
    assert_unique_training_records(out)


# ---------------------------------------------------------------------------
# D4 fail-closed invariant raises on a genuine duplicate
# ---------------------------------------------------------------------------
def test_D4_invariant_raises_on_injected_duplicate():
    data = _make_batch(PLAN)
    dup = _make_batch([PLAN[0]])  # same (uid, traj_uid, env_step) as record 0
    data2 = DataProto.concat([data, dup])
    assert len(data2) == 801
    with pytest.raises(RuntimeError, match="duplicate training record identity"):
        assert_unique_training_records(data2)


# ---------------------------------------------------------------------------
# D5 no content-based dedup: same content, different trajectory -> both kept
# ---------------------------------------------------------------------------
def test_D5_no_false_deletion_same_content_different_traj():
    data = _make_batch([("uid_a", "traj_a", 0), ("uid_b", "traj_b", 0)])
    # make the two records byte-identical in CONTENT
    data.batch["input_ids"][1] = data.batch["input_ids"][0]
    data.batch["responses"][1] = data.batch["responses"][0]
    data.batch["attention_mask"][1] = data.batch["attention_mask"][0]
    data.batch["prompts"][1] = data.batch["prompts"][0]
    out = adjust_batch(_dp6_config(), data, mode="copy")
    # identity is (uid, traj_uid, env_step), never bytes: both records kept
    assert _identities(out, skip_padding=False) == {("uid_a", "traj_a", 0), ("uid_b", "traj_b", 0),
                                                    ("__pad__0", "__pad__0", -1), ("__pad__1", "__pad__1", -1),
                                                    ("__pad__2", "__pad__2", -1), ("__pad__3", "__pad__3", -1)}
    assert_unique_training_records(out)
    # "delete" mode removes whole rows (2 % 6 = 2 -> both removed), never merges
    out_del = adjust_batch(_dp6_config(), data, mode="delete")
    assert len(out_del) == 0


# ---------------------------------------------------------------------------
# D6 trajectory returns are not double-counted after padding
# ---------------------------------------------------------------------------
def test_D6_trajectory_return_not_double_counted():
    data = _make_batch(PLAN)
    placed_pre: dict[str, int] = {}
    for (uid, tid, step) in PLAN:
        placed_pre[tid] = placed_pre.get(tid, 0) + _placed_cents(uid, tid, step)
    out = adjust_batch(_dp6_config(), data, mode="copy")
    placed_post: dict[str, int] = {}
    for i in range(len(out)):
        tid = str(out.non_tensor_batch["traj_uid"][i])
        if tid.startswith("__pad__"):
            continue  # synthetic rows contribute nothing to real trajectories
        sv = out.non_tensor_batch["search_v1"][i]
        placed_post[tid] = placed_post.get(tid, 0) + (
            int(sv.get("answer_reward_c", 0)) + int(sv.get("format_reward_c", 0))
            + int(sv.get("sce_c", 0)) if sv.get("terminal") else int(sv.get("step_shaping_c", 0))
        )
    assert set(placed_post) == set(placed_pre)
    for tid in placed_pre:
        assert placed_post[tid] == placed_pre[tid], (
            f"traj {tid}: return double-counted after adjust_batch "
            f"({placed_pre[tid]} -> {placed_post[tid]})"
        )
    # the OLD copy-padding would have inflated 4 trajectories (the 4 copies);
    # the synthetic padding inflates none
    n_inflated = sum(1 for t in placed_pre if placed_post[t] != placed_pre[t])
    assert n_inflated == 0


# ---------------------------------------------------------------------------
# D7 record total == sum of actual steps per trajectory
# ---------------------------------------------------------------------------
def test_D7_record_total_equals_sum_of_steps():
    out = adjust_batch(_dp6_config(), _make_batch(PLAN), mode="copy")
    per_traj: dict[str, int] = {}
    for i in range(len(out)):
        tid = str(out.non_tensor_batch["traj_uid"][i])
        if tid.startswith("__pad__"):
            continue
        per_traj[tid] = per_traj.get(tid, 0) + 1
    expected = {tid: n for n, tid in zip(STEP_COUNTS, (f"traj{i:08x}" for i in range(N_TRAJ)))}
    assert per_traj == expected, "record counts per trajectory changed"
    assert sum(per_traj.values()) == sum(STEP_COUNTS) == 800
    assert len(out) == 800 + 4  # 4 synthetic padding rows only


# ---------------------------------------------------------------------------
# D8 padding rows inert in the v2 reward manager and the audit
# ---------------------------------------------------------------------------
def test_D8_padding_inert_in_reward_manager_and_audit():
    from agent_system.reward_manager.episode import EpisodeRewardManager
    from searchr1_repro.training_audit import build_rollout_audit_records

    out = adjust_batch(_dp6_config(), _make_batch(PLAN), mode="copy")
    n = len(out)
    reward_tensor = torch.zeros(n, R_LEN, dtype=torch.float32)
    mgr = EpisodeRewardManager(tokenizer=None, num_examine=0, search_aware_step_reward=True)
    mgr._apply_search_aware_v2(out, reward_tensor)
    # real rows placed at their last active token; padding rows untouched
    for i in range(n):
        placed_cents_total = int(round(float(reward_tensor[i].sum()) * 100))
        if bool(out.non_tensor_batch["is_padding"][i]):
            assert placed_cents_total == 0, f"padding row {i} got reward"
        else:
            uid = str(out.non_tensor_batch["uid"][i])
            tid = str(out.non_tensor_batch["traj_uid"][i])
            step = int(out.non_tensor_batch["env_step"][i])
            assert placed_cents_total == _placed_cents(uid, tid, step), f"row {i}"
    # episode/group aggregates: version "v2" on every real record; padding rows
    # carry None (never aggregated)
    ep_real = [out.non_tensor_batch["search_v1_episode"][i] for i in range(n)
               if not bool(out.non_tensor_batch["is_padding"][i])]
    grp_real = [out.non_tensor_batch["search_v1_group"][i] for i in range(n)
                if not bool(out.non_tensor_batch["is_padding"][i])]
    assert len(ep_real) == 800 and len(grp_real) == 800
    assert all(e.get("version") == "v2" for e in ep_real)
    assert all(g.get("version") == "v2" for g in grp_real)
    # per-trajectory episode total == component sum (exact cents, fail-closed)
    COMPONENTS = ("answer_reward_c", "format_reward_c", "evidence_hit_reward_c",
                  "searched_correct_bonus_c", "invalid_penalty_c",
                  "redundant_penalty_c", "answer_leak_penalty_c")
    for e in ep_real:
        assert e["total_reward_c"] == sum(e[k] for k in COMPONENTS)
    # audit: padding rows skipped, is_padding=False recorded on real rows
    records = build_rollout_audit_records(out.batch, out.non_tensor_batch, multi_turn=False)
    assert len(records) == 800, "audit must skip the 4 padding rows"
    assert all(r["metadata"]["is_padding"] is False for r in records)
    # audit record indices: every real record of the adjusted batch appears once
    assert len({r["record_index"] for r in records}) == 800


# ---------------------------------------------------------------------------
# D9 padding rows get advantage exactly 0.0 (own size-1 uid group)
# ---------------------------------------------------------------------------
def test_D9_padding_advantage_zero():
    from verl.trainer.ppo.core_algos import compute_grpo_trajectory_return_advantage

    out = adjust_batch(_dp6_config(), _make_batch(PLAN), mode="copy")
    n = len(out)
    token_rewards = torch.zeros(n, R_LEN)
    for i in range(n):
        if bool(out.non_tensor_batch["is_padding"][i]):
            continue
        token_rewards[i, -1] = 15.0 / 100.0  # any non-zero placed value
    response_mask = out.batch["attention_mask"][:, P_LEN:].float()
    assert response_mask[out.non_tensor_batch["is_padding"]].sum().item() == 0, \
        "padding rows must have zero response_mask (zero policy loss)"
    adv, _ = compute_grpo_trajectory_return_advantage(
        token_level_rewards=token_rewards,
        response_mask=response_mask,
        index=out.non_tensor_batch["uid"],
        traj_index=out.non_tensor_batch["traj_uid"],
        epsilon=1e-6,
    )
    for i in range(n):
        if bool(out.non_tensor_batch["is_padding"][i]):
            assert float(adv[i].abs().sum()) == 0.0, f"padding row {i} has non-zero advantage"
    # sanity: real rows of real trajectories actually get non-zero advantages
    real_adv = float(adv[~out.non_tensor_batch["is_padding"]].abs().sum())
    assert real_adv > 0.0


# ---------------------------------------------------------------------------
# D10 forward-path regression: the actor unpad never sees an empty sequence
# (2026-08-23a smoke crashed in actor_rollout_compute_log_prob -> dp_actor
# _forward_micro_batch -> unpad_input -> HF qwen2 view([1, 0, -1, 128]))
# ---------------------------------------------------------------------------
def test_D10_padding_row_unpad_never_empty():
    out = adjust_batch(_dp6_config(), _make_batch(PLAN), mode="copy")
    pad = out.non_tensor_batch["is_padding"]
    for i in range(len(out)):
        attended = out.batch["attention_mask"][i].nonzero(as_tuple=False)
        assert attended.shape[0] >= 1, f"row {i} would produce an empty sequence"
        if bool(pad[i]):
            assert attended.shape[0] == 1, f"padding row {i} must have exactly 1 token"
            assert int(attended[0, 0]) == 0, "padding row's only token must be position 0"
    # a pure-padding micro-batch (micro_bsz=1, as in the run config) therefore
    # always forwards exactly 1 token -> no 0-element reshape
    n_pad = int(pad.sum())
    assert n_pad == 4
    micro = out.batch["attention_mask"][pad].nonzero(as_tuple=False)
    assert micro.shape[0] == 4, "4 padding rows x 1 attended token = 4 unpad indices"


# ---------------------------------------------------------------------------
# D11 update_policy guard condition: padding rows have zero response_mask, so
# their micro-batch takes the collective-uniform zero-loss path (token-mean
# 0/0 would be NaN; D13 covers the update_policy behavior itself)
# ---------------------------------------------------------------------------
def test_D11_padding_response_mask_zero_guard_condition():
    out = adjust_batch(_dp6_config(), _make_batch(PLAN), mode="copy")
    pad = out.non_tensor_batch["is_padding"]
    response_mask = out.batch["attention_mask"][:, P_LEN:].float()
    assert response_mask[pad].sum().item() == 0, \
        "padding rows' response_mask sums to 0 -> zero-loss path in update_policy"
    assert response_mask[~pad].sum().item() > 0, "real rows still train"
    # single-turn path: update_policy response_mask = attention[:, -response_length:]
    assert response_mask.shape == (len(out), R_LEN)


# ---------------------------------------------------------------------------
# D12 multi-turn batches: padding rows' copied loss_mask is zeroed
# ---------------------------------------------------------------------------
def test_D12_padding_loss_mask_zeroed():
    data = _make_batch(PLAN)
    data.batch["loss_mask"] = torch.ones(len(data), P_LEN + R_LEN, dtype=torch.long)
    out = adjust_batch(_dp6_config(), data, mode="copy")
    pad = out.non_tensor_batch["is_padding"]
    assert out.batch["loss_mask"][pad].sum().item() == 0, \
        "padding rows must not carry row-0's loss_mask (multi-turn would train on them)"
    assert out.batch["loss_mask"][~pad].sum().item() == len(data) * (P_LEN + R_LEN)
    # multi-turn update path: response_mask = loss_mask[:, -R_LEN:] -> still 0
    assert out.batch["loss_mask"][pad][:, -R_LEN:].sum().item() == 0


# ---------------------------------------------------------------------------
# D13 update_policy padding micro-batch: COLLECTIVE-UNIFORM zero-loss
# forward+backward (deadlock root cause, 2026-08-23).
#
# The first guard `continue`-skipped zero-response-mask micro-batches PER-RANK.
# _balance_batch dispatches the batch to DP ranks via
# get_seqlen_balanced_partitions (Karmarkar-Karp), so the 1-token padding rows
# land in DIFFERENT ranks' partitions and ranks execute DIFFERENT numbers of
# FSDP forwards/backwards. Their NCCL collective sequences diverge and hang at
# the mini-batch boundary: ranks done with their mini enter _optimizer_step's
# grad-norm ALLREDUCE while ranks still in the loop issue the next forward's
# embedding ALLGATHER -- the deterministic 20260823b/c watchdog SIGABRT (4
# ranks at ALLREDUCE vs 2 ranks at _ALLGATHER_BASE, same seq 26282).
#
# The fix runs the SAME forward+backward on every rank for a padding
# micro-batch, with a loss connected to the model (FSDP post-backward hooks
# fire uniformly) but exactly zero. This test runs the real update_policy loop
# on CPU with a mocked forward and asserts: every micro-batch gets exactly one
# forward (no skip), every padding micro-batch's backward runs with an
# exactly-zero-valued connected loss, and the optimizer step still runs.
# ---------------------------------------------------------------------------
def test_D13_padding_micro_batch_zero_loss_backward_collective_uniform():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    # small batch: 4 real records (2 trajectories x 2 steps) -> 4 % 6 = 4
    # -> adjust_batch appends 2 padding rows -> 6 rows, 6 micro-batches
    small_plan = [
        ("uid00000001", "traj00000001", 0),
        ("uid00000001", "traj00000001", 1),
        ("uid00000002", "traj00000002", 0),
        ("uid00000002", "traj00000002", 1),
    ]
    data = _make_batch(small_plan)
    data.batch["old_log_probs"] = torch.zeros(len(data), R_LEN, dtype=torch.float32)
    data.batch["advantages"] = torch.zeros(len(data), R_LEN, dtype=torch.float32)
    data.meta_info["temperature"] = 1.0
    out = adjust_batch(_dp6_config(), data, mode="copy")
    assert out.non_tensor_batch["is_padding"].sum() == 2

    actor = object.__new__(DataParallelPPOActor)  # skip heavy __init__ on CPU
    actor.actor_module = torch.nn.Linear(2, 2)
    actor.actor_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    actor.config = SimpleNamespace(
        ppo_epochs=1,
        ppo_mini_batch_size=330,
        ppo_micro_batch_size_per_gpu=1,
        use_dynamic_bsz=False,
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        clip_ratio_c=3.0,
        entropy_coeff=0,
        loss_agg_mode="token-mean",
        policy_loss={"loss_mode": "vanilla"},
        use_kl_loss=False,
        kl_loss_type="low_var_kl",
        kl_loss_coef=0.001,
    )
    actor.config.get = lambda key, default=None: getattr(actor.config, key, default)

    # mock the FSDP forward: count calls, return a model-CONNECTED log_prob
    # (requires_grad) so the zero-loss backward's graph + hooks are real
    forward_calls = []
    captured_padding_grads = []

    def fake_forward(micro_batch, temperature, calculate_entropy=False):
        n = micro_batch["responses"].shape[0]
        log_prob = torch.zeros(n, R_LEN, dtype=torch.float32, requires_grad=True)
        forward_calls.append(micro_batch)
        return None, log_prob

    actor._forward_micro_batch = fake_forward
    actor._optimizer_step = lambda: torch.zeros(())

    # the @GPUMemoryLogger decorator reads GPU memory; CPU has no such API
    import verl.utils.debug.performance as perf_mod

    perf_mod._get_current_mem_info = lambda **kw: ("0", "0", "0", "0")

    # grab the log_prob tensors the guard path sees, to inspect .grad after
    # update_policy (populated => backward ran; all-zero => loss was exactly 0).
    # The tensor keeps its autograd graph (no detach): after backward, .grad is
    # populated on the tensor itself even though the graph was freed.
    orig_masked = torch.Tensor.masked_fill
    captured = []

    def spy_masked_fill(self, mask, value):
        captured.append(self)
        return orig_masked(self, mask, value)

    torch.Tensor.masked_fill = spy_masked_fill

    try:
        metrics = actor.update_policy(out)
    finally:
        torch.Tensor.masked_fill = orig_masked

    # 1) EVERY micro-batch got exactly one forward -- NO per-rank skip
    assert len(forward_calls) == len(out) == 6, \
        f"every micro-batch must run a forward (padding included); got {len(forward_calls)}/6"
    # 2) the zero-loss path ran a connected backward with EXACTLY-zero loss:
    #    .grad populated and all-zero on every padding micro-batch's log_prob
    pad = out.non_tensor_batch["is_padding"]
    pad_indexes = [i for i in range(len(out)) if bool(pad[i])]
    assert len(captured) == len(pad_indexes), \
        f"zero-loss path must run once per padding micro-batch; got {len(captured)} vs {len(pad_indexes)}"
    for log_prob in captured:
        assert log_prob.grad is not None, "backward must run on the padding micro-batch"
        assert float(log_prob.grad.abs().sum()) == 0.0, \
            "masked_fill(all-True) makes the loss exactly zero -> zero gradients"
    # 3) the real micro-batches went through the normal policy-loss path and
    #    produced finite metrics, and the optimizer step still ran
    #    (append_to_dict accumulates per-micro-batch values into lists)
    assert "actor/pg_loss" in metrics and "actor/grad_norm" in metrics
    assert float(metrics["actor/grad_norm"][0]) == 0.0
    flat = [x for v in metrics.values() if isinstance(v, list) for x in v]
    assert all(np.isfinite(x) for x in flat)
    # 4) sanity: without padding rows the same run still works (no regression)
    data2 = _make_batch(small_plan)
    data2.batch["old_log_probs"] = torch.zeros(len(data2), R_LEN, dtype=torch.float32)
    data2.batch["advantages"] = torch.zeros(len(data2), R_LEN, dtype=torch.float32)
    data2.meta_info["temperature"] = 1.0
    out2 = adjust_batch(_dp6_config(), data2, mode="copy")  # 4 % 6 = 4 -> 2 padding
    forward_calls.clear()
    captured.clear()
    try:
        metrics2 = actor.update_policy(out2)
    finally:
        torch.Tensor.masked_fill = orig_masked
    assert len(forward_calls) == len(out2) == 6
    assert "actor/pg_loss" in metrics2
