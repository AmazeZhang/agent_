"""CPU-only tests for patch 0006 LR-schedule segment continuity.

Run with:  CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_scheduler_continuity.py
Requires PYTHONPATH="$PWD/vendor/verl-agent:$PWD".

What is verified (formal profile semantics, fork upstream pin 20bd331b):

  - The formal schedule horizon is fixed at 300 steps and warmup at 85 steps
    (== int(0.285 * 300)). Segment stops (50/100/300) are decoupled from the
    horizon via trainer.segment_stop_step (patch 0006): the trainer stops after
    the segment's last global step, saves a checkpoint (DataLoader/Optimizer/
    Scheduler/RNG + model), and returns normally. The scheduler is rebuilt on
    resume with the SAME 300/85 parameters, so the LR trajectory is pointwise
    identical to an uninterrupted 300-step run.
  - The scheduler is stepped exactly once per global step, after update_policy
    (fsdp_workers.py update_actor: lr = get_last_lr()[0]; ...; step()). LR at
    global step N is lr_lambda(N-1) (empirically verified in this env:
    construction leaves last_epoch=0, _step_count=1, _last_lr=[0.0]; after one
    global step the checkpoint shows last_epoch=1, _step_count=2).
  - A wrong horizon (e.g. total_training_steps=50 -> warmup=int(0.285*50)=14)
    is DETECTABLE at the step-50 checkpoint: the saved scheduler _last_lr
    deviates from the expected min(1, last_epoch/85)*base_lr, even though the
    post-resume LR trajectory looks identical. The checkpoint-level check is
    the required detection point (the wrapper also pins resume sources to the
    exact global_step_50/100 produced by the previous segment under the frozen
    config, so the wrong-horizon config can never reach resume).
"""

from __future__ import annotations

import math

import pytest
import torch
from transformers import get_constant_schedule_with_warmup

HORIZON = 300
WARMUP = 85          # == int(0.285 * 300)
BASE_LR = 1e-6
SEGMENTS = (50, 100, 300)

# mocks the missing verl dependency in CPU tests


def expected_lr(step: int) -> float:
    """LR used at global step `step` (1-indexed) under the 300/85 schedule.

    Mirrors verl's update_actor: lr = get_last_lr()[0] with last_epoch==step-1,
    i.e. lr_lambda(step-1) from get_constant_schedule_with_warmup.
    """
    current = step - 1
    if current < WARMUP:
        return BASE_LR * current / float(WARMUP)
    return BASE_LR


def make_scheduler(num_warmup_steps: int, lr: float = BASE_LR):
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps), optimizer


def run_global_step(scheduler):
    """One verl global step: read LR, then step the scheduler exactly once."""
    lr = scheduler.get_last_lr()[0]
    scheduler.step()
    return lr


def run_segment(scheduler, start: int, end: int):
    """Steps start..end inclusive; returns {step: lr}."""
    history = {}
    for step in range(start, end + 1):
        history[step] = run_global_step(scheduler)
    return history


def checkpoint_mismatch(state: dict) -> bool:
    """Detection routine: does the checkpointed scheduler state deviate from the
    expected 300/85 schedule? Compares the saved _last_lr against
    expected min(1, last_epoch/85) * base_lr (the value the SAME config would
    produce at that position)."""
    last_epoch = state["last_epoch"]
    saved_lr = state["_last_lr"][0]
    expected = BASE_LR * min(1.0, last_epoch / float(WARMUP))
    return not math.isclose(saved_lr, expected, rel_tol=0.0, abs_tol=1e-12)


def test_warmup_formula_matches_documented_value():
    assert WARMUP == int(0.285 * HORIZON)


def test_uninterrupted_baseline_matches_closed_form():
    scheduler, _ = make_scheduler(WARMUP)
    history = run_segment(scheduler, 1, HORIZON)
    assert history[1] == pytest.approx(expected_lr(1), abs=1e-12)
    assert history[85] == pytest.approx(expected_lr(85), abs=1e-12)
    assert history[86] == pytest.approx(BASE_LR, abs=1e-12)
    assert history[HORIZON] == pytest.approx(BASE_LR, abs=1e-12)
    # one step per global step: state matches the resume observation pattern
    assert scheduler.state_dict()["last_epoch"] == HORIZON
    assert scheduler.state_dict()["_step_count"] == HORIZON + 1


@pytest.mark.parametrize("stop", SEGMENTS[:-1])
def test_resume_from_segment_stop_is_pointwise_identical(stop):
    """Save at `stop`, rebuild with the SAME 300/85 params, continue: every LR
    from stop+1..300 equals the uninterrupted baseline bit-for-bit."""
    baseline_scheduler, _ = make_scheduler(WARMUP)
    baseline = run_segment(baseline_scheduler, 1, HORIZON)

    segment_scheduler, _ = make_scheduler(WARMUP)
    first_half = run_segment(segment_scheduler, 1, stop)
    assert first_half == {s: baseline[s] for s in range(1, stop + 1)}

    # checkpoint: scheduler state is serialized into extra_state on save
    saved_state = segment_scheduler.state_dict()
    assert saved_state["last_epoch"] == stop
    assert saved_state["_step_count"] == stop + 1
    assert not checkpoint_mismatch(saved_state)

    # resume: fresh scheduler with identical params + load_state_dict
    resumed_scheduler, _ = make_scheduler(WARMUP)
    resumed_scheduler.load_state_dict(saved_state)

    second_half = run_segment(resumed_scheduler, stop + 1, HORIZON)
    for step, lr in second_half.items():
        assert lr == baseline[step], f"LR diverged at step {step}: {lr} vs {baseline[step]}"
    # last_epoch / _step_count continuity (scheduler stepped once per global step)
    final_state = resumed_scheduler.state_dict()
    assert final_state["last_epoch"] == HORIZON
    assert final_state["_step_count"] == HORIZON + 1


def test_wrong_50step_horizon_is_detectable_at_checkpoint():
    """total_training_steps=50 would build warmup=int(0.285*50)=14; the step-50
    checkpoint of such a run has _last_lr=BASE_LR (warmup already finished) while
    the frozen 300/85 config expects 50/85*BASE_LR -- the checkpoint-level check
    flags it. (The naive 'compare post-resume LR' check would NOT: steps 51..300
    are identical because the saved last_epoch carries the position.)"""
    wrong_scheduler, _ = make_scheduler(int(0.285 * 50))  # warmup=14
    run_segment(wrong_scheduler, 1, 50)
    wrong_state = wrong_scheduler.state_dict()
    assert checkpoint_mismatch(wrong_state), "wrong horizon must be detectable"
    # for contrast: same position under the frozen config is NOT a mismatch
    good_scheduler, _ = make_scheduler(WARMUP)
    run_segment(good_scheduler, 1, 50)
    assert not checkpoint_mismatch(good_scheduler.state_dict())


def test_wrong_100step_horizon_is_detectable_at_checkpoint():
    """The step-50 checkpoint also catches a 100-step horizon (warmup=28)."""
    wrong_scheduler, _ = make_scheduler(int(0.285 * 100))  # warmup=28
    run_segment(wrong_scheduler, 1, 50)
    assert checkpoint_mismatch(wrong_scheduler.state_dict())
