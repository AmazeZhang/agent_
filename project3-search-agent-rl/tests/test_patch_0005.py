"""CPU-only tests for patch 0005 (env.projection=official loose passthrough).

Run with:  CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_patch_0005.py
Requires PYTHONPATH="$PWD/vendor/verl-agent:$PWD".

What patch 0005 changes:
  - agent_system/environments/env_package/search/projection.py: new top-level
    passthrough_projection() that returns raw text actions untouched with
    valids all True (official-loose semantics, matching
    scripts/run_p3_eval_vllm_official.py).
  - env_manager.py make_envs: env.projection=official -> projection_f is the
    passthrough_projection function; env.projection=strict (or key absent) ->
    original search_projection (validity gating). Any other value raises
    ValueError (fail-fast, so a typo can never silently fall back to strict).
    The underlying skyrl SearchEnv receives the raw action; is_action_valid is
    always True; the invalid-action penalty is disabled separately by config
    (actor_rollout_ref.actor.use_invalid_action_penalty=false).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "verl-agent"))

from agent_system.environments.env_manager import make_envs
from agent_system.environments.env_package.search.projection import (
    passthrough_projection,
    search_projection,
)


def _config(projection=None, history_length=0):
    env = {
        "env_name": "search",
        "seed": 0,
        "max_steps": 2,
        "history_length": history_length,
        "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
        "rollout": {"n": 5},
    }
    if projection is not None:
        env["projection"] = projection
    return OmegaConf.create(
        {"env": env, "data": {"train_batch_size": 66, "val_batch_size": 16}}
    )


class _StubPool:
    """Stands in for SearchMultiProcessEnv: records the actions it receives."""

    def __init__(self):
        self.received_actions = None
        self.reset_kwargs = None

    def reset(self, kwargs):
        self.reset_kwargs = kwargs
        obs = ["what channel is celebrity big brother on in the usa?"]
        return obs, [{"question": obs[0]}]

    def step(self, actions):
        self.received_actions = list(actions)
        n = len(actions)
        return ["observation"] * n, [0.0] * n, [True] * n, [{"won": False}] * n


@pytest.fixture
def stub_pool(monkeypatch):
    pool = _StubPool()
    monkeypatch.setattr(
        "agent_system.environments.env_package.search.build_search_envs",
        lambda **kwargs: pool,
    )
    return pool


# --------------------------------------------------------------------------
# projection selection
# --------------------------------------------------------------------------

def test_official_projection_is_top_level_passthrough_function(stub_pool):
    # projection_f must be the real top-level function (not an inline lambda),
    # so the semantics are a named, testable, importable contract.
    envs, val_envs = make_envs(_config(projection="official"))
    assert envs.projection_f is passthrough_projection
    assert val_envs.projection_f is passthrough_projection


def test_official_projection_is_raw_passthrough_all_valid(stub_pool):
    envs, _val_envs = make_envs(_config(projection="official"))

    raw = [
        "<search>who wrote imagine</search> trailing junk",
        "mixed <search>x</search> and <answer>y</answer>",
        "plain text with no tags",
    ]
    actions, valids = envs.projection_f(raw)

    assert actions == raw  # byte-identical passthrough, no trimming/rewrite
    assert all(v is True for v in valids)  # official-loose: no invalid concept
    assert len(actions) == len(valids) == 3


def test_strict_default_projection_is_search_projection(stub_pool):
    # key absent -> strict default (fork behavior unchanged)
    envs, _val_envs = make_envs(_config())
    assert envs.projection_f.func is search_projection

    # explicit strict -> same
    envs2, _val_envs2 = make_envs(_config(projection="strict"))
    assert envs2.projection_f.func is search_projection


def test_unknown_projection_value_fails_fast(stub_pool):
    # A typo must never silently fall back to strict semantics: any value
    # other than strict/official aborts at env construction time.
    with pytest.raises(ValueError, match="projection"):
        make_envs(_config(projection="offical"))
    with pytest.raises(ValueError, match="projection"):
        make_envs(_config(projection="LOOSE"))
    with pytest.raises(ValueError, match="projection"):
        make_envs(_config(projection=""))


def test_strict_projection_still_gates_validity(stub_pool):
    envs, _val_envs = make_envs(_config(projection="strict"))
    # mixed search+answer is invalid under search_projection rules
    actions, valids = envs.projection_f(["<search>x</search><answer>y</answer>"])
    assert valids[0] == 0
    # well-formed single search is valid and trimmed
    actions, valids = envs.projection_f(["junk <search>  who won  </search> junk"])
    assert valids[0] == 1
    assert actions[0] == "<search>who won</search>"


# --------------------------------------------------------------------------
# manager.step end-to-end with official projection
# --------------------------------------------------------------------------

def test_official_manager_step_passes_raw_actions_and_valid(stub_pool):
    envs, _val_envs = make_envs(_config(projection="official"))

    observations, _infos = envs.reset({"question": "q"})
    assert len(observations["text"]) == 1

    raw = ["<search>who wrote imagine</search> trailing junk"]
    next_obs, rewards, dones, infos = envs.step(raw)

    # raw action reached the underlying env untouched (no projection rewrite)
    assert stub_pool.received_actions == raw
    # valids all True -> is_action_valid all True (no penalty downstream)
    assert all(bool(i["is_action_valid"]) for i in infos)
    assert len(next_obs["text"]) == 1


def test_official_val_envs_also_loose(stub_pool):
    envs, val_envs = make_envs(_config(projection="official"))
    raw = ["<search>a</search>"]
    actions, valids = val_envs.projection_f(raw)
    assert actions == raw and all(valids)
