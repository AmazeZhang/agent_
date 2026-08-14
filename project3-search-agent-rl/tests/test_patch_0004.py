"""CPU-only tests for patch 0004 (search prompt instruction + format reward).

Run with:  CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_patch_0004.py
Requires PYTHONPATH="$PWD/vendor/verl-agent:$PWD".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "verl-agent"))

from agent_system.environments.env_package.search.envs import (
    SEARCH_PROMPT_PREFIX,
    SearchMultiProcessEnv,
)
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import (
    SearchEnv,
)
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (
    compute_score,
)


# --------------------------------------------------------------------------
# prompt instruction
# --------------------------------------------------------------------------

def test_prompt_prefix_contains_protocol_and_fewshot():
    assert "<search>query</search>" in SEARCH_PROMPT_PREFIX
    assert "<answer>...</answer>" in SEARCH_PROMPT_PREFIX
    # few-shot example present (question -> search -> answer flow)
    assert "Example:" in SEARCH_PROMPT_PREFIX
    assert "<search>who wrote the song Imagine</search>" in SEARCH_PROMPT_PREFIX
    assert "<answer>John Lennon</answer>" in SEARCH_PROMPT_PREFIX
    # ends with instruction to answer
    assert SEARCH_PROMPT_PREFIX.rstrip().endswith("Now answer the following question:")


class _StubEnv:
    def __init__(self):
        self.reset_called = False

    def reset(self, extras):
        self.reset_called = True
        self.extras = extras


def test_sync_reset_prepends_prompt_prefix(monkeypatch):
    manager = SearchMultiProcessEnv.__new__(SearchMultiProcessEnv)
    manager.max_steps = 2
    stub = _StubEnv()
    question = "what channel is celebrity big brother on in the usa?"
    obs, info = manager._sync_reset(stub, {
        "ground_truth": ["CBS"],
        "data_source": "nq",
        "question": question,
    })
    assert stub.reset_called and stub.extras["ground_truth"] == ["CBS"]
    assert obs == SEARCH_PROMPT_PREFIX + question
    assert obs.endswith(question)
    assert info["data_source"] == "nq"


# --------------------------------------------------------------------------
# format reward (format_score=0.1)
# --------------------------------------------------------------------------

def test_compute_score_with_format_score_0_1():
    # no <answer> at all -> 0 regardless of format_score
    assert compute_score("<think>hmm</think>", {"target": ["x"]}, format_score=0.1) == 0.0
    # well-formed but wrong answer -> format score 0.1
    assert compute_score("<answer>wrong</answer>", {"target": ["x"]}, format_score=0.1) == pytest.approx(0.1)
    # correct answer -> full score 1.0
    assert compute_score("<answer>x</answer>", {"target": ["x"]}, format_score=0.1) == 1.0
    # default format_score stays 0.0 (old behavior for non-0004 callers)
    assert compute_score("<answer>wrong</answer>", {"target": ["x"]}) == 0.0


def test_env_get_reward_uses_format_score_0_1():
    """SearchEnv._get_reward must pass format_score=0.1 to compute_score."""
    env = SearchEnv.__new__(SearchEnv)
    env.chat_history = [{"role": "assistant", "content": "<think>r</think><answer>wrong</answer>"}]
    env.ground_truth = {"target": ["x"]}
    reward = env._get_reward("<answer>wrong</answer>", done=True)
    assert reward == pytest.approx(0.1)
    env2 = SearchEnv.__new__(SearchEnv)
    env2.chat_history = [{"role": "assistant", "content": "<think>r</think><answer>x</answer>"}]
    env2.ground_truth = {"target": ["x"]}
    assert env2._get_reward("<answer>x</answer>", done=True) == pytest.approx(1.0)
    env3 = SearchEnv.__new__(SearchEnv)
    env3.chat_history = [{"role": "assistant", "content": "<think>r</think>"}]
    env3.ground_truth = {"target": ["x"]}
    assert env3._get_reward("<think>r</think>", done=True) == 0.0


def test_intermediate_steps_still_unrewarded():
    env = SearchEnv.__new__(SearchEnv)
    env.chat_history = [{"role": "assistant", "content": "<search>q</search>"}]
    env.ground_truth = {"target": ["x"]}
    assert env._get_reward("<search>q</search>", done=False) == 0.0
