"""CPU-only tests for the CLEAN-UPSTREAM evaluation line
(scripts/run_p3_eval_upstream_clean.py + wrapper).

These tests import agent_system from the PRISTINE upstream worktree
(vendor/upstream-20bd331b, NO patches 0001-0008) - never from the patched
vendor/verl-agent. Run with:

    CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_eval_upstream_clean.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_TREE = PROJECT_ROOT / "vendor" / "upstream-20bd331b"
PATCHED_VENDOR = PROJECT_ROOT / "vendor" / "verl-agent"

assert CLEAN_TREE.is_dir(), f"clean tree missing: {CLEAN_TREE}"

# The clean tree must shadow the patched vendor for every import in this file.
if CLEAN_TREE not in sys.path:
    sys.path.insert(0, str(CLEAN_TREE))
for name in list(sys.modules):
    if name == "agent_system" or name.startswith("agent_system."):
        del sys.modules[name]

import agent_system  # noqa: E402
import agent_system.environments.env_package.search.projection as projection  # noqa: E402

from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402

EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "run_p3_eval_upstream_clean.py"
spec = importlib.util.spec_from_file_location("run_p3_eval_upstream_clean", EVAL_SCRIPT)
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)  # noqa: F811 (module under test)


def test_imports_come_from_clean_tree():
    paths = [str(Path(p).resolve()) for p in getattr(agent_system, "__path__", [])]
    assert any(p.startswith(str(CLEAN_TREE.resolve())) for p in paths), (
        f"agent_system resolved outside the clean tree: {paths}"
    )


class TestProjection:
    def test_valid_search(self):
        actions, valids = projection.search_projection(["prefix <search> what is paris </search> suffix"])
        assert actions == ["<search>what is paris</search>"]
        assert valids == [1]

    def test_valid_answer(self):
        actions, valids = projection.search_projection(["think <answer> Paris </answer> done"])
        assert actions == ["<answer>Paris</answer>"]
        assert valids == [1]

    def test_no_tags_invalid(self):
        actions, valids = projection.search_projection(["just some text"])
        assert actions == [""]
        assert valids == [0]

    def test_both_tags_invalid_but_extracts_search(self):
        actions, valids = projection.search_projection(["<search>q1</search> <answer>a1</answer>"])
        assert actions == ["<search>q1</search>"]
        assert valids == [0]

    def test_duplicate_tags_invalid(self):
        actions, valids = projection.search_projection(["<search>a</search> then <search>b</search>"])
        assert actions == ["<search>a</search>"]
        assert valids == [0]

    def test_truncation_at_first_close_tag(self):
        actions, _ = projection.search_projection(["<search>a</search> <search>b</search>"])
        assert actions == ["<search>a</search>"]


class FakeSearchEnvs:
    """Minimal SearchMultiProcessEnv-protocol stub: no network, no threads.

    step() asserts it receives PROJECTED actions (manager contract) and returns
    upstream-shaped outputs: search -> information observation; answer -> done
    with reward 1.0; anything else -> empty observation, not done.
    """

    def __init__(self, answers: list[str]):
        self.answers = answers
        self.stepped_actions: list[list[str]] = []

    def reset(self, kwargs):
        self.tasks = [kw["question"] for kw in kwargs]
        return self.tasks, [{"data_source": kw.get("data_source", "unknown")} for kw in kwargs]

    def step(self, actions):
        self.stepped_actions.append(list(actions))
        obs, rewards, dones, infos = [], [], [], []
        for i, action in enumerate(actions):
            if action.startswith("<search>"):
                obs.append("\n<information>{'result': [{'content': 'answer-ish'}]}</information>\n")
                rewards.append(0.0)
                dones.append(False)
                infos.append({"tool_calling": True, "tool_input": action, "data_source": "test"})
            elif action.startswith("<answer>"):
                obs.append("")
                rewards.append(1.0 if "<answer>Paris</answer>" == action else 0.0)
                dones.append(True)
                infos.append({"tool_calling": False, "data_source": "test"})
            else:
                obs.append("")
                rewards.append(0.0)
                dones.append(False)
                infos.append({"tool_calling": True, "tool_input": None, "data_source": "test"})
        return obs, rewards, dones, infos


def make_config(history_length: int = 4):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "env": {
                "max_steps": 4,
                "history_length": history_length,
                "search": {"search_url": "http://127.0.0.1:18080/retrieve", "topk": 3, "timeout": 180, "log_requests": True},
            }
        }
    )


class TestSearchEnvironmentManager:
    def test_round2_prompt_contains_question_query_information(self):
        envs = FakeSearchEnvs(answers=["Paris"])
        config = make_config()
        manager = SearchEnvironmentManager(envs, projection.search_projection, config)
        kwargs = [
            {"question": "What is the capital of France?", "ground_truth": {"target": ["Paris"]}, "data_source": "test"}
        ]
        observations, infos = manager.reset(kwargs)
        # Round 1: official single-layer NO-HIS template; task question present.
        # The NO-HIS template never mentions <information>; it does mention the
        # <search>/<answer> format in its instructions, so no tag-absence claim.
        round1 = observations["text"][0]
        assert "What is the capital of France?" in round1
        assert "<information>" not in round1

        next_observations, rewards, dones, infos = manager.step(["<search>capital of France</search>"])
        round2 = next_observations["text"][0]
        assert "What is the capital of France?" in round2  # original question
        assert "<search>capital of France</search>" in round2  # projected query in history
        assert "<information>" in round2  # returned information in history
        assert "Step 1:" in round2
        # The env must have received the PROJECTED action, not the raw one.
        assert envs.stepped_actions[-1] == ["<search>capital of France</search>"]
        assert "<information>" in next_observations["anchor"][0]

    def test_step_rewards_and_dones_shaped_upstream(self):
        envs = FakeSearchEnvs(answers=["Paris"])
        config = make_config()
        manager = SearchEnvironmentManager(envs, projection.search_projection, config)
        kwargs = [{"question": "Q?", "ground_truth": {"target": ["Paris"]}, "data_source": "test"}]
        manager.reset(kwargs)
        _, rewards, dones, infos = manager.step(["<answer>Paris</answer>"])
        assert float(rewards[0]) == 1.0
        assert bool(dones[0]) is True  # upstream to_numpy -> numpy bool
        assert infos[0]["tool_calling"] is False
        assert "is_action_valid" in infos[0]


def make_episode(question="What is the capital of France?", searched=True, correct=True, answer_present=True):
    steps = []
    if searched:
        steps.append(
            {
                "step": 1,
                "prompt": "round1 prompt",
                "raw_action": "I search: <search>capital of France</search>",
                "projected_action": "<search>capital of France</search>",
                "action_valid": 1,
                "observation": "\n<information>{'result': [{'content': 'Paris is the capital'}]}</information>\n",
                "prompt_next_round": "round2 prompt with <search>capital of France</search> and <information> and " + question,
                "reward": 0.0,
                "done": False,
                "won": False,
                "tool_calling": True,
                "tool_input": "capital of France",
                "information_returned": True,
                "info": {"tool_calling": True, "tool_input": "capital of France"},
                "batch_generation_seconds": 1.0,
            }
        )
    final_reward = 1.0 if correct else 0.0
    final_action = "<answer>Paris</answer>" if answer_present else "no answer here"
    steps.append(
        {
            "step": 2,
            "prompt": steps[-1]["prompt_next_round"] if steps else "round1 prompt",
            "raw_action": final_action,
            "projected_action": final_action,
            "action_valid": 1,
            "observation": "",
            "prompt_next_round": "",
            "reward": final_reward,
            "done": True,
            "won": bool(correct),
            "tool_calling": False,
            "tool_input": None,
            "information_returned": False,
            "info": {"tool_calling": False},
            "batch_generation_seconds": 1.0,
        }
    )
    return {
        "question": question,
        "answers": ["Paris"],
        "source": "test",
        "steps": steps,
        "reward": final_reward,
        "done": True,
        "won": bool(correct),
        "offline": {"final_answer": "Paris" if answer_present else None, "score": final_reward, "has_answer": answer_present},
    }


def re_search_info(prompt: str) -> bool:
    import re

    return (
        re.search(r"<information>(?!\s*</information>)", prompt, re.IGNORECASE | re.DOTALL)
        is not None
    )


class TestRound2PromptCheck:
    def test_passes_when_checked_episode_complete(self):
        episodes = [make_episode(searched=True, correct=True)]
        result = eval_mod.check_round2_prompts(episodes)
        assert result["checked_episodes"] == 1
        assert result["passed_episodes"] == 1
        assert result["passed"] is True

    def test_fails_when_query_missing_from_round2(self):
        episode = make_episode(searched=True, correct=True)
        episode["steps"][1]["prompt"] = episode["steps"][1]["prompt"].replace("<search>capital of France</search>", "")
        result = eval_mod.check_round2_prompts([episode])
        assert result["passed"] is False
        assert result["failures"][0]["contains_query"] is False

    def test_fails_closed_when_nothing_checked(self):
        episodes = [make_episode(searched=False, correct=False)]
        result = eval_mod.check_round2_prompts(episodes)
        assert result["checked_episodes"] == 0
        assert result["passed"] is False

    def test_template_placeholder_does_not_count_as_information(self):
        # SEARCH_TEMPLATE's instructions contain "<information> </information>";
        # only a real returned block (non-whitespace content) passes.
        step = {"projected_action": "<search>q</search>", "tool_calling": True}
        prompt = (
            "Your question: What is the capital of France?\n"
            "<information> </information> wrapped the corresponding search results.\n"
            "History:\nStep 1:<search>q</search>\n"
        )
        assert re_search_info(prompt) is False
        prompt_real = prompt.replace("<information> </information>", "<information>{'result': [1]}</information>")
        assert re_search_info(prompt_real) is True

    def test_done_rounds_are_not_search_attempts(self):
        # An episode that ANSWERED at step 1 has tool_calling=False; its later
        # rounds (which the upstream loop still steps) must not count as searches.
        episode = make_episode(searched=False, correct=True)
        episode["steps"] = [
            {
                "step": 1,
                "prompt": "p1",
                "raw_action": "<answer>Paris</answer>",
                "projected_action": "<answer>Paris</answer>",
                "action_valid": 1,
                "observation": "",
                "prompt_next_round": "p2",
                "reward": 1.0,
                "done": True,
                "won": True,
                "tool_calling": False,
                "tool_input": None,
                "information_returned": False,
                "info": {},
                "batch_generation_seconds": 1.0,
            },
            {
                "step": 2,
                "prompt": "p2",
                "raw_action": "<search>capital of France</search>",
                "projected_action": "<search>capital of France</search>",
                "action_valid": 1,
                "observation": "",
                "prompt_next_round": "",
                "reward": 1.0,
                "done": True,
                "won": True,
                "tool_calling": False,  # done round: env does not execute tools
                "tool_input": None,
                "information_returned": False,
                "info": {"tool_calling": False},
                "batch_generation_seconds": 1.0,
            },
        ]
        assert eval_mod.step_search_query(episode["steps"][1]) is None
        assert eval_mod.step_search_query(episode["steps"][0]) is None

    def test_empty_query_is_not_a_search(self):
        step = {"projected_action": "<search></search>", "tool_calling": True}
        assert eval_mod.step_search_query(step) is None


class TestAggregateMetrics:
    def test_counts(self):
        episodes = [
            make_episode(searched=True, correct=True),  # search -> correct, answered
            make_episode(searched=True, correct=False, answer_present=False),  # search -> no answer
            make_episode(searched=False, correct=True),  # no search -> correct (impossible but counts)
        ]
        metrics = eval_mod.aggregate_metrics(episodes)
        assert metrics["overall"]["n"] == 3
        assert metrics["overall"]["em"] == 2
        assert metrics["search"]["searched_episodes"] == 2
        assert metrics["search"]["no_search_episodes"] == 1
        assert metrics["search"]["search_successful_steps"] == 2
        assert metrics["search"]["search_to_answer"] == 0.5
        assert metrics["search"]["search_to_correct"] == 0.5
        assert metrics["search"]["no_search_to_correct"] == 1.0
        assert metrics["offline_rescore"]["matches"] == 3

    def test_per_source_bucket(self):
        episode = make_episode(searched=True, correct=True)
        episode["source"] = "hotpotqa"
        metrics = eval_mod.aggregate_metrics([episode])
        assert metrics["per_source"]["hotpotqa"]["n"] == 1
        assert metrics["per_source"]["hotpotqa"]["em_rate"] == 1.0


class TestOfflineRescore:
    def test_extracts_last_answer_and_scores(self):
        episode = make_episode(searched=True, correct=True)
        # Remove precomputed offline and recompute from actions_text.
        result = eval_mod.offline_rescore(episode["steps"], episode["answers"])
        assert result["score"] == 1.0
        assert result["final_answer"] == "Paris"

    def test_actions_text_mirrors_chat_history(self):
        episode = make_episode(searched=True, correct=True)
        text = eval_mod.actions_text(episode["steps"])
        assert episode["steps"][0]["raw_action"] in text
        assert episode["steps"][0]["observation"] in text  # information interleaved
        assert "<answer>Paris</answer>" in text


class TestVerifyCleanUpstream:
    @staticmethod
    def _purge_agent_system() -> dict[str, object]:
        snapshot = {name: sys.modules[name] for name in list(sys.modules) if name == "agent_system" or name.startswith("agent_system.")}
        for name in snapshot:
            del sys.modules[name]
        return snapshot

    def test_marker_detection_in_fake_tree(self, tmp_path, monkeypatch):
        fake = tmp_path / "fake-clean"
        (fake / "agent_system" / "environments").mkdir(parents=True)
        (fake / "agent_system" / "__init__.py").write_text("")
        marker_file = fake / "agent_system" / "environments" / "envs.py"
        marker_file.write_text("x = bool(getattr(env_config, 'search_aware_step_reward', False))")

        monkeypatch.syspath_prepend(str(fake))
        snapshot = self._purge_agent_system()
        try:
            with pytest.raises(RuntimeError, match="patch marker"):
                eval_mod.verify_clean_upstream(fake)

            marker_file.write_text("x = 1")
            info = eval_mod.verify_clean_upstream(fake)
            assert info["patch_markers"] == 0
            assert str(Path(info["agent_system_module"]).resolve()).startswith(str(fake.resolve()))
        finally:
            for name in list(sys.modules):
                if name == "agent_system" or name.startswith("agent_system."):
                    del sys.modules[name]
            sys.modules.update(snapshot)

    def test_clean_tree_itself_passes(self):
        snapshot = self._purge_agent_system()
        try:
            # Re-import from the CLEAN tree (it must be first on sys.path).
            sys.path.insert(0, str(CLEAN_TREE))
            try:
                info = eval_mod.verify_clean_upstream(CLEAN_TREE)
            finally:
                sys.path.remove(str(CLEAN_TREE))
            assert info["patch_markers"] == 0
            assert str(Path(info["agent_system_module"]).resolve()).startswith(str(CLEAN_TREE.resolve()))
        finally:
            for name in list(sys.modules):
                if name == "agent_system" or name.startswith("agent_system."):
                    del sys.modules[name]
            sys.modules.update(snapshot)


class TestDataManifest:
    def test_smoke_manifest_sha_matches_eval_read(self):
        # The wrapper pins smoke test.parquet; verify the script's sha256 helper
        # agrees with the manifest's recorded value (data integrity for the gate).
        data_files = (
            Path("/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke/test.parquet")
        )
        manifest = Path(
            "/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke/manifest.json"
        )
        if not data_files.is_file() or not manifest.is_file():
            pytest.skip("data files not present on this host")
        expected = json.loads(manifest.read_text())["outputs"]["test"]["sha256"]
        assert eval_mod.sha256_file(data_files) == expected
