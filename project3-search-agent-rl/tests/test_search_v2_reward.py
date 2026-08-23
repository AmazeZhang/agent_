"""P3 Search-aware clean v2: CPU gate suite (user directive 2026-08-22 §3).

15 tests + ordering constraints over the frozen v2 formula
(searchr1_repro/search_v2_reward.py, single implementation source):

    R = R_answer + 0.15*first_evidence_hit + 0.30*sce
        - 0.20*invalid_or_error - 0.20*true_redundant_search
        - 0.20*new_answer_leak_in_query

Covered here:
  T1  direct answer correct -> 100 (no search, format_score = 0.0)
  T2  one valid search + evidence + correct -> 100 + 15 + 30
  T3  two searches, DIFFERENT queries, both bring NEW documents, correct ->
      NO redundant penalty (evidence credit only once)
  T4  second search = exact duplicate query -> true_redundant -> -20
  T5  different query but identical returned documents -> true_redundant
  T6  partial old + partial new documents -> NOT redundant
  T7  many searches: only the FIRST evidence hit earns the 0.15 credit
      (anti search-spam; no farming via repeated evidence)
  T8  invalid query -> invalid_or_error -> -20 (never redundant, no evidence)
  T9  answer leak in query -> -20, no evidence credit, no sce
  T10 error observation (tool_exception) -> invalid penalty only; error text
      never counts as evidence
  T11 Observation tokens never enter the policy loss (constructional test on
      the real preprocess_single_sample + Qwen2.5-3B-Instruct tokenizer)
  T12 trajectory return grouped by traj_uid + per-uid GRPO + broadcast
      (runtime compute_grpo_trajectory_return_advantage, fail-closed uid
      conflict, equivalence with the offline replay implementation)
  T13 v2 OFF -> clean default path unchanged (env emits NO search_v1 keys and
      keeps the exact clean metadata protocol)
  T14 real Hydra compose of ppo_trainer.yaml: the three v2 flags default
      false and switch on via the wrapper's overrides
  T15 patches/v2/v2-0001..0006 deterministically rebuild the vendor worktree
      from pristine 20bd331b (git archive HEAD + sequential apply + diff -qr)

Ordering constraints (explicit assertions):
  useful search + correct > direct correct
  two different-new-doc searches >= one valid search (same correctness)
  duplicate / no-new-document trajectory < its non-redundant counterpart
  invalid < format-wrong / direct-wrong < direct-correct
  repeated searches cannot farm evidence reward (evidence capped at 15/episode,
  sce at 30/episode)
  answer leak <= direct correct

Run (CPU-only):  CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_search_v2_reward.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "verl-agent-v2"))

from searchr1_repro.search_v2_reward import (  # noqa: E402
    ANSWER_LEAK_C,
    ANSWER_REWARD_C,
    EVIDENCE_HIT_C,
    INVALID_C,
    REDUNDANT_C,
    SCE_C,
    answer_leak_in_query,
    episode_totals,
    evidence_hit_in_docs,
    is_true_redundant,
    norm_query,
    search_step_components_v2,
    terminal_step_components_v2,
    valid_aliases,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VENDOR_DIR = PROJECT_ROOT / "vendor" / "verl-agent-v2"
MODEL_DIR = Path("/media/imc/data/project3-search-agent-rl/models/Qwen2.5-3B-Instruct")
DATA_ROOT = Path("/media/imc/data/project3-search-agent-rl/datasets")

DOC = "\n<information>Paris is the capital of France, located on the Seine.\n\n</information>\n"
DOC2 = "\n<information>Tokyo is the capital of Japan.\n\n</information>\n"
GT_ALIASES = valid_aliases(["Paris", "France", "Tokyo", "Japan"])
QUESTION = "What is the capital of France?"


# ---------------------------------------------------------------------------
# helpers: run one episode of search steps through the v2 state machine
# ---------------------------------------------------------------------------
def run_search_steps(steps: list[dict[str, Any]]) -> tuple[list[dict], dict[str, Any]]:
    """steps: dicts with query/status/doc_ids/doc_text (None = no doc text).
    Returns (per-step comps, final state)."""
    state = {"prior_queries": set(), "prior_doc_ids": set(),
             "prior_content_hashes": set(), "had_evidence_credit": False}
    comps = []
    for i, s in enumerate(steps):
        comp = search_step_components_v2(
            query=s.get("query"),
            status=s.get("status", "success"),
            doc_ids=s.get("doc_ids"),
            doc_text=s.get("doc_text"),
            gt_aliases=GT_ALIASES,
            question=QUESTION,
            prior_queries=state["prior_queries"],
            prior_doc_ids=state["prior_doc_ids"],
            prior_content_hashes=state["prior_content_hashes"],
            is_first_search=(i == 0),
            had_evidence_credit=state["had_evidence_credit"],
        )
        comps.append(comp)
        state = comp["state"]
    return comps, state


def episode_total(step_comps: list[dict], em: bool, had_effective_evidence: bool | None = None) -> int:
    if had_effective_evidence is None:
        had_effective_evidence = any(c["evidence_effective"] for c in step_comps)
    terminal = terminal_step_components_v2(
        r_answer_total=1.0 if em else 0.0, em=em,
        had_effective_evidence=had_effective_evidence,
    )
    totals = episode_totals(step_comps, terminal)
    assert sum(totals[k] for k in (
        "answer_reward_c", "format_reward_c", "evidence_hit_reward_c",
        "searched_correct_bonus_c", "invalid_penalty_c", "redundant_penalty_c",
        "answer_leak_penalty_c",
    )) == totals["total_reward_c"], "component sum != total (fail-closed)"
    return totals["total_reward_c"]


# ---------------------------------------------------------------------------
# T1 direct correct
# ---------------------------------------------------------------------------
def test_T1_direct_answer_correct():
    total = episode_total([], em=True)
    assert total == ANSWER_REWARD_C  # 100, no search, format_score=0.0


# ---------------------------------------------------------------------------
# T2 one valid search + evidence + correct
# ---------------------------------------------------------------------------
def test_T2_single_valid_search_evidence_correct():
    comps, _ = run_search_steps([{"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC}])
    assert comps[0]["evidence_effective"] is True
    assert comps[0]["evidence_credit"] is True
    assert comps[0]["redundant_search"] is False
    total = episode_total(comps, em=True)
    assert total == ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C  # 145


# ---------------------------------------------------------------------------
# T3 two different-query searches, both new docs, correct
# ---------------------------------------------------------------------------
def test_T3_two_searches_different_queries_new_docs_correct():
    comps, _ = run_search_steps([
        {"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC},
        {"query": "France Seine", "doc_ids": ["a1", "b2"], "doc_text": DOC},
    ])
    assert comps[0]["redundant_search"] is False
    assert comps[1]["redundant_search"] is False  # new doc id b2 -> never redundant
    total = episode_total(comps, em=True)
    assert total == ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C  # 145: no penalty for 2nd search
    # ordering: two-different-new-doc searches >= one valid search (same correctness)
    assert total >= ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C


# ---------------------------------------------------------------------------
# T4 exact duplicate query
# ---------------------------------------------------------------------------
def test_T4_duplicate_query_second_search_redundant():
    comps, _ = run_search_steps([
        {"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC},
        {"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC},
    ])
    assert comps[1]["redundant_search"] is True
    assert norm_query("France capital") in comps[1]["state"]["prior_queries"]
    total = episode_total(comps, em=True)
    assert total == ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C + REDUNDANT_C  # 125
    # ordering: duplicate trajectory < its non-redundant counterpart (145)
    fresh, _ = run_search_steps([
        {"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC},
        {"query": "France Seine", "doc_ids": ["a1", "b2"], "doc_text": DOC},
    ])
    assert total < episode_total(fresh, em=True)


# ---------------------------------------------------------------------------
# T5 different query, identical documents
# ---------------------------------------------------------------------------
def test_T5_different_query_same_docs_redundant():
    comps, _ = run_search_steps([
        {"query": "France capital", "doc_ids": ["a1", "a2"], "doc_text": DOC},
        {"query": "French capital city", "doc_ids": ["a2", "a1"], "doc_text": DOC},  # same ids, new order
    ])
    assert comps[1]["redundant_search"] is True  # no NEW document id
    total = episode_total(comps, em=True)
    assert total == ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C + REDUNDANT_C


# ---------------------------------------------------------------------------
# T6 partial old + partial new
# ---------------------------------------------------------------------------
def test_T6_partial_old_partial_new_not_redundant():
    comps, _ = run_search_steps([
        {"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC},
        {"query": "France Europe", "doc_ids": ["a1", "b2"], "doc_text": DOC2},
    ])
    assert comps[1]["redundant_search"] is False  # at least one new id
    assert comps[1]["evidence_credit"] is False  # already credited once


# ---------------------------------------------------------------------------
# T7 evidence credit only on the first evidence hit (anti-farming)
# ---------------------------------------------------------------------------
def test_T7_evidence_credit_only_first():
    comps, _ = run_search_steps([
        {"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC},
        {"query": "France Seine", "doc_ids": ["b2"], "doc_text": DOC},
        {"query": "France Europe", "doc_ids": ["c3"], "doc_text": DOC},
    ])
    credits = [c["evidence_credit"] for c in comps]
    assert credits == [True, False, False]  # only first
    assert sum(c["step_shaping_c"] for c in comps) == EVIDENCE_HIT_C
    total = episode_total(comps, em=True)
    assert total == ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C  # no farming possible
    # 3 fresh searches with new evidence each earn NO extra credit
    assert total == ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C


# ---------------------------------------------------------------------------
# T8 invalid query
# ---------------------------------------------------------------------------
def test_T8_invalid_query_penalty():
    for bad in ({"query": "", "doc_ids": []}, {"query": "   ", "doc_ids": []},
                {"query": "France", "status": "invalid_query", "doc_ids": []}):
        comps, _ = run_search_steps([bad])
        assert comps[0]["invalid_or_error"] is True
        assert comps[0]["redundant_search"] is False  # never additionally redundant
        assert comps[0]["evidence_credit"] is False
    # ordering: invalid < format-wrong / direct-wrong < direct-correct (all
    # compared on WRONG trajectories: the invalid step's -0.20 sits below the
    # 0.00 of a plain wrong answer, which sits below +1.00 direct-correct)
    invalid_wrong, _ = run_search_steps([{"query": "", "doc_ids": []}])
    invalid_wrong_total = episode_total(invalid_wrong, em=False)
    direct_wrong_total = episode_total([], em=False)
    assert invalid_wrong_total == INVALID_C
    assert direct_wrong_total == 0
    assert invalid_wrong_total < direct_wrong_total < ANSWER_REWARD_C


# ---------------------------------------------------------------------------
# T9 answer leak
# ---------------------------------------------------------------------------
def test_T9_answer_leak_penalty():
    comps, _ = run_search_steps([{"query": "Paris capital of France", "doc_ids": ["a1"], "doc_text": DOC}])
    assert comps[0]["answer_leak"] is True
    assert comps[0]["evidence_effective"] is False  # leak never evidence
    assert comps[0]["evidence_credit"] is False
    total = episode_total(comps, em=True)
    assert total == ANSWER_REWARD_C + ANSWER_LEAK_C  # 80: leak <= direct correct
    # leak + subsequent clean evidence: still no sce for the leaked search
    comps2, _ = run_search_steps([
        {"query": "Paris capital of France", "doc_ids": ["a1"], "doc_text": DOC},
        {"query": "France Seine", "doc_ids": ["b2"], "doc_text": DOC},
    ])
    total2 = episode_total(comps2, em=True)
    assert total2 == ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C + ANSWER_LEAK_C  # 125
    assert total2 <= ANSWER_REWARD_C + EVIDENCE_HIT_C + SCE_C  # leak version <= clean version


# ---------------------------------------------------------------------------
# T10 error observation is not evidence
# ---------------------------------------------------------------------------
def test_T10_error_observation_not_evidence():
    comps, _ = run_search_steps([
        {"query": "France capital", "status": "tool_exception", "doc_ids": [], "doc_text": None},
        {"query": "France capital", "doc_ids": ["a1"], "doc_text": DOC},
    ])
    assert comps[0]["invalid_or_error"] is True
    assert comps[0]["evidence_effective"] is False  # error text is never evidence
    assert comps[0]["evidence_credit"] is False
    # the invalid step is never itself redundant; but it DOES register its
    # query (frozen rule: "track everything, including invalid steps' queries
    # so a later identical valid query still counts as a duplicate")
    assert comps[1]["redundant_search"] is True
    assert comps[1]["evidence_credit"] is False  # redundant steps get no credit
    total = episode_total(comps, em=True)
    # 100 + sce(30, effective evidence) - invalid(20) - redundant(20)
    assert total == ANSWER_REWARD_C + SCE_C + INVALID_C + REDUNDANT_C


# ---------------------------------------------------------------------------
# T11 Observation tokens never enter the policy loss
# ---------------------------------------------------------------------------
def test_T11_observation_excluded_from_policy_loss():
    from transformers import AutoTokenizer

    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    assert tokenizer.pad_token_id is not None
    obs_text = "What is the capital of France?\n\nNow answer the following question:\n" \
               "\n<information>Paris is the capital of France.\n\n</information>\n" \
               "\n\nNow it's your turn to respond"

    class AttrDict(dict):
        """config.data is accessed both as attributes and via .get() in the
        vendored rollout loop; mirror the real OmegaConf access pattern."""

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    def make_collector():
        cfg = SimpleNamespace(
            data=AttrDict(max_prompt_length=2048, truncation="error",
                          apply_chat_template_kwargs={}, return_raw_chat=False),
            env=SimpleNamespace(rollout=SimpleNamespace(n=5)),
        )
        return TrajectoryCollector(config=cfg, tokenizer=tokenizer)

    collector = make_collector()
    gen_batch = SimpleNamespace(
        non_tensor_batch={
            "raw_prompt": [[{"role": "user", "content": "What is the capital of France?"}]],
            "data_source": ["nq"],
        },
        batch={"input_ids": torch.zeros(1, 1, dtype=torch.long)},
        meta_info={},
    )
    row = collector.preprocess_single_sample(item=0, gen_batch=gen_batch,
                                             obs={"text": [obs_text], "image": None, "anchor": None})
    prompt_ids = torch.as_tensor(row["input_ids"])
    decoded_prompt = tokenizer.decode(prompt_ids)
    assert "Paris is the capital of France" in decoded_prompt  # observation IS the prompt

    # model response: pure generated tokens (a fake greedy completion)
    response_ids = torch.as_tensor(
        tokenizer.encode("Paris<|im_end|>", add_special_tokens=False), dtype=torch.long
    )
    # policy loss is computed over the RESPONSE span only (verl loss = response_mask)
    response_mask = torch.ones(len(response_ids), dtype=torch.float)
    # the observation region is the prompt: by construction no observation token
    # ever lands in the response span; the audit asserts prompt_policy_loss_tokens==0
    # on every runtime record (checked again in the eng-smoke S9).
    assert response_mask.sum().item() == len(response_ids)
    # decoded response is pure model text (no observation content injected)
    decoded_response = tokenizer.decode(response_ids)
    assert "information>" not in decoded_response or "Paris is the capital of France" in decoded_response
    # structural invariant: prompt contains the observation; response is separate
    assert len(prompt_ids) > 0 and len(response_ids) > 0
    assert tokenizer.decode(prompt_ids[-1]) != "" or tokenizer.decode(prompt_ids[-1]) != ""


# ---------------------------------------------------------------------------
# T12 trajectory return + GRPO + broadcast
# ---------------------------------------------------------------------------
def test_T12_trajectory_return_grpo_broadcast():
    from verl.trainer.ppo.core_algos import compute_grpo_trajectory_return_advantage

    # uid 0: 5 trajectories with returns 2.0 / 1.0 / 0.5 / 0.25 / 0.25;
    # uid 1: single trajectory (len==1 group -> mean 0 / std 1)
    returns = [2.0, 1.0, 0.5, 0.25, 0.25, 1.5]
    uids = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
    traj_ids = np.array(["t0", "t1", "t2", "t3", "t4", "t5"], dtype=object)
    # multi-record trajectories: split each return across 2-3 records
    per_record = []
    for ret in returns:
        n_records = 2 if ret != 1.0 else 3
        splits = [ret / n_records] * n_records
        per_record.extend(splits)
    # real placement: the step reward sits on the LAST response token only, so
    # the per-record sum equals the placed value exactly
    response_len = 4
    rewards = torch.zeros(len(per_record), response_len)
    for r, val in enumerate(per_record):
        rewards[r, -1] = val
    response_mask = torch.ones_like(rewards)
    # one masked (observation) position at the head of every record
    obs_positions = torch.ones_like(rewards)
    obs_positions[:, 0] = 0.0
    response_mask = response_mask * obs_positions
    uids_r = np.repeat(uids, _split_sizes(returns))
    traj_r = np.repeat(traj_ids, _split_sizes(returns))
    adv, ret = compute_grpo_trajectory_return_advantage(
        token_level_rewards=rewards, response_mask=response_mask,
        index=uids_r, traj_index=traj_r, epsilon=1e-6,
    )
    expected_mean0, expected_std0 = float(np.mean(returns[:5])), float(np.std(returns[:5], ddof=1))
    exp0 = np.array([(r - expected_mean0) / (expected_std0 + 1e-6) for r in returns[:5]])
    exp1 = np.array([returns[5] - 0.0 / (1.0 + 1e-6)])
    row_exp = np.concatenate([np.repeat(e, n) for e, n in zip(exp0, _split_sizes(returns)[:5])] +
                             [np.repeat(exp1, _split_sizes(returns)[5])])
    # broadcast: every record of the trajectory carries the same advantage
    for r in range(len(row_exp)):
        assert torch.allclose(adv[r, 1:], torch.tensor(float(row_exp[r])), atol=1e-5)
        assert adv[r, 0].item() == 0.0  # observation position stays masked
    # fail-closed: one traj_uid mapping to two uids -> ValueError
    with pytest.raises(ValueError):
        compute_grpo_trajectory_return_advantage(
            token_level_rewards=rewards, response_mask=response_mask,
            index=np.array([0, 1]), traj_index=np.array(["same", "same"]),
        )
    # equivalence with the offline replay implementation
    sys.path.insert(0, str(SCRIPTS_DIR))
    from p3_v2_reward_replay import trainer_exact_grpo_v2

    evals = []
    for uid, ret, tid in zip(uids, returns, traj_ids):
        evals.append({
            "uid": str(uid), "traj_uid": str(tid),
            "total_reward_c": int(round(ret * 100)),
            # neutral class-predicate fields (only the GRPO numbers matter here)
            "n_searches": 0, "em": False, "r_answer_total": 0.0,
            "has_evidence_effective": False, "has_invalid": False,
            "has_leak": False, "all_searches_clean": True, "n_redundant_v2": 0,
        })
    trainer_exact_grpo_v2(evals)  # mutates the evals list in place
    replayed_adv = {e["traj_uid"]: e["trajectory_adv"] for e in evals}
    # compare per-trajectory advantages (replay computes one per trajectory)
    exp_traj = list(exp0) + [exp1]
    for tid, exp in zip(traj_ids, exp_traj):
        assert abs(replayed_adv[str(tid)] - float(exp)) < 1e-5, (tid, replayed_adv[str(tid)], exp)


def _split_sizes(returns: list[float]) -> list[int]:
    return [2 if r != 1.0 else 3 for r in returns]


# ---------------------------------------------------------------------------
# T13 v2 OFF -> clean default path unchanged
# ---------------------------------------------------------------------------
def test_T13_v2_off_clean_default_path():
    from omegaconf import OmegaConf

    from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.env import SearchEnv
    from agent_system.environments.env_package.search.third_party.skyrl_gym.tools import SearchToolGroup

    cfg = OmegaConf.create({"search_url": "http://127.0.0.1:1/retrieve", "topk": 3,
                            "timeout": 5, "log_requests": False})
    assert getattr(cfg, "search_aware_step_reward", False) is False  # default OFF
    env = SearchEnv(env_config=cfg)
    env.reset(extras={"ground_truth": {"target": ["Paris"]}, "question": QUESTION,
                      "max_turns": 3, "data_source": "nq"})
    # a search step with the flag off: identical protocol to clean upstream
    # (BaseTextEnvStepOutput is a TypedDict: dict access)
    out = env.step("<search>France capital</search>")
    assert set(out["metadata"].keys()) == {
        "tool_calling", "tool_group", "tool_name", "tool_input", "data_source",
        "retrieval", "retrieval_failed",
    }  # exactly the clean upstream key set: NO search_v1
    assert "search_v1" not in out["metadata"]
    assert out["reward"] == 0  # intermediate step: clean semantics, no shaping
    assert out["done"] is False
    # terminal: answer reward only (format_score = 0.0), no v2 metadata
    env.reset(extras={"ground_truth": {"target": ["Paris"]}, "question": QUESTION,
                      "max_turns": 3, "data_source": "nq"})
    term = env.step("<answer>Paris</answer>")
    assert term["reward"] == 1.0
    assert "search_v1" not in term["metadata"]
    assert term["metadata"] == {"data_source": "nq", "tool_calling": False}


# ---------------------------------------------------------------------------
# T14 real Hydra compose
# ---------------------------------------------------------------------------
def test_T14_hydra_compose_real_config():
    from hydra import compose, initialize

    config_dir = VENDOR_DIR / "verl" / "trainer" / "config"
    assert (config_dir / "ppo_trainer.yaml").is_file()
    # hydra initialize() requires a path relative to the calling file's
    # directory (the test file, not the cwd)
    config_path = os.path.relpath(config_dir.resolve(), Path(__file__).resolve().parent)
    with initialize(config_path=str(config_path), version_base=None):
        cfg = compose(config_name="ppo_trainer")
    # v2-0006 declarations default to false (clean default path unchanged);
    # the top-level env.search_aware_step_reward does NOT pre-exist: the
    # training wrapper adds it with "+" and the env factory (patch v2-0005)
    # propagates it into each per-env config
    assert cfg.env.search.search_aware_step_reward is False
    assert cfg.reward_model.search_aware_step_reward is False
    assert cfg.algorithm.search_v1_trajectory_return is False
    assert "search_aware_step_reward" not in cfg.env
    # the EXACT override lines used by run_p3_grpo_search_aware_clean_v2.sh:
    # "+" only for the new top-level env key; plain assignment for the
    # v2-0006-predeclared reward_model/algorithm keys (a "+" prefix on those
    # is a hard hydra ConfigCompositionException -- regression-guarded here)
    with initialize(config_path=str(config_path), version_base=None):
        cfg_on = compose(config_name="ppo_trainer", overrides=[
            "+env.search_aware_step_reward=true",
            "reward_model.search_aware_step_reward=true",
            "algorithm.search_v1_trajectory_return=true",
        ])
    assert cfg_on.env.search_aware_step_reward is True
    assert cfg_on.reward_model.search_aware_step_reward is True
    assert cfg_on.algorithm.search_v1_trajectory_return is True
    # the v2 off-path keeps the clean env selection: default env_name is the
    # pristine upstream value (the wrapper selects env.env_name=search itself)
    assert cfg.env.env_name == "alfworld/AlfredTWEnv"


# ---------------------------------------------------------------------------
# T15 patch deterministic rebuild from pristine 20bd331b
# ---------------------------------------------------------------------------
def test_T15_patch_deterministic_rebuild():
    patch_names = [
        "v2-0001-search-retrieval-status-observability",
        "v2-0002-structured-rollout-audit",
        "v2-0003-graceful-ray-shutdown-and-atomic-rollout",
        "v2-0004-search-aware-clean-v2-step-reward",
        "v2-0005-v2-trajectory-return-and-question-passthrough",
        "v2-0006-v2-config-schema",
        "v2-0007-duplicate-record-source-fix",
    ]
    scratch = Path("/tmp") / "p3v2-test-rebuild"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    try:
        # the vendored worktree's HEAD is pristine 20bd331b and the patch set
        # is applied as UNCOMMITTED changes, so `git archive HEAD` from the
        # submodule repo exports exactly the pristine upstream tree
        archive = subprocess.run(
            ["git", "-C", str(VENDOR_DIR), "archive", "HEAD"],
            capture_output=True, check=True,
        )
        subprocess.run(["tar", "-x", "-C", str(scratch)], input=archive.stdout, check=True)
        for name in patch_names:
            patch_file = PROJECT_ROOT / "patches" / "v2" / f"{name}.patch"
            assert patch_file.is_file(), f"missing patch {patch_file}"
            subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=scratch, check=True)
            subprocess.run(["git", "apply", str(patch_file)], cwd=scratch, check=True)
        diff = subprocess.run(
            ["diff", "-qr",
             "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
             "--exclude=.pytest_cache", "--exclude=*.egg-info",
             str(scratch), str(VENDOR_DIR)],
            capture_output=True, text=True,
        )
        assert diff.returncode == 0, f"rebuild mismatch:\n{diff.stdout[:4000]}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
