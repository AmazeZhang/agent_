#!/usr/bin/env python3
"""Search-aware clean v2 evaluation harness (2026-08-22 directive).

Evaluates a model under the v2 line's CLEAN protocol -- identical to the
clean-upstream line, because the v2 patches must not change prompt, projection
or terminal reward semantics (v1's 0004/0005 are NOT applied):

- SearchEnvironmentManager (env_manager.py, byte-identical to pristine) +
  upstream search_projection (byte-identical) + SEARCH_TEMPLATE_NO_HIS /
  SEARCH_TEMPLATE prompts (byte-identical) + skyrl compute_score
  format_score=0.0 (byte-identical). The eval env config does NOT set
  env.search.search_aware_step_reward, so the env runs its byte-unchanged
  clean path (the v2 shaping branch never executes).
- max_steps=4 / history_length=4 / topk=3 (clean-line protocol).

The tree gate (--v2-dir + --pristine-dir) aborts unless:
  (1) the imported agent_system resolves under --v2-dir, and
  (2) the v2 tree DOES contain the search_aware_step_reward marker (v2 line),
      and
  (3) the four protocol-critical files are BYTE-IDENTICAL to the pristine
      reference tree (--pristine-dir = vendor/upstream-20bd331b):
      env_manager.py, projection.py, environments/prompts/search.py,
      skyrl utils.py
      (compute_score / extract_solution). The remaining patched files
      (envs.py/env.py/search.py/rollout_loop.py/...) add ONLY observability
      metadata, the search_v1 shaping dicts and the config switches; their
      clean-path byte-equivalence is covered by the CPU default-path
      regression test (tests/test_search_v2_reward.py) and by this script's
      empirical protocol gate (EM / search rate / invalid / compliance bands
      vs the clean Step0 baseline).

Modes:
  - main (temperature=0.0, num_rollouts=1, greedy): confirm-256 evaluation;
    the smoke-16 round-2 prompt gate applies fail-closed.
  - diagnose (temperature>0 and/or num_rollouts>1): behaviour probe on the
    fixed dev64 set (5 rollouts/question, per-rollout fixed seeds); reports the
    same metrics plus the v2 search-behaviour stats; never blocks.

v2 search-behaviour metrics (from the v2-0001 observability metadata in
info.retrieval): per-episode search rounds, multi-hop (>=2 searches) ratio,
valid/invalid query distribution, per-episode NEW document increments (v2
"no new document" definition) and TRUE-redundant rate (duplicate normalized
query OR no new document ID, content-hash fallback -- first search never
redundant). Evidence-hit/correct linkage uses the frozen v2 matcher
(searchr1_repro/search_v2_reward.py, project-side single implementation).

By construction this script creates no optimizer, no scheduler, never calls
backward, and does not import Ray. torch.cuda.max_memory_allocated/reserved
are recorded as torch-allocator views (vLLM's internal blocks may be tracked
outside the torch allocator); the per-GPU PHYSICAL nvidia-smi peaks are
recorded by the wrapper (run_p3_eval_v2.sh) separately -- never conflate the
two numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import vllm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.env_manager import SearchEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv  # noqa: E402
from agent_system.environments.env_package.search.projection import search_projection  # noqa: E402
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (  # noqa: E402
    compute_score as skyrl_compute_score,
    extract_solution,
)
from searchr1_repro.search_v2_reward import (  # noqa: E402 -- frozen v2 matcher
    evidence_hit_in_docs,
    is_true_redundant,
    norm_query,
    valid_aliases,
)
from searchr1_repro.stepsearch_protocol import (  # noqa: E402
    STEPSEARCH_SOURCE_COMMIT,
    build_stepsearch_prompt,
    stepsearch_prompt_contains_query,
    truncate_stepsearch_response,
)

EXPECTED_RETRIEVER_VECTORS = 21_015_324
REAL_INDEX_NOTE = "real Wiki-18 IndexFlatIP (21,015,324 vectors); ground-truth-derived fixture is prohibited for evaluation"
PATCH_MARKER = "search_aware_step_reward"  # introduced by the v2 patch series; MUST be PRESENT in the v2 tree

# Protocol-critical files that must be byte-identical to pristine (relative to
# the agent_system package root): the v2 line restores the clean protocol, so
# any byte drift here is a protocol violation, not a style preference.
PROTOCOL_FILES = (
    "environments/env_manager.py",
    "environments/env_package/search/projection.py",
    "environments/prompts/search.py",
    "environments/env_package/search/third_party/skyrl_gym/envs/search/utils.py",
)

# Engine parity with the training rollout path (V0 engine).
VLLM_DTYPE = "bfloat16"
VLLM_GPU_MEMORY_UTILIZATION = 0.6
VLLM_MAX_MODEL_LEN = 3328  # max_input_tokens(3072) + max_new_tokens(256)


class StepSearchEnvironmentManager(SearchEnvironmentManager):
    """Preserve StepSearch's raw plan/observation text in the next-turn trace.

    The tool receives the unchanged upstream projected action, while memory
    retains the raw model response.  This matches StepSearch's public rollout,
    which appends generated plan/search text and the returned information to
    the rolling context.
    """

    def step_projected(
        self,
        raw_actions: list[str],
        projected_actions: list[str],
        valids,
    ):
        from agent_system.environments.base import to_numpy

        next_obs, rewards, dones, infos = self.envs.step(projected_actions)
        self.memory.store({"search": raw_actions, "information": next_obs})
        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
        }
        for index, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[index])
        return next_observations, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, text_obs: list[str], init: bool = False) -> list[str]:
        memory_contexts = [""] * len(text_obs)
        if not init and self.config.env.history_length > 0:
            memory_contexts, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search",
            )
        return [
            build_stepsearch_prompt(self.tasks[index], memory_contexts[index])
            for index in range(len(text_obs))
        ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def atomic_write_text(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    with partial.open("ab") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [
        json.dumps(jsonable(item), ensure_ascii=False, sort_keys=True) + "\n" for item in records
    ]
    atomic_write_text(path, "".join(lines))


# --------------------------------------------------------------------------- #
# Gates (abort on failure, never warn-and-continue)
# --------------------------------------------------------------------------- #

def verify_v2_tree(v2_dir: Path, pristine_dir: Path) -> dict[str, Any]:
    """The imported agent_system must live inside --v2-dir AND the four
    protocol-critical files must be byte-identical to the pristine reference
    tree. Any deviation aborts (v2 line contract: prompt / projection /
    terminal reward are the clean protocol, restored byte-for-byte).

    agent_system is a NAMESPACE package (no __init__.py in upstream), so the
    package location is read from __path__, not __file__."""
    import agent_system
    package_paths = [Path(p).resolve() for p in getattr(agent_system, "__path__", [])]
    root = v2_dir.resolve()
    if not package_paths or not any(str(p).startswith(str(root)) for p in package_paths):
        raise RuntimeError(
            f"imported agent_system does not resolve under the v2 tree {root}: {package_paths}"
        )
    markers = []
    for path in sorted((root / "agent_system").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        if PATCH_MARKER in text:
            markers.append(str(path))
    if not markers:
        raise RuntimeError(
            f"v2 tree contains NO patch marker '{PATCH_MARKER}': this is not the v2 "
            f"line (use --pristine-dir with run_p3_eval_upstream_clean.py instead)"
        )
    pristine = pristine_dir.resolve()
    if not pristine.is_dir():
        raise RuntimeError(f"pristine reference tree missing: {pristine}")
    protocol_checks: dict[str, Any] = {}
    violations = []
    for relative in PROTOCOL_FILES:
        v2_path = root / "agent_system" / relative
        pristine_path = pristine / "agent_system" / relative
        if not v2_path.is_file():
            violations.append(f"{relative}: missing in v2 tree")
            continue
        if not pristine_path.is_file():
            violations.append(f"{relative}: missing in pristine tree")
            continue
        v2_bytes = v2_path.read_bytes()
        pristine_bytes = pristine_path.read_bytes()
        same = v2_bytes == pristine_bytes
        protocol_checks[relative] = {
            "byte_identical": same,
            "sha256_v2": hashlib.sha256(v2_bytes).hexdigest(),
            "sha256_pristine": hashlib.sha256(pristine_bytes).hexdigest(),
        }
        if not same:
            violations.append(
                f"{relative}: byte drift vs pristine "
                f"(v2 {protocol_checks[relative]['sha256_v2'][:12]} != "
                f"pristine {protocol_checks[relative]['sha256_pristine'][:12]})"
            )
    if violations:
        raise RuntimeError(
            "v2 protocol gate failed (clean protocol files must be byte-identical "
            "to pristine 20bd331b):\n" + "\n".join(violations)
        )
    return {
        "v2_dir": str(root),
        "pristine_dir": str(pristine),
        "agent_system_module": str(package_paths[0]),
        "patch_marker_files": len(markers),
        "protocol_files": protocol_checks,
    }


def load_eval_records(data_files: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(data_files)
    records = []
    for _, row in frame.iterrows():
        env_kwargs = row["env_kwargs"]
        records.append(
            {
                "question": str(env_kwargs["question"]),
                "answers": list(env_kwargs["ground_truth"]["target"]),
                "data_source": str(row["data_source"]),
            }
        )
    return records


def leakage_check(records: list[dict[str, Any]], leakage_reference: Path) -> dict[str, int]:
    reference = pd.read_parquet(leakage_reference)
    reference_questions = {
        normalize_question(str(item["question"])) for item in reference["env_kwargs"]
    }
    eval_questions = {normalize_question(record["question"]) for record in records}
    overlap = eval_questions & reference_questions
    if overlap:
        raise RuntimeError(f"leakage: {len(overlap)} eval questions appear in the training split")
    return {"eval_questions": len(eval_questions), "reference_questions": len(reference_questions), "overlap": len(overlap)}


def retriever_health_check(search_url: str, expected_vectors: int = EXPECTED_RETRIEVER_VECTORS) -> dict[str, Any]:
    import json as _json
    from urllib.request import urlopen

    health_url = search_url.rsplit("/", 1)[0] + "/health"
    with urlopen(health_url, timeout=5) as response:
        payload = _json.load(response)
    if payload.get("status") != "ready" or payload.get("vectors") != expected_vectors:
        raise RuntimeError(f"retriever health gate failed: {payload}")
    return payload


def verify_data_hash(data_files: Path, manifest: Path | None, manifest_key: str) -> dict[str, Any]:
    actual = sha256_file(data_files)
    if manifest is None:
        return {"checked": False, "sha256": actual}
    data = json.loads(manifest.read_text())
    expected = data["outputs"][manifest_key]["sha256"]
    if actual != expected:
        raise RuntimeError(f"data file SHA256 mismatch: expected {expected}, got {actual}")
    return {"checked": True, "sha256": actual, "expected": expected}


def validate_managed_environment() -> dict[str, str]:
    run_id = os.environ.get("PROJECT3_RUN_ID", "")
    run_dir = os.environ.get("PROJECT3_RUN_DIR", "")
    if not run_id or not run_dir:
        raise RuntimeError("evaluation must run under scripts/run_managed.sh (PROJECT3_RUN_ID/PROJECT3_RUN_DIR)")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation must expose exactly one logical GPU (run_managed.sh with one physical GPU)")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and any(token.strip() == "0" for token in visible.split(",")):
        raise RuntimeError("physical GPU 0 must never be exposed to evaluation")
    return {"run_id": run_id, "run_dir": run_dir, "cuda_visible_devices": visible}


def validate_vllm_engine_parity() -> dict[str, Any]:
    if os.environ.get("VLLM_USE_V1", "0") != "0":
        raise RuntimeError(
            f"VLLM_USE_V1 must be \"0\" (training rollout engine path), got: {os.environ.get('VLLM_USE_V1')}"
        )
    return {"vllm_use_v1": "0", "vllm_version": vllm.__version__}


# --------------------------------------------------------------------------- #
# Action / answer analysis
# --------------------------------------------------------------------------- #

def actions_text(episode_steps: list[dict[str, Any]]) -> str:
    """Mirror the env chat_history: assistant action, then (if any) the user
    tool observation, per round — the exact string upstream compute_score sees."""
    parts: list[str] = []
    for step in episode_steps:
        parts.append(step["raw_action"])
        observation = step.get("observation") or ""
        if observation:
            parts.append(observation)
    return "\n".join(parts)


def offline_rescore(episode_steps: list[dict[str, Any]], answers: list[str]) -> dict[str, Any]:
    """Audit-only EM re-score over the concatenated model actions (upstream
    format_score=0.0, i.e. the EM-compatible component)."""
    solution = actions_text(episode_steps)
    final_answer = extract_solution(solution)
    if final_answer is None:
        return {"final_answer": None, "score": 0.0, "has_answer": False}
    score = skyrl_compute_score(
        solution, {"target": answers}, method="strict", format_score=0.0, score=1.0
    )
    return {"final_answer": final_answer, "score": float(score), "has_answer": True}


def sanitize_empty_search_actions(
    projected_actions: list[str], valids: list[int]
) -> tuple[list[str], list[int]]:
    """Degenerate empty queries must never reach the retriever.

    The official Search-R1 model occasionally emits ``<search></search>`` in
    later rounds (query degeneration). Upstream SearchToolGroup.search guards
    only ``None``, so an empty string reaches the retriever and 422s against
    its ``query: str = Field(min_length=1)`` schema; the error blob then
    pollutes the context and kills the episode's search chain. We replace
    such actions with ``""`` (the env returns an empty observation with NO
    HTTP request -- the same semantics as search_projection's no-tags case)
    and mark them invalid so episode records stay truthful."""
    sanitized: list[str] = []
    sanitized_valids: list[int] = []
    for action, valid in zip(projected_actions, valids):
        if action.startswith("<search>") and not action[8:-9].strip():
            sanitized.append("")
            sanitized_valids.append(0)
        else:
            sanitized.append(action)
            sanitized_valids.append(int(valid))
    return sanitized, sanitized_valids


def step_search_query(step: dict[str, Any]) -> str | None:
    """The query of a genuine search attempt: the projected action must be a
    <search> block AND the env must have actually executed the tool. Done
    rounds never execute tools (upstream metadata tool_calling=False), and
    non-done rounds always report tool_calling=True, so the conjunction is
    exactly 'the env called the search tool with this query'."""
    projected = (step.get("projected_action") or "").strip()
    match = re.fullmatch(r"<search>(.*?)</search>", projected, re.DOTALL)
    if match is None:
        return None
    if not step.get("tool_calling"):
        return None
    query = match.group(1).strip()
    return query if query else None


NO_EVIDENCE_TEXT = (
    "No relevant documents were found for this query. "
    "The knowledge base returned no evidence about this question."
)
RETRIEVAL_CONDITIONS = ("real", "shuffled", "no-evidence")


def install_retrieval_condition(
    condition: str,
    shuffle_step: int,
    no_evidence_docs: int,
    question_texts: list[str],
):
    """Runtime-only counterfactual retrieval conditions (vendor tree untouched).

    The v2-tree gate verifies the vendor directory byte-for-byte; this patch
    exists only in this process and only changes the EVIDENCE CONTENT, never
    the model's prompts, projection, decoding or step budget.

      - real:        no patch (baseline run).
      - shuffled:    the model's real query is still retrieved for real first
                     (status truthfulness: errors / empty / no-results are kept
                     verbatim and never remapped). On success, the returned
                     evidence is replaced by the REAL docs of the fixed-mapped
                     other question ((i + shuffle_step) mod N), fetched with
                     that question's text as the query -- deterministic mapping,
                     real counts/format/status, never fabricated errors.
      - no-evidence: every successful search returns the fixed neutral envelope
                     (fixed cached content, no retriever call, no
                     invalid_query/api_error); a genuinely invalid query (None)
                     keeps its original invalid path.

    Each env's SearchToolGroup carries _p3_question_index (stamped by the
    returned callable before use) so the mapping keys on the QUESTION index,
    not on the model-generated query text.
    """
    if condition == "real":

        def noop_stamp(envs, batch_records):  # type: ignore[return-value]
            return {}

        noop_stamp.counters = {}  # type: ignore[attr-defined]
        return noop_stamp

    import json as _json

    from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.core import tool  # noqa: E402
    from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.search import (  # noqa: E402
        SearchToolGroup,
        _passages2string,
        call_search_api,
    )

    n_questions = len(question_texts)
    counters = {
        "shuffled_served": 0,
        "shuffled_fallback_to_real": 0,
        "real_failure_kept": 0,
        "no_evidence_served": 0,
    }
    # SearchToolGroup.search is a @tool DESCRIPTOR (tools/core.py): instances
    # register tools at __init__ time by scanning class attributes for `tool`
    # instances, and execute_tool() dispatches through the registry. The patch
    # must therefore (a) keep the original UNDECORATED function as the real
    # retrieval path, and (b) re-wrap the replacement as a `tool` descriptor,
    # or every new env's _register_tools() misses "search" and every search
    # step dies with "Tool 'search' not found".
    orig_search = SearchToolGroup.search.func

    def patched_search(self, query):
        qidx = getattr(self, "_p3_question_index", None)
        if query is None or qidx is None:
            return orig_search(self, query)
        qidx = int(qidx)
        if condition == "shuffled":
            # 1) real retrieval of the model's own query (real status recorded)
            real_result = orig_search(self, query)
            if self.last_call_metadata.get("status") != "success":
                counters["real_failure_kept"] += 1
                return real_result  # never remap errors/empty/no-results
            # 2) fixed mapping: real docs of the other question (its text as query)
            other_idx = (qidx + shuffle_step) % n_questions
            api_response, err = call_search_api(
                self.search_url,
                question_texts[other_idx],
                topk=self.topk,
                timeout=self.timeout,
                log_requests=False,
                session=self.session,
            )
            if err is not None or api_response is None:
                counters["shuffled_fallback_to_real"] += 1
                return real_result  # honest fallback: real docs for THIS question
            raw = api_response.get("result", [])
            pretty = [_passages2string(r) for r in raw]
            final_result = "\n---\n".join(pretty)
            total = sum(len(r) if isinstance(r, list) else 1 for r in raw)
            doc_ids = [
                str(item.get("document", {}).get("id"))
                for r in raw
                if isinstance(r, list)
                for item in r
                if item.get("document", {}).get("id") is not None
            ]
            # keep "query" = the model's actual query; evidence fields = the
            # other question's REAL retrieval, including its real outcome
            # (genuinely empty result -> faithful no_results envelope, same
            # as the original search's no_results branch)
            if not raw:
                self.last_call_metadata.update(
                    {
                        "api_response": api_response,
                        "status": "no_results",
                        "total_results": 0,
                        "document_ids": [],
                        "formatted_result": None,
                    }
                )
                counters["shuffled_served"] += 1
                return _json.dumps({"result": "No search results found."})
            self.last_call_metadata.update(
                {
                    "api_response": api_response,
                    "status": "success",
                    "total_results": total,
                    "document_ids": doc_ids,
                    "formatted_result": final_result,
                }
            )
            counters["shuffled_served"] += 1
            return _json.dumps({"result": final_result})
        # no-evidence: fixed neutral cache, legal success envelope. Build a
        # fresh metadata dict: on a fresh env's FIRST search last_call_metadata
        # is None (vendor __init__), so update() on it would crash.
        neutral = "\n---\n".join(
            f"Doc {k + 1}: {NO_EVIDENCE_TEXT}" for k in range(no_evidence_docs)
        )
        self.last_call_metadata = {
            "query": query,
            "api_request_error": None,
            "api_response": None,
            "status": "success",
            "total_results": no_evidence_docs,
            "document_ids": [f"noev-{k}" for k in range(no_evidence_docs)],
            "formatted_result": neutral,
        }
        counters["no_evidence_served"] += 1
        return _json.dumps({"result": neutral})

    # the tool descriptor registers under func.__name__ -- it must stay
    # "search" or execute_tool("search", ...) misses the registry
    patched_search.__name__ = "search"
    SearchToolGroup.search = tool(patched_search)

    def stamp(envs, batch_records):
        stamped = {}
        for local_index, env in enumerate(envs.envs):
            tool_group = getattr(env, "tool_group", None)
            if tool_group is not None:
                qidx = batch_records[local_index].get("question_id")
                tool_group._p3_question_index = qidx
                stamped[local_index] = qidx
        return stamped

    stamp.counters = counters  # type: ignore[attr-defined]
    return stamp


def step_search_succeeded(step: dict[str, Any]) -> bool:
    observation = step.get("observation") or ""
    return "<information>" in observation and bool(step.get("information_returned"))


def step_retrieval_status(step: dict[str, Any]) -> str | None:
    """Typed retrieval status of a genuine search step (v2-0001 metadata)."""
    if step_search_query(step) is None:
        return None
    info = step.get("info") or {}
    retrieval = info.get("retrieval") or {}
    return retrieval.get("status")


def v2_search_behavior(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """v2 search-behaviour stats over the frozen v2 semantics.

    Uses ONLY the real retrieval metadata (info.retrieval: status/document_ids)
    plus the actual returned document text (the observation anchor). Evidence
    hits are checked against the returned doc bodies only; queries / errors /
    model outputs never count as evidence. Redundancy uses the frozen v2 rule:
    first search never redundant; duplicate normalized query or no new document
    ID (content-hash fallback) is redundant.
    """
    total_searches = 0
    status_counts: dict[str, int] = {}
    episodes_with_search = 0
    episodes_multi_hop = 0
    search_rounds: list[int] = []
    new_doc_increments: list[int] = []
    redundant_search_count = 0
    evidence_hit_search_count = 0
    search_after_correct_with_evidence = 0
    n_correct = 0
    for episode in episodes:
        episode_answers = episode["answers"]
        gt_aliases = valid_aliases(episode_answers)
        prior_queries: set[str] = set()
        prior_doc_ids: set[str] = set()
        prior_hashes: set[str] = set()
        episode_searches = 0
        episode_new_docs = 0
        episode_redundant = 0
        episode_evidence = 0
        first_search = True
        for step in episode["steps"]:
            query = step_search_query(step)
            if query is None:
                continue
            info = step.get("info") or {}
            retrieval = info.get("retrieval") or {}
            status = retrieval.get("status")
            doc_ids = retrieval.get("document_ids") or []
            observation = step.get("observation") or ""
            total_searches += 1
            episode_searches += 1
            status_counts[status or "unknown"] = status_counts.get(status or "unknown", 0) + 1
            # v2 TRUE redundancy (valid searches only; invalid pays invalid only)
            valid = status in {"success", "no_results"}
            redundant = False
            if valid:
                nq = norm_query(query)
                if first_search:
                    redundant = False
                elif nq and nq in prior_queries:
                    redundant = True
                elif doc_ids:
                    redundant = not any(d not in prior_doc_ids for d in doc_ids)
                else:
                    redundant = is_true_redundant(
                        query=query, status=status, doc_ids=None, doc_text=observation,
                        prior_queries=prior_queries, prior_doc_ids=prior_doc_ids,
                        prior_content_hashes=prior_hashes, is_first_search=first_search,
                    )
                if doc_ids:
                    new_ids = [d for d in doc_ids if d not in prior_doc_ids]
                    episode_new_docs += len(new_ids)
                    prior_doc_ids.update(str(d) for d in doc_ids)
                elif observation:
                    prior_hashes.add(hashlib.sha256(observation.encode()).hexdigest())
            if nq:
                prior_queries.add(nq)
            if redundant:
                episode_redundant += 1
                redundant_search_count += 1
            if valid and evidence_hit_in_docs(observation, gt_aliases):
                episode_evidence += 1
                evidence_hit_search_count += 1
            first_search = False
        if episode_searches > 0:
            episodes_with_search += 1
            search_rounds.append(episode_searches)
            new_doc_increments.append(episode_new_docs)
            if episode_searches >= 2:
                episodes_multi_hop += 1
        if episode["reward"] >= 1.0:
            n_correct += 1
            if episode_evidence >= 1:
                search_after_correct_with_evidence += 1
    n_episodes = max(len(episodes), 1)
    return {
        "total_search_calls": total_searches,
        "search_status_counts": {k: status_counts[k] for k in sorted(status_counts)},
        "valid_search_calls": sum(status_counts.get(s, 0) for s in ("success", "no_results")),
        "invalid_search_calls": total_searches - sum(status_counts.get(s, 0) for s in ("success", "no_results")),
        "invalid_search_rate": (total_searches - sum(status_counts.get(s, 0) for s in ("success", "no_results"))) / max(total_searches, 1),
        "episodes_with_search": episodes_with_search,
        "episodes_with_search_rate": episodes_with_search / n_episodes,
        "multi_hop_episodes": episodes_multi_hop,
        "multi_hop_ratio": episodes_multi_hop / max(episodes_with_search, 1),
        "search_rounds_per_episode_mean": (sum(search_rounds) / len(search_rounds)) if search_rounds else None,
        "search_rounds_distribution": {
            str(n): search_rounds.count(n) for n in sorted(set(search_rounds))
        },
        "new_document_increment_mean": (sum(new_doc_increments) / len(new_doc_increments)) if new_doc_increments else None,
        "true_redundant_searches": redundant_search_count,
        "true_redundant_rate": redundant_search_count / max(total_searches, 1),
        "evidence_hit_searches": evidence_hit_search_count,
        "evidence_hit_rate": evidence_hit_search_count / max(total_searches, 1),
        "correct_with_evidence_searches": search_after_correct_with_evidence,
        "correct_total": n_correct,
    }


def aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    per_source: dict[str, dict[str, Any]] = {}
    total_steps = 0
    executed_searches = 0
    successful_searches = 0
    searched_episodes = 0
    searched_correct = 0
    searched_answered = 0
    no_search_episodes = 0
    no_search_correct = 0

    for episode in episodes:
        source = episode["source"]
        bucket = per_source.setdefault(source, {"n": 0, "em": 0, "success": 0, "answer_compliance": 0})
        bucket["n"] += 1
        bucket["em"] += 1 if episode["reward"] >= 1.0 else 0
        bucket["success"] += 1 if episode["won"] else 0
        bucket["answer_compliance"] += 1 if episode["offline"]["has_answer"] else 0

        episode_searches = 0
        for step in episode["steps"]:
            total_steps += 1
            if step_search_query(step) is not None:
                executed_searches += 1
                episode_searches += 1
                if step_search_succeeded(step):
                    successful_searches += 1
        if episode_searches > 0:
            searched_episodes += 1
            searched_correct += 1 if episode["reward"] >= 1.0 else 0
            searched_answered += 1 if episode["offline"]["has_answer"] else 0
        else:
            no_search_episodes += 1
            no_search_correct += 1 if episode["reward"] >= 1.0 else 0

    # Per-question aggregation (rollout-level, e.g. behaviour diagnosis with
    # num_rollouts=5): a question counts as searched/answered/correct if ANY of
    # its rollouts satisfies the condition.
    per_question: dict[int, dict[str, Any]] = {}
    for episode in episodes:
        qid = episode.get("question_id", 0)
        bucket = per_question.setdefault(
            qid,
            {
                "n": 0,
                "searched": 0,
                "answered": 0,
                "correct": 0,
                "search_success": 0,
            },
        )
        bucket["n"] += 1
        episode_searched = any(
            step_search_query(step) is not None for step in episode["steps"]
        )
        episode_answered = bool(episode["offline"]["has_answer"])
        episode_correct = episode["reward"] >= 1.0
        if episode_searched:
            bucket["searched"] += 1
            if any(step_search_succeeded(step) for step in episode["steps"]):
                bucket["search_success"] += 1
        if episode_answered:
            bucket["answered"] += 1
        if episode_correct:
            bucket["correct"] += 1
    n_questions = len(per_question)
    questions_searched = sum(1 for b in per_question.values() if b["searched"] > 0)
    questions_answered = sum(1 for b in per_question.values() if b["answered"] > 0)
    questions_correct = sum(1 for b in per_question.values() if b["correct"] > 0)

    def rates(bucket: dict[str, Any]) -> dict[str, float]:
        n = max(bucket["n"], 1)
        return {
            "em_rate": bucket["em"] / n,
            "success_rate": bucket["success"] / n,
            "answer_compliance_rate": bucket["answer_compliance"] / n,
        }

    return {
        "per_question": {
            "n_questions": n_questions,
            "questions_searched": questions_searched,
            "questions_searched_rate": questions_searched / max(n_questions, 1),
            "questions_answered": questions_answered,
            "questions_answered_rate": questions_answered / max(n_questions, 1),
            "questions_correct": questions_correct,
            "questions_correct_rate": questions_correct / max(n_questions, 1),
            "search_to_answer_question_level": sum(
                1 for b in per_question.values() if b["searched"] > 0 and b["answered"] > 0
            )
            / max(questions_searched, 1),
            "search_to_correct_question_level": sum(
                1 for b in per_question.values() if b["searched"] > 0 and b["correct"] > 0
            )
            / max(questions_searched, 1),
        },
        "overall": {
            "n": len(episodes),
            "em": sum(b["em"] for b in per_source.values()),
            "em_rate": sum(b["em"] for b in per_source.values()) / max(len(episodes), 1),
            "success": sum(b["success"] for b in per_source.values()),
            "answer_compliance": sum(b["answer_compliance"] for b in per_source.values()),
        },
        "per_source": {source: {**bucket, **rates(bucket)} for source, bucket in sorted(per_source.items())},
        "search": {
            "search_attempt_steps": executed_searches,
            "search_successful_steps": successful_searches,
            "search_success_rate": successful_searches / max(executed_searches, 1),
            "searched_episodes": searched_episodes,
            "no_search_episodes": no_search_episodes,
            "search_to_answer": searched_answered / max(searched_episodes, 1),
            "search_to_correct": searched_correct / max(searched_episodes, 1),
            "no_search_to_correct": no_search_correct / max(no_search_episodes, 1),
        },
        "action_stats": {
            "total_steps": total_steps,
            "steps_per_episode": total_steps / max(len(episodes), 1),
        },
        "search_behavior_v2": v2_search_behavior(episodes),
        "offline_rescore": {
            "matches": sum(
                1 for e in episodes if e["offline"]["score"] == e["reward"]
                or (e["offline"]["score"] >= 1.0 and e["reward"] >= 1.0)
            ),
            "mismatches": sum(
                1 for e in episodes if not (e["offline"]["score"] == e["reward"]
                or (e["offline"]["score"] >= 1.0 and e["reward"] >= 1.0))
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Round-2 prompt check (smoke-16 gate): the round-2 prompt must contain the
# original question, the search query and the returned information.
# --------------------------------------------------------------------------- #

def check_round2_prompts(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for episode in episodes:
        if len(episode["steps"]) < 2:
            continue
        first = episode["steps"][0]
        query = step_search_query(first)
        if query is None:
            continue
        second = episode["steps"][1]
        prompt = second["prompt"]
        norm_prompt = normalize_question(prompt)
        # NOTE: SEARCH_TEMPLATE's instruction text contains the literal
        # "<information> </information>" placeholder, so a naive substring test
        # would pass even when no result ever entered the history. A real
        # returned block is "<information><content></information>" with
        # non-whitespace content, which the negative lookahead distinguishes.
        result = {
            "question": episode["question"],
            "step1_query": query,
            "step1_observation": first["observation"],
            "contains_question": normalize_question(episode["question"]) in norm_prompt,
            "contains_query": stepsearch_prompt_contains_query(prompt, query),
            "contains_information": re.search(
                r"<information>(?!\s*</information>)", prompt, re.IGNORECASE | re.DOTALL
            ) is not None,
        }
        checked.append(result)
        if not (result["contains_question"] and result["contains_query"] and result["contains_information"]):
            failures.append(result)
    return {
        "checked_episodes": len(checked),
        "passed_episodes": len(checked) - len(failures),
        "passed": len(checked) >= 1 and len(failures) == 0,
        "failures": failures,
        "checks": checked,
    }


# --------------------------------------------------------------------------- #
# Generation (vLLM native, same engine config as the training rollout)
# --------------------------------------------------------------------------- #

def build_engine(tokenizer, args: argparse.Namespace) -> vllm.LLM:
    engine_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "dtype": VLLM_DTYPE,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "enforce_eager": True,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "seed": args.seed,
        "trust_remote_code": False,
    }
    return vllm.LLM(**engine_kwargs)


def generate_actions(
    llm: vllm.LLM, tokenizer, prompts: list[str], seeds: list[int], args: argparse.Namespace
) -> list[str]:
    """Generate one action per prompt. In diagnose mode (temperature>0) each
    rollout carries a FIXED per-rollout seed (args.seed + rollout index), so the
    whole behaviour-diagnosis run is reproducible. In main mode (temperature=0)
    the seed is irrelevant (greedy)."""
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        chats,
        return_tensors=None,  # ragged lists: the vLLM engine batches internally
        padding=False,
        truncation=True,
        max_length=args.max_input_tokens,
    )
    prompt_token_ids = inputs["input_ids"]
    sampling_params = [
        vllm.SamplingParams(
            temperature=args.temperature,
            top_p=1.0,
            top_k=-1,
            max_tokens=args.max_new_tokens,
            ignore_eos=False,
            seed=seed,
        )
        for seed in seeds
    ]
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    return [output.outputs[0].text for output in outputs]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=Path, required=True, help="v2 tree (vendor/verl-agent-v2): pristine 20bd331b + patches/v2/v2-0001..0007, must be FIRST on PYTHONPATH")
    parser.add_argument("--pristine-dir", type=Path, required=True, help="pristine reference tree (vendor/upstream-20bd331b) for the protocol byte-equality gate")
    parser.add_argument("--model", type=Path, required=True, help="full HF model directory (official merged GRPO checkpoint)")
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer directory for input rendering (defaults to --model)")
    parser.add_argument("--data-files", type=Path, required=True, help="held-out parquet (smoke test or official confirm set)")
    parser.add_argument("--manifest", type=Path, default=None, help="manifest.json holding the data file SHA256")
    parser.add_argument("--manifest-key", default="heldout", help="outputs key of the data file in the manifest")
    parser.add_argument("--leakage-reference", type=Path, required=True, help="training parquet for the leakage gate")
    parser.add_argument("--search-url", default="http://127.0.0.1:18080/retrieve")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-envs-per-batch",
        type=int,
        default=24,
        help="max concurrent eval envs per chunk (CPU retriever capacity; pure concurrency control)",
    )
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="decoding temperature: 0.0 = greedy main evaluation; >0 = behaviour diagnosis",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1,
        help="rollouts per question (main mode: 1; behaviour diagnosis: 5). "
        "Each rollout i uses seed = --seed + i for reproducibility.",
    )
    parser.add_argument(
        "--retrieval-condition",
        choices=RETRIEVAL_CONDITIONS,
        default="real",
        help="evidence condition: real (baseline); shuffled (real retrieval "
        "first, then evidence replaced by the fixed-mapped other question's "
        "real docs via (i + --shuffle-step) mod N); no-evidence (legal "
        "success envelope with fixed neutral content). Only in main mode "
        "(greedy, num_rollouts=1).",
    )
    parser.add_argument(
        "--shuffle-step",
        type=int,
        default=17,
        help="fixed mapping offset for the shuffled condition: evidence of "
        "question i is replaced by question (i + shuffle_step) mod N.",
    )
    parser.add_argument(
        "--no-evidence-docs",
        type=int,
        default=3,
        help="number of neutral docs returned by the no-evidence envelope.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stepsearch_protocol = os.environ.get("PROJECT3_EXTERNAL_STEPSEARCH_PROTOCOL") == "1"
    if args.tokenizer is None:
        args.tokenizer = args.model
    v2_info = verify_v2_tree(args.v2_dir, args.pristine_dir)
    env_info = validate_managed_environment()
    engine_info = validate_vllm_engine_parity()
    run_dir = Path(env_info["run_dir"]).resolve()
    output_path = args.output or run_dir / "results.json"
    episodes_path = run_dir / "episodes.jsonl"

    # Gates (abort on failure, never warn-and-continue).
    leakage = leakage_check(load_eval_records(args.data_files), args.leakage_reference)
    health = retriever_health_check(args.search_url)
    data_hash = verify_data_hash(args.data_files, args.manifest, args.manifest_key)

    records = load_eval_records(args.data_files)
    if args.num_rollouts < 1:
        raise ValueError(f"--num-rollouts must be >= 1, got {args.num_rollouts}")
    main_mode = args.temperature == 0.0 and args.num_rollouts == 1
    if args.retrieval_condition != "real" and not main_mode:
        raise ValueError(
            f"--retrieval-condition={args.retrieval_condition} requires main mode "
            "(temperature=0.0, num_rollouts=1): counterfactual evidence conditions "
            "are only defined for the single greedy pass per question"
        )
    # Behaviour diagnosis (num_rollouts>1): replicate each question, tagging
    # the rollout index; each rollout i is seeded with --seed + i so the whole
    # run is reproducible. Main mode (greedy, 1 rollout): no replication.
    expanded: list[dict[str, Any]] = []
    for question_index, record in enumerate(records):
        for rollout_index in range(args.num_rollouts):
            expanded.append({**record, "question_id": question_index, "rollout_index": rollout_index})
    records = expanded
    # Counterfactual conditions key on the QUESTION index (0..N-1, file order,
    # identical to the real run: same loader, same data file -> same order).
    # The mapping is FIXED and pre-registered before any episode runs.
    question_texts = [record["question"] for record in records]
    stamp_question_index = install_retrieval_condition(
        args.retrieval_condition, args.shuffle_step, args.no_evidence_docs, question_texts
    )
    if args.retrieval_condition != "real":
        mapping = {
            qidx: (qidx + args.shuffle_step) % len(question_texts)
            for qidx in range(len(question_texts))
        }
        prereg = {
            "kind": "p3-counterfactual-preregistration",
            "condition": args.retrieval_condition,
            "shuffle_step": args.shuffle_step,
            "no_evidence_docs": args.no_evidence_docs,
            "n_questions": len(question_texts),
            "mapping_spec": "shuffled: evidence of question i replaced by the REAL "
            "docs of question (i + shuffle_step) mod N, fetched with that question's "
            "text as the query; real retrieval of the model's own query executes "
            "first and non-success statuses are kept verbatim",
            "mapping": mapping,
            "mapping_sha256": hashlib.sha256(
                json.dumps(mapping, sort_keys=True).encode()
            ).hexdigest(),
            "data_sha256": data_hash["sha256"],
            "model_path": str(args.model.resolve()),
            "seed": args.seed,
            "temperature": args.temperature,
            "max_steps": args.max_steps,
            "topk": args.topk,
            "created_before_episode_loop": True,
        }
        prereg_path = run_dir / "retrieval_condition_preregistration.json"
        prereg_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(prereg_path, prereg)
        print(f"preregistration={json.dumps({k: v for k, v in prereg.items() if k != 'mapping'}, ensure_ascii=False)}")
        print(f"preregistration_mapping_sha256={prereg['mapping_sha256']}")
        print(f"preregistration_file={prereg_path}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.monotonic()
    llm = build_engine(tokenizer, args)
    torch.cuda.reset_peak_memory_stats()

    episodes: list[dict[str, Any]] = [
        {
            "question": record["question"],
            "answers": record["answers"],
            "source": record["data_source"],
            "question_id": record["question_id"],
            "rollout_index": record["rollout_index"],
            "steps": [],
            "reward": 0.0,
            "done": False,
            "won": False,
        }
        for record in records
    ]
    done = np.zeros(len(records), dtype=bool)
    final_rewards = np.zeros(len(records), dtype=float)
    won = np.zeros(len(records), dtype=bool)

    try:
        # Chunked episode loop (CPU retriever capacity).
        for batch_start in range(0, len(records), args.max_envs_per_batch):
            batch_records = records[batch_start : batch_start + args.max_envs_per_batch]
            config = OmegaConf.create(
                {
                    "env": {
                        "max_steps": args.max_steps,
                        "history_length": args.history_length,
                        "search": {
                            "search_url": args.search_url,
                            "topk": args.topk,
                            "timeout": args.timeout,
                            "log_requests": True,
                        },
                    }
                }
            )
            env_config = config.env
            # Clean upstream: SearchEnvironmentManager (question + search
            # history) over SearchMultiProcessEnv with the upstream projection.
            envs = SearchMultiProcessEnv(
                seed=args.seed,
                env_num=len(batch_records),
                group_n=1,
                is_train=False,
                env_config=env_config,
            )
            stamped = stamp_question_index(envs, batch_records)
            if args.retrieval_condition != "real":
                assert len(stamped) == len(batch_records), (
                    f"question-index stamping incomplete: {len(stamped)}/{len(batch_records)}"
                )
            manager_class = StepSearchEnvironmentManager if stepsearch_protocol else SearchEnvironmentManager
            manager = manager_class(envs, partial(search_projection), config)
            kwargs = [
                {
                    "question": record["question"],
                    "ground_truth": {"target": record["answers"]},
                    "data_source": record["data_source"],
                }
                for record in batch_records
            ]
            observations, _ = manager.reset(kwargs)

            batch_done = np.zeros(len(batch_records), dtype=bool)
            batch_final = np.zeros(len(batch_records), dtype=float)
            batch_won = np.zeros(len(batch_records), dtype=bool)

            try:
                for step_index in range(args.max_steps):
                    active_before = ~batch_done
                    if not active_before.any():
                        break
                    generation_started = time.monotonic()
                    seeds = [
                        args.seed + batch_records[local_index]["rollout_index"]
                        for local_index in range(len(batch_records))
                    ]
                    raw_actions = generate_actions(llm, tokenizer, observations["text"], seeds, args)
                    if stepsearch_protocol:
                        # The public StepSearch generator applies this boundary
                        # before projection. Text sampled after </search> must
                        # not fabricate tool results in next-turn memory.
                        raw_actions = [truncate_stepsearch_response(action) for action in raw_actions]
                    projected_actions, valids = search_projection(raw_actions)
                    # Empty <search></search> is a degenerate action: upstream
                    # SearchToolGroup.search guards only None (not ""), so the
                    # empty query reaches the retriever and 422s against its
                    # min_length=1 schema. The error blob pollutes the context
                    # and kills the episode's search chain. Treat empty queries
                    # as invalid actions ("" -> env returns an empty observation
                    # with NO HTTP request, same semantics as the no-tags case).
                    projected_actions, valids = sanitize_empty_search_actions(projected_actions, valids)
                    if stepsearch_protocol:
                        next_observations, rewards, step_done, infos = manager.step_projected(
                            raw_actions, projected_actions, valids
                        )
                    else:
                        next_observations, rewards, step_done, infos = manager.step(projected_actions)
                    generation_seconds = time.monotonic() - generation_started
                    for local_index in range(len(batch_records)):
                        if not active_before[local_index]:
                            continue
                        global_index = batch_start + local_index
                        observation_text = (
                            next_observations["text"][local_index]
                            if local_index < len(next_observations["text"]) else ""
                        )
                        observation_anchor = (
                            next_observations["anchor"][local_index]
                            if local_index < len(next_observations["anchor"]) else ""
                        )
                        episodes[global_index]["steps"].append(
                            {
                                "step": step_index + 1,
                                "prompt": observations["text"][local_index],
                                "raw_action": raw_actions[local_index],
                                "projected_action": projected_actions[local_index],
                                "action_valid": int(valids[local_index]),
                                "observation": str(observation_anchor),
                                "prompt_next_round": str(observation_text),
                                "reward": float(rewards[local_index]),
                                "done": bool(step_done[local_index]),
                                "won": bool(infos[local_index].get("won", False)),
                                "tool_calling": bool(infos[local_index].get("tool_calling")),
                                "tool_input": infos[local_index].get("tool_input"),
                                "information_returned": "<information>" in str(observation_anchor),
                                "info": jsonable(infos[local_index]),
                                "batch_generation_seconds": generation_seconds,
                            }
                        )
                        batch_final[local_index] += float(rewards[local_index])
                        batch_won[local_index] = batch_won[local_index] or bool(infos[local_index].get("won", False))
                    batch_done |= np.asarray(step_done, dtype=bool)
                    observations = next_observations
            finally:
                envs.close()
            done[batch_start : batch_start + len(batch_records)] |= batch_done
            final_rewards[batch_start : batch_start + len(batch_records)] += batch_final
            won[batch_start : batch_start + len(batch_records)] |= batch_won
    finally:
        try:
            del llm
        except Exception:
            pass
        import gc

        gc.collect()

    elapsed = time.monotonic() - started

    for index, episode in enumerate(episodes):
        episode["reward"] = float(final_rewards[index])
        episode["done"] = bool(done[index])
        episode["won"] = bool(won[index]) or episode["reward"] >= 1.0
        episode["offline"] = offline_rescore(episode["steps"], episode["answers"])

    prompt_check = check_round2_prompts(episodes)
    metrics = aggregate_metrics(episodes)
    result = {
        "schema_version": 1,
        "kind": "p3-stepsearch-external-evaluation" if stepsearch_protocol else "p3-search-aware-clean-v2-evaluation",
        "line": (
            "external StepSearch-3B policy on the P3 v2 search environment"
            if stepsearch_protocol
            else "v2 line: pristine 20bd331b + patches/v2/v2-0001..0007 (clean protocol restored)"
        ),
        "runtime_script_sha256": sha256_file(Path(__file__).resolve()),
        "training": False,
        "training_operations": "none by construction: no optimizer, no scheduler, no backward, no Ray",
        # truthful label: greedy main eval vs sampling diagnosis (the
        # authoritative decoding parameters live in the `parameters` block)
        "decoding_backend": "vllm-native-greedy" if args.temperature == 0.0 else "vllm-native-sampling",
        "engine": {
            **engine_info,
            "dtype": VLLM_DTYPE,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "enforce_eager": True,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "enable_lora": False,
        },
        "semantics": {
            "line": (
                f"StepSearch evaluation-only prompt/history adapter from public commit {STEPSEARCH_SOURCE_COMMIT}; no reward or training changes"
                if stepsearch_protocol
                else "v2 clean protocol: prompt/projection/terminal reward byte-identical to pristine 20bd331b (v1's 0004/0005 NOT applied)"
            ),
            "env": "SearchMultiProcessEnv + SearchEnvironmentManager (question + search history via SearchMemory); env.search.search_aware_step_reward=false in the eval config -> the env's clean path executes (the v2 shaping branch never runs during evaluation)",
            "projection": "upstream search_projection (first <search> else <answer> else empty; both/duplicate tags -> invalid)",
            "prompt": (
                "official StepSearch plan/search/information/observation prefix; raw model plan/observation/search text plus real tool information retained in subsequent context"
                if stepsearch_protocol
                else "single-layer official Search prompt (SEARCH_TEMPLATE_NO_HIS round 1 / SEARCH_TEMPLATE round 2+ with memory_context)"
            ),
            "action_parse": "upstream SearchEnv: re.search(r'<search>(.*?)</search>', re.DOTALL); no query -> empty tool output -> no information",
            "termination": "done when '<answer>' and '</answer>' both appear in action, or turns >= max_steps",
            "terminal_reward": "upstream skyrl compute_score(chat_history, ground_truth) format_score=0.0; EM = reward >= 1.0",
        },
        "v2_tree": v2_info,
        "retriever_note": REAL_INDEX_NOTE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": env_info["run_id"],
        "model_path": str(args.model.resolve()),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "data_files": {
            "path": str(args.data_files.resolve()),
            "sha256": data_hash["sha256"],
            "hash_verified_against_manifest": data_hash["checked"],
        },
        "leakage": leakage,
        "retriever_health": health,
        "physical_gpu_ids": env_info["cuda_visible_devices"],
        "logical_cuda_device": torch.cuda.get_device_name(0),
        "parameters": {
            "seed": args.seed,
            "temperature": args.temperature,
            "num_rollouts": args.num_rollouts,
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "topk": args.topk,
            "timeout": args.timeout,
            "max_envs_per_batch": args.max_envs_per_batch,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "main_mode": main_mode,
            "prompt_protocol": "stepsearch-public" if stepsearch_protocol else "search-r1-clean",
            "retrieval_condition": args.retrieval_condition,
            "shuffle_step": args.shuffle_step,
            "no_evidence_docs": args.no_evidence_docs,
        },
        "retrieval_condition": {
            "condition": args.retrieval_condition,
            "shuffle_step": args.shuffle_step,
            "no_evidence_docs": args.no_evidence_docs,
            "counters": stamp_question_index.counters,
            "preregistration": (
                {
                    "file": str((run_dir / "retrieval_condition_preregistration.json").resolve()),
                    "mapping_sha256": prereg["mapping_sha256"] if args.retrieval_condition != "real" else None,
                }
                if args.retrieval_condition != "real"
                else None
            ),
            "note": (
                "vendor tree byte-identical (v2-tree gate); this patch is runtime-only, "
                "changes EVIDENCE CONTENT only, never prompt/projection/decoding/step budget; "
                "real condition = no patch"
            ),
        },
        "metrics": metrics,
        "prompt_check": prompt_check,
        "elapsed_seconds": elapsed,
        # torch-allocator views ONLY (may undercount vLLM-managed blocks);
        # the per-GPU PHYSICAL nvidia-smi peaks are recorded by the wrapper
        # (run_p3_eval_v2.sh) in peak_memory_nvidia_smi.json -- the two numbers
        # must never be conflated.
        "peak_gpu_memory_allocated_bytes_torch_view": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes_torch_view": int(torch.cuda.max_memory_reserved()),
        "outputs": {
            "episodes": str(episodes_path.resolve()),
        },
    }
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(episodes_path, episodes)
    atomic_write_json(output_path, result)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"prompt_check={json.dumps({k: v for k, v in prompt_check.items() if k != 'checks' and k != 'failures'}, ensure_ascii=False)}")
    if prompt_check["failures"]:
        print("prompt_check failures:")
        for failure in prompt_check["failures"]:
            print(json.dumps(failure, ensure_ascii=False))
    print(f"output={output_path}")
    print(f"episodes={episodes_path}")
    print(f"main_mode={main_mode}")
    # Fail-closed gate applies ONLY to main mode (greedy, 1 rollout): smoke-16
    # must PASS the round-2 prompt check before confirm-256. Behaviour
    # diagnosis (temperature>0 / num_rollouts>1) is a strategy-support probe:
    # it reports the same metrics but never blocks.
    if main_mode and not prompt_check["passed"]:
        print("SMOKE GATE FAILED: round-2 prompt check not satisfied (checked_episodes >= 1 and all three components present)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
