#!/usr/bin/env python3
"""Clean-upstream (20bd331b, NO patches 0001-0008) evaluation harness.

Evaluates the official 3B GRPO checkpoint under the UNMODIFIED upstream
verl-agent semantics:

- SearchEnvironmentManager (upstream env_manager.py) manages the question and
  the search history (SearchMemory: "<search>...</search>" actions +
  "<information>...</information>" observations, formatted by SEARCH_TEMPLATE
  with task_description / step_count / memory_context).
- upstream search_projection (env_package/search/projection.py): first complete
  <search>...</search> block, else first <answer>...</answer>, else empty
  (valids=0); both-tags / duplicated-tags actions are marked invalid.
- single-layer official Search prompt: SEARCH_TEMPLATE_NO_HIS (round 1) and
  SEARCH_TEMPLATE (round 2+, with history) from prompts/search.py.
- max_steps / history_length / topk / timeout are passed as CLI arguments
  (the clean-line protocol is max_steps=4, history_length=4, topk=3).
- Terminal reward: upstream skyrl compute_score(chat_history, ground_truth)
  with format_score=0.0 -> EM = reward >= 1.0.

Smoke-16 mode runs a hard round-2 prompt check: every episode that searched at
step 1 and continued to step 2 must show a round-2 prompt containing (a) the
original question, (b) the search query, (c) the returned information. If no
search happened or any such prompt is missing a component, the script exits 2
(fail-closed; the confirm-256 phase must not start).

confirm-256 mode reports: EM, successful search rate, search->answer and
search->correct (plus no-search contrast).

By construction this script creates no optimizer, no scheduler, never calls
backward, and does not import Ray. The clean tree (vendor/upstream-20bd331b)
must be the FIRST entry on PYTHONPATH; --upstream-dir is verified against the
imported agent_system location, and the imported package must not contain any
patch marker (search_aware_step_reward), otherwise the script aborts.
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

EXPECTED_RETRIEVER_VECTORS = 21_015_324
REAL_INDEX_NOTE = "real Wiki-18 IndexFlatIP (21,015,324 vectors); ground-truth-derived fixture is prohibited for evaluation"
PATCH_MARKER = "search_aware_step_reward"  # introduced by patches 0007/0008; must be ABSENT

# Engine parity with the training rollout path (V0 engine).
VLLM_DTYPE = "bfloat16"
VLLM_GPU_MEMORY_UTILIZATION = 0.6
VLLM_MAX_MODEL_LEN = 3328  # max_input_tokens(3072) + max_new_tokens(256)


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

def verify_clean_upstream(upstream_dir: Path) -> dict[str, Any]:
    """The imported agent_system must live inside --upstream-dir and carry no
    patch markers. Any deviation aborts (clean-line contract: NO 0001-0008).

    agent_system is a NAMESPACE package (no __init__.py in upstream), so the
    package location is read from __path__, not __file__."""
    import agent_system
    package_paths = [Path(p).resolve() for p in getattr(agent_system, "__path__", [])]
    root = upstream_dir.resolve()
    if not package_paths or not any(str(p).startswith(str(root)) for p in package_paths):
        raise RuntimeError(
            f"imported agent_system does not resolve under the clean tree {root}: {package_paths}"
        )
    markers = []
    for path in sorted((root / "agent_system").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        if PATCH_MARKER in text:
            markers.append(str(path))
    if markers:
        raise RuntimeError(
            f"clean tree contains patch marker '{PATCH_MARKER}' in {len(markers)} file(s): "
            f"{markers[:5]}"
        )
    return {
        "upstream_dir": str(root),
        "agent_system_module": str(package_paths[0]),
        "patch_markers": 0,
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


def step_search_succeeded(step: dict[str, Any]) -> bool:
    observation = step.get("observation") or ""
    return "<information>" in observation and bool(step.get("information_returned"))


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

    def rates(bucket: dict[str, Any]) -> dict[str, float]:
        n = max(bucket["n"], 1)
        return {
            "em_rate": bucket["em"] / n,
            "success_rate": bucket["success"] / n,
            "answer_compliance_rate": bucket["answer_compliance"] / n,
        }

    return {
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
            "contains_query": re.search(
                re.escape(f"<search>{query}</search>"), prompt, re.IGNORECASE | re.DOTALL
            ) is not None,
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


def generate_actions(llm: vllm.LLM, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
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
    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        ignore_eos=False,
    )
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
    parser.add_argument("--upstream-dir", type=Path, required=True, help="pristine upstream 20bd331b tree (vendor/upstream-20bd331b), must be FIRST on PYTHONPATH")
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
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    clean_info = verify_clean_upstream(args.upstream_dir)
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
            manager = SearchEnvironmentManager(envs, partial(search_projection), config)
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
                    raw_actions = generate_actions(llm, tokenizer, observations["text"], args)
                    projected_actions, valids = search_projection(raw_actions)
                    next_observations, rewards, step_done, infos = manager.step(raw_actions)
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
        "kind": "p3-clean-upstream-evaluation",
        "line": "clean-upstream-20bd331b (no patches 0001-0008)",
        "runtime_script_sha256": sha256_file(Path(__file__).resolve()),
        "training": False,
        "training_operations": "none by construction: no optimizer, no scheduler, no backward, no Ray",
        "decoding_backend": "vllm-native-greedy",
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
            "line": "clean upstream 20bd331b, patches 0001-0008 NOT applied",
            "env": "SearchMultiProcessEnv + SearchEnvironmentManager (question + search history via SearchMemory)",
            "projection": "upstream search_projection (first <search> else <answer> else empty; both/duplicate tags -> invalid)",
            "prompt": "single-layer official Search prompt (SEARCH_TEMPLATE_NO_HIS round 1 / SEARCH_TEMPLATE round 2+ with memory_context)",
            "action_parse": "upstream SearchEnv: re.search(r'<search>(.*?)</search>', re.DOTALL); no query -> empty tool output -> no information",
            "termination": "done when '<answer>' and '</answer>' both appear in action, or turns >= max_steps",
            "terminal_reward": "upstream skyrl compute_score(chat_history, ground_truth) format_score=0.0; EM = reward >= 1.0",
        },
        "clean_upstream": clean_info,
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
            "max_steps": args.max_steps,
            "history_length": args.history_length,
            "topk": args.topk,
            "timeout": args.timeout,
            "max_envs_per_batch": args.max_envs_per_batch,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
        },
        "metrics": metrics,
        "prompt_check": prompt_check,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
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
    # Fail-closed: smoke-16 must PASS the round-2 prompt check before confirm-256.
    if not prompt_check["passed"]:
        print("SMOKE GATE FAILED: round-2 prompt check not satisfied (checked_episodes >= 1 and all three components present)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
