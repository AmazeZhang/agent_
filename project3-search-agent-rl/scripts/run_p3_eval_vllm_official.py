#!/usr/bin/env python3
"""Pure-evaluation harness for the P3 OFFICIAL-LOOSE line (vLLM-native greedy).

Official Search-R1 semantics line, per docs/P3_EXPERIMENT_LINES_2026-08-15.md
section 1 (left column). Independent script on purpose: the strict-fork line
(run_p3_eval_vllm.py) must never grow switches, and vice versa.

Semantics (implemented by the vendored skyrl SearchEnv, which IS the official
verl-integrated environment — the only deviation is patch 0004 aligning the
reward format_score to the Search-R1 paper's 0.1):

- The model's RAW action string goes directly into the environment. There is
  NO projection layer and NO validity flag.
- Action parsing: SearchEnv._parse_action = re.search(r"<search>(.*?)</search>",
  re.DOTALL) — loose: only the first <search> block is used; anything else
  (no query, malformed tags, uppercase tags) falls through to a tool call
  whose query is None, which raises -> the observation text is the error
  message and the model simply retries. No penalty is ever applied.
- Termination: done when "<answer>" and "</answer>" BOTH appear in the action
  (no well-formedness requirement) or turns >= max_steps.
- Terminal reward: skyrl compute_score(chat_history, ground_truth,
  format_score=0.1): EM hit 1.0 / well-formed-but-wrong 0.1 / nothing 0.0.
- EM (primary metric) = env reward >= 1.0, identical to the strict line.

Engine: identical to the strict line and to the training rollout path
(V0 engine, VLLM_USE_V1=0, bfloat16, gpu_memory_utilization 0.6,
enforce_eager, max_model_len 2304, temperature 0 greedy).

Input side: the tokenizer is taken from --tokenizer (defaults to --model).
The official-line wrapper pins --tokenizer to the Qwen2.5-3B BASE tokenizer
so that Base and the official GRPO checkpoint receive byte-identical input
token ids — the official checkpoint's own tokenizer_config embeds a
tools-flavoured chat_template that would otherwise change the rendered input.
Only the weights differ; that is the single variable under test.

Gates are identical to the strict line: managed environment (physical GPU1,
never GPU0), VLLM_USE_V1=0, real Wiki-18 retriever health (21,015,324
vectors), data SHA against manifest, leakage=0 against the training split,
chunked episode loop (max_envs_per_batch <= 32) to protect the CPU retriever.

By construction this script creates no optimizer, no scheduler, never calls
backward, and does not import Ray.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
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

from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv  # noqa: E402
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (  # noqa: E402
    compute_score as skyrl_compute_score,
    extract_solution,
)

EXPECTED_RETRIEVER_VECTORS = 21_015_324
REAL_INDEX_NOTE = "real Wiki-18 IndexFlatIP (21,015,324 vectors); ground-truth-derived fixture is prohibited for evaluation"

# Engine parity with training rollout (run_p3_grpo_fix_exp.sh overrides).
VLLM_DTYPE = "bfloat16"
VLLM_GPU_MEMORY_UTILIZATION = 0.6
VLLM_MAX_MODEL_LEN = 2304  # max_input_tokens(2048) + max_new_tokens(256)


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
# Gates (identical to the strict line; abort on failure, never warn-and-continue)
# --------------------------------------------------------------------------- #

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
# Action / answer analysis (official-loose: no projection, no validity flag)
# --------------------------------------------------------------------------- #

def actions_text(episode_steps: list[dict[str, Any]]) -> str:
    return "\n".join(step["raw_action"] for step in episode_steps)


def offline_rescore(episode_steps: list[dict[str, Any]], answers: list[str]) -> dict[str, Any]:
    """Audit-only EM re-score over the concatenated model actions (format 0.0,
    i.e. the EM-compatible component; env reward may additionally include the
    paper's 0.1 format score)."""
    solution = actions_text(episode_steps)
    final_answer = extract_solution(solution)
    if final_answer is None:
        return {"final_answer": None, "score": 0.0, "has_answer": False}
    score = skyrl_compute_score(
        solution, {"target": answers}, method="strict", format_score=0.0, score=1.0
    )
    return {"final_answer": final_answer, "score": float(score), "has_answer": True}


def aggregate_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {"n": len(episodes)}
    per_source: dict[str, dict[str, Any]] = {}
    total_steps = 0
    error_observation_steps = 0  # loose line's analogue of 'invalid': env returned an error observation
    executed_searches = 0
    retrieval_statuses: dict[str, int] = {}
    format_scored_episodes = 0  # env reward == 0.1 exactly (well-formed but wrong)
    rescore_matches = 0
    rescore_mismatches = 0

    for episode in episodes:
        source = episode["source"]
        bucket = per_source.setdefault(source, {"n": 0, "em": 0, "success": 0, "answer_compliance": 0})
        bucket["n"] += 1
        bucket["em"] += 1 if episode["reward"] >= 1.0 else 0
        bucket["success"] += 1 if episode["won"] else 0
        bucket["answer_compliance"] += 1 if episode["offline"]["has_answer"] else 0
        if episode["reward"] == 0.1:
            format_scored_episodes += 1
        for step in episode["steps"]:
            total_steps += 1
            info_retrieval = step.get("info", {}).get("retrieval", {}) or {}
            status = info_retrieval.get("status")
            if status is not None:
                retrieval_statuses[status] = retrieval_statuses.get(status, 0) + 1
            if step.get("error_observation"):
                error_observation_steps += 1
            if step.get("executed_search"):
                executed_searches += 1
        if episode["offline"]["score"] == episode["reward"] or (
            episode["offline"]["score"] >= 1.0 and episode["reward"] >= 1.0
        ):
            rescore_matches += 1
        else:
            rescore_mismatches += 1

    def rates(bucket: dict[str, Any]) -> dict[str, float]:
        n = max(bucket["n"], 1)
        return {
            "em_rate": bucket["em"] / n,
            "success_rate": bucket["success"] / n,
            "answer_compliance_rate": bucket["answer_compliance"] / n,
        }

    executed_searches_total = max(executed_searches, 1)
    return {
        "overall": {
            "n": overall["n"],
            "em": sum(b["em"] for b in per_source.values()),
            "success": sum(b["success"] for b in per_source.values()),
            **rates(
                {
                    "n": overall["n"],
                    "em": sum(b["em"] for b in per_source.values()),
                    "success": sum(b["success"] for b in per_source.values()),
                    "answer_compliance": sum(b["answer_compliance"] for b in per_source.values()),
                }
            ),
        },
        "per_source": {source: {**bucket, **rates(bucket)} for source, bucket in sorted(per_source.items())},
        "action_stats": {
            "total_steps": total_steps,
            "error_observation_steps": error_observation_steps,
            "error_observation_ratio": error_observation_steps / max(total_steps, 1),
            "format_scored_episodes": format_scored_episodes,
        },
        "retrieval": {
            "executed_searches": executed_searches,
            "statuses": retrieval_statuses,
            "invalid_query_rate": retrieval_statuses.get("invalid_query", 0) / executed_searches_total,
            "tool_exception_rate": retrieval_statuses.get("tool_exception", 0) / executed_searches_total,
            "api_error_rate": (retrieval_statuses.get("api_error", 0) + retrieval_statuses.get("processing_error", 0))
            / executed_searches_total,
        },
        "offline_rescore": {"matches": rescore_matches, "mismatches": rescore_mismatches},
    }


# --------------------------------------------------------------------------- #
# Generation (vLLM native, same engine config as the strict line)
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
    parser.add_argument("--model", type=Path, required=True, help="full model directory (Base or official merged GRPO checkpoint)")
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer directory for input rendering (official-line wrapper pins the Base tokenizer so both models see byte-identical inputs)")
    parser.add_argument("--data-files", type=Path, required=True, help="held-out parquet (smoke test or official confirm set)")
    parser.add_argument("--manifest", type=Path, default=None, help="manifest.json holding the data file SHA256")
    parser.add_argument("--manifest-key", default="heldout", help="outputs key of the data file in the manifest")
    parser.add_argument("--leakage-reference", type=Path, required=True, help="training parquet for the leakage gate")
    parser.add_argument("--search-url", default="http://127.0.0.1:18080/retrieve")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--max-envs-per-batch",
        type=int,
        default=32,
        help="max concurrent eval envs per chunk (CPU retriever capacity; pure concurrency control)",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
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
        # Chunked episode loop (CPU retriever capacity; see strict-line fix 0fe39f1).
        for batch_start in range(0, len(records), args.max_envs_per_batch):
            batch_records = records[batch_start : batch_start + args.max_envs_per_batch]
            env_config = OmegaConf.create(
                {
                    "max_steps": args.max_steps,
                    "history_length": args.history_length,
                    "search": {
                        "search_url": args.search_url,
                        "topk": args.topk,
                        "timeout": args.timeout,
                        "log_requests": True,
                    },
                }
            )
            # Official-loose: NO SearchEnvironmentManager, NO projection.
            # Raw actions go straight into the vendored skyrl SearchEnv.
            raw_envs = SearchMultiProcessEnv(
                seed=args.seed,
                env_num=len(batch_records),
                group_n=1,
                is_train=False,
                env_config=env_config,
            )
            kwargs = [
                {
                    "question": record["question"],
                    "ground_truth": {"target": record["answers"]},
                    "data_source": record["data_source"],
                }
                for record in batch_records
            ]
            observations, _ = raw_envs.reset(kwargs=kwargs)

            batch_done = np.zeros(len(batch_records), dtype=bool)
            batch_final = np.zeros(len(batch_records), dtype=float)
            batch_won = np.zeros(len(batch_records), dtype=bool)

            try:
                for step_index in range(args.max_steps):
                    active_before = ~batch_done
                    if not active_before.any():
                        break
                    generation_started = time.monotonic()
                    raw_actions = generate_actions(llm, tokenizer, observations, args)
                    next_observations, rewards, step_done, infos = raw_envs.step(raw_actions)
                    generation_seconds = time.monotonic() - generation_started
                    for local_index in range(len(batch_records)):
                        if not active_before[local_index]:
                            continue
                        global_index = batch_start + local_index
                        info_retrieval = (infos[local_index].get("retrieval") or {}) or {}
                        error_observation = (
                            info_retrieval.get("status") == "tool_exception"
                            or bool(infos[local_index].get("retrieval_failed"))
                        )
                        episodes[global_index]["steps"].append(
                            {
                                "step": step_index + 1,
                                "prompt": observations[local_index],
                                "raw_action": raw_actions[local_index],
                                "observation": jsonable(next_observations[local_index]) if local_index < len(next_observations) else None,
                                "reward": float(rewards[local_index]),
                                "done": bool(step_done[local_index]),
                                "won": bool(infos[local_index].get("won", False)),
                                "executed_search": bool(infos[local_index].get("tool_calling")),
                                "error_observation": error_observation,
                                "info": jsonable(infos[local_index]),
                                "batch_generation_seconds": generation_seconds,
                            }
                        )
                        batch_final[local_index] += float(rewards[local_index])
                        batch_won[local_index] = batch_won[local_index] or bool(infos[local_index].get("won", False))
                    batch_done |= np.asarray(step_done, dtype=bool)
                    observations = next_observations
            finally:
                raw_envs.close()
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

    metrics = aggregate_metrics(episodes)
    result = {
        "schema_version": 1,
        "kind": "p3-heldout-evaluation-official-line",
        "line": "official-loose",
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
            "line": "official-loose",
            "env": "vendored skyrl SearchEnv (SearchMultiProcessEnv, no SearchEnvironmentManager, no projection)",
            "action_parse": "re.search(r'<search>(.*?)</search>', re.DOTALL); no query -> tool exception -> error observation -> retry, no penalty",
            "termination": "done when '<answer>' and '</answer>' both appear in action, or turns >= max_steps",
            "terminal_reward": "skyrl compute_score(chat_history, ground_truth, format_score=0.1); EM = reward >= 1.0",
            "note": "patch 0004 aligns env reward format_score to the Search-R1 paper (0.1)",
        },
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
    print(f"output={output_path}")
    print(f"episodes={episodes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
