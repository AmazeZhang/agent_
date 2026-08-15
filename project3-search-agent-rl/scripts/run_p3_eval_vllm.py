#!/usr/bin/env python3
"""Pure-evaluation harness for P3 (vLLM-native greedy backend).

Inference-only twin of run_p3_eval_heldout.py. Identical gates, identical
environment semantics (SearchMultiProcessEnv + SearchEnvironmentManager +
search_projection + skyRL strict-EM episode reward), identical fixed
parameters (seed=0, max_steps=2, history_length=2, topk=3, timeout=180,
max_input_tokens=2048, max_new_tokens=256, temperature 0). The only
difference is the decoding backend: a native vLLM engine with
SamplingParams(temperature=0) greedy instead of HF transformers
generate(do_sample=False).

By construction this script creates no optimizer, no scheduler, never calls
backward, and does not import Ray. There is no HF optimizer state to mount:
LoRA is loaded through vLLM's native LoRARequest (PEFT directory) and is
inference-only.

Engine parity with the training rollout path (run_p3_grpo_fix_exp.sh):
  - VLLM_USE_V1=0  (wrapper exports it; script aborts unless set to "0")
  - dtype=bfloat16, tensor_parallel_size=1, gpu_memory_utilization=0.6,
    enforce_eager=True, max_model_len=2304
  - tokenizer preprocessing identical to the HF eval script
    (apply_chat_template with add_generation_prompt, truncation at
    max_input_tokens=2048); token ids are handed to the engine directly so
    the input side is byte-identical to the HF backend.
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import vllm
from vllm.lora.request import LoRARequest  # vLLM 0.8.5 exposes it here, not at top level

# Importing the training trainer would drag in optimizers and Ray; this module
# deliberately never imports verl.trainer.main_ppo / torch.optim / ray.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.env_manager import SearchEnvironmentManager
from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv
from agent_system.environments.env_package.search.projection import search_projection
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import (
    compute_score as skyrl_compute_score,
    extract_solution,
)

EXPECTED_RETRIEVER_VECTORS = 21_015_324
REAL_INDEX_NOTE = "real Wiki-18 IndexFlatIP (21,015,324 vectors); ground-truth-derived fixture is prohibited for evaluation"
DECODING_NOTE = (
    "decoding_backend=vllm-native-greedy (temperature 0, same vLLM engine path as the "
    "training rollout: V0 engine, bfloat16, gpu_memory_utilization 0.6, enforce_eager, "
    "max_model_len 2304). Training-time sampling keeps exploration; evaluation is fixed "
    "greedy. Same-condition comparison against hf-transformers-greedy is per-question on "
    "EM, search rate, invalid actions and raw action text."
)

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
# Gates (pure, unit-testable; identical to run_p3_eval_heldout.py)
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
    """Return overlap counts; raise if any eval question appears in the reference split."""
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
    """Engine-path gate: the vLLM eval must ride the same engine configuration
    as the training rollout (V0 hybrid engine), otherwise the backend-parity
    claim is void."""
    if os.environ.get("VLLM_USE_V1", "0") != "0":
        raise RuntimeError(
            f"VLLM_USE_V1 must be \"0\" (training rollout engine path), got: {os.environ.get('VLLM_USE_V1')}"
        )
    return {"vllm_use_v1": "0", "vllm_version": vllm.__version__}


def validate_adapter_for_vllm(adapter: Path) -> dict[str, Any]:
    """PEFT-directory checks that must hold for vLLM's native LoRA loader.
    The actual weight load still happens on GPU and aborts the run on failure."""
    if not adapter.is_dir() or not (adapter / "adapter_config.json").is_file():
        raise RuntimeError(f"adapter directory invalid: {adapter}")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("peft_type") != "LORA":
        raise RuntimeError(f"adapter must be a LoRA adapter, peft_type={config.get('peft_type')}")
    rank = config.get("r")
    if rank != 32:
        raise RuntimeError(f"adapter rank must be 32 (training lora_rank=32), got r={rank}")
    target_modules = config.get("target_modules")
    if not target_modules:
        raise RuntimeError("adapter target_modules must not be empty for vLLM LoRA loading")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("vLLM LoRA loading requires adapter_model.safetensors in the adapter directory")
    return {
        "path": str(adapter.resolve()),
        "peft_type": config.get("peft_type"),
        "r": rank,
        "target_modules": target_modules,
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "adapter_model.safetensors": sha256_file(adapter / "adapter_model.safetensors"),
        "adapter_config.json": sha256_file(adapter / "adapter_config.json"),
    }


# --------------------------------------------------------------------------- #
# Action / answer analysis (pure; identical to run_p3_eval_heldout.py)
# --------------------------------------------------------------------------- #

SEARCH_TAG_RE = re.compile(r"<search>", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer>", re.IGNORECASE)


def action_quality(raw_action: str, projected_valid: bool) -> dict[str, bool]:
    has_search = SEARCH_TAG_RE.search(raw_action) is not None
    has_answer = ANSWER_TAG_RE.search(raw_action) is not None
    duplicate_tags = (
        len(SEARCH_TAG_RE.findall(raw_action)) > 1
        or len(ANSWER_TAG_RE.findall(raw_action)) > 1
        or len(ANSWER_CLOSE_RE.findall(raw_action)) > 1
    )
    return {
        "has_search_tag": has_search,
        "has_answer_tag": has_answer,
        "mixed_tags": has_search and has_answer,
        "duplicate_tags": duplicate_tags,
        "projected_valid": projected_valid,
    }


def actions_text(episode_steps: list[dict[str, Any]]) -> str:
    return "\n".join(step["raw_action"] for step in episode_steps)


def offline_rescore(episode_steps: list[dict[str, Any]], answers: list[str]) -> dict[str, Any]:
    """Audit-only re-score over the concatenated model actions.

    Uses only the model's own outputs (not retriever document text, which may
    contain '<answer>' strings) and the same skyRL strict-EM semantics as the
    training-time environment reward.
    """
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
    invalid_actions = 0
    mixed_tag_steps = 0
    duplicate_tag_steps = 0
    executed_searches = 0
    retrieval_statuses: dict[str, int] = {}
    rescore_matches = 0
    rescore_mismatches = 0

    for episode in episodes:
        source = episode["source"]
        bucket = per_source.setdefault(source, {"n": 0, "em": 0, "success": 0, "answer_compliance": 0})
        bucket["n"] += 1
        bucket["em"] += 1 if episode["reward"] >= 1.0 else 0
        bucket["success"] += 1 if episode["won"] else 0
        bucket["answer_compliance"] += 1 if episode["offline"]["has_answer"] else 0
        for step in episode["steps"]:
            total_steps += 1
            quality = step["action_quality"]
            if not quality["projected_valid"]:
                invalid_actions += 1
            if quality["mixed_tags"]:
                mixed_tag_steps += 1
            if quality["duplicate_tags"]:
                duplicate_tag_steps += 1
            if step.get("executed_search"):
                executed_searches += 1
                status = step["info"].get("retrieval", {}).get("status")
                if status is not None:
                    retrieval_statuses[status] = retrieval_statuses.get(status, 0) + 1
        if episode["offline"]["score"] == episode["reward"]:
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
            "invalid_actions": invalid_actions,
            "invalid_action_ratio": invalid_actions / max(total_steps, 1),
            "mixed_tag_steps": mixed_tag_steps,
            "duplicate_tag_steps": duplicate_tag_steps,
        },
        "retrieval": {
            "executed_searches": executed_searches,
            "statuses": retrieval_statuses,
            "invalid_query_rate": retrieval_statuses.get("invalid_query", 0) / executed_searches_total,
            "api_error_rate": (retrieval_statuses.get("api_error", 0) + retrieval_statuses.get("processing_error", 0))
            / executed_searches_total,
        },
        "offline_rescore": {"matches": rescore_matches, "mismatches": rescore_mismatches},
    }


# --------------------------------------------------------------------------- #
# Generation (vLLM native; the only backend difference vs the HF twin)
# --------------------------------------------------------------------------- #

def build_engine(tokenizer, args: argparse.Namespace) -> vllm.LLM:
    """vLLM engine with the training-rollout engine configuration (V0)."""
    engine_kwargs: dict[str, Any] = {
        "model": str(args.model),
        "tokenizer": str(args.model),
        "dtype": VLLM_DTYPE,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "enforce_eager": True,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "seed": args.seed,
        "trust_remote_code": False,
    }
    if args.adapter is not None:
        # Native vLLM LoRA mount; inference-only, no optimizer state by construction.
        engine_kwargs.update({"enable_lora": True, "max_loras": 1, "max_lora_rank": 32})
    return vllm.LLM(**engine_kwargs)


def generate_actions(llm: vllm.LLM, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    """Greedy generation through the vLLM engine.

    Input side is byte-identical to the HF twin: apply_chat_template with
    add_generation_prompt=True, truncation at max_input_tokens. Token ids go to
    the engine directly so padding/truncation behavior cannot diverge.
    """
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
    lora_request = None
    if args.adapter is not None:
        # vLLM 0.8.5 LoRARequest takes lora_path (path is a read-only property).
        lora_request = LoRARequest(
            lora_name="p3-adapter", lora_int_id=1, lora_path=str(args.adapter)
        )
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        lora_request=lora_request,
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
    parser.add_argument("--model", type=Path, required=True, help="base HF model directory")
    parser.add_argument("--adapter", type=Path, default=None, help="PEFT LoRA adapter directory (absent for base model)")
    parser.add_argument("--data-files", type=Path, required=True, help="held-out parquet (smoke test or heldout-32)")
    parser.add_argument("--manifest", type=Path, default=None, help="manifest.json holding the data file SHA256")
    parser.add_argument("--manifest-key", default="heldout", help="outputs key of the data file in the manifest")
    parser.add_argument("--leakage-reference", type=Path, required=True, help="training parquet for the leakage gate")
    parser.add_argument("--search-url", default="http://127.0.0.1:18080/retrieve")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=2)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_info = validate_managed_environment()
    engine_info = validate_vllm_engine_parity()
    run_dir = Path(env_info["run_dir"]).resolve()
    output_path = args.output or run_dir / "results.json"
    episodes_path = run_dir / "episodes.jsonl"

    # Gates (abort on failure, never warn-and-continue).
    leakage = leakage_check(load_eval_records(args.data_files), args.leakage_reference)
    health = retriever_health_check(args.search_url)
    data_hash = verify_data_hash(args.data_files, args.manifest, args.manifest_key)

    adapter_info = None
    if args.adapter is not None:
        adapter_info = validate_adapter_for_vllm(args.adapter)

    records = load_eval_records(args.data_files)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.monotonic()
    # vLLM engine init loads the model (and, when --adapter is given, mounts the
    # LoRA weights through vLLM's native LoRARequest path). Any load failure
    # aborts here — it is the GPU-side verification of the load paths.
    llm = build_engine(tokenizer, args)
    torch.cuda.reset_peak_memory_stats()

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
    raw_envs = SearchMultiProcessEnv(
        seed=args.seed,
        env_num=len(records),
        group_n=1,
        is_train=False,
        env_config=env_config,
    )
    manager = SearchEnvironmentManager(raw_envs, search_projection, OmegaConf.create({"env": env_config}))
    kwargs = [
        {
            "question": record["question"],
            "ground_truth": {"target": record["answers"]},
            "data_source": record["data_source"],
        }
        for record in records
    ]
    observations, _ = manager.reset(kwargs=kwargs)

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
        for step_index in range(args.max_steps):
            active_before = ~done
            if not active_before.any():
                break
            generation_started = time.monotonic()
            raw_actions = generate_actions(llm, tokenizer, observations["text"], args)
            projected_actions, projected_valids = search_projection(raw_actions)
            next_observations, rewards, step_done, infos = manager.step(raw_actions)
            generation_seconds = time.monotonic() - generation_started
            for index in range(len(records)):
                if not active_before[index]:
                    continue
                executed_search = bool(infos[index].get("tool_calling"))
                episodes[index]["steps"].append(
                    {
                        "step": step_index + 1,
                        "prompt": observations["text"][index],
                        "raw_action": raw_actions[index],
                        "projected_action": projected_actions[index],
                        "action_quality": action_quality(raw_actions[index], bool(projected_valids[index])),
                        "observation": jsonable(next_observations["anchor"][index]),
                        "reward": float(rewards[index]),
                        "done": bool(step_done[index]),
                        "won": bool(infos[index].get("won", bool(step_done[index]) and float(rewards[index]) >= 1.0)),
                        "executed_search": executed_search,
                        "info": jsonable(infos[index]),
                        "batch_generation_seconds": generation_seconds,
                    }
                )
                final_rewards[index] += float(rewards[index])
                won[index] = won[index] or bool(infos[index].get("won", False))
            done |= np.asarray(step_done, dtype=bool)
            observations = next_observations
    finally:
        raw_envs.close()
        # vLLM 0.8.5 LLM has no shutdown(); releasing the engine is what frees
        # the GPU allocations before process exit.
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
        "kind": "p3-heldout-evaluation",
        # Runtime code fingerprint: this eval script's own SHA256, so every run
        # records exactly which harness version produced the numbers.
        "runtime_script_sha256": sha256_file(Path(__file__).resolve()),
        "training": False,
        "training_operations": "none by construction: no optimizer, no scheduler, no backward, no Ray",
        "decoding_backend": "vllm-native-greedy",
        "decoding_note": DECODING_NOTE,
        "engine": {
            **engine_info,
            "dtype": VLLM_DTYPE,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "enforce_eager": True,
            "max_model_len": VLLM_MAX_MODEL_LEN,
            "enable_lora": args.adapter is not None,
            "max_lora_rank": 32 if args.adapter is not None else None,
        },
        "retriever_note": REAL_INDEX_NOTE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": env_info["run_id"],
        "model_path": str(args.model.resolve()),
        "adapter": adapter_info,
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
