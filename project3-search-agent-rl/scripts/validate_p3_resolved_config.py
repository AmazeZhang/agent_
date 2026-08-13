#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def nested(config: dict, dotted: str):
    value = config
    for part in dotted.split("."):
        value = value[part]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--mode", choices=("one-step", "resume-step2"), default="one-step")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    expected = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "data.train_batch_size": 8,
        "data.val_batch_size": 16,
        "data.max_prompt_length": 2048,
        "data.max_response_length": 256,
        "actor_rollout_ref.model.lora_rank": 32,
        "actor_rollout_ref.model.lora_alpha": 32,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "env.env_name": "search",
        "env.max_steps": 2,
        "env.rollout.n": 2,
        "env.search.topk": 3,
        "env.search.timeout": 180,
        "trainer.n_gpus_per_node": 1,
        "trainer.nnodes": 1,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": 1,
        "trainer.save_freq": 1,
        "trainer.test_freq": -1,
        "trainer.resume_mode": "disable",
        "ray_init.num_cpus": 32,
    }
    if args.mode == "resume-step2":
        expected.update(
            {
                "trainer.total_epochs": 2,
                "trainer.total_training_steps": 2,
                "trainer.resume_mode": "resume_path",
            }
        )
    mismatches = {}
    for key, wanted in expected.items():
        actual = nested(config, key)
        if actual != wanted:
            mismatches[key] = {"expected": wanted, "actual": actual}

    url = nested(config, "env.search.search_url")
    if url != "http://127.0.0.1:18080/retrieve":
        mismatches["env.search.search_url"] = {"expected": "loopback port 18080", "actual": url}
    if nested(config, "trainer.logger") != ["console"]:
        mismatches["trainer.logger"] = {"expected": ["console"], "actual": nested(config, "trainer.logger")}
    if "fixture" in nested(config, "env.search.search_url").lower():
        mismatches["fixture_retriever"] = {"expected": False, "actual": True}
    if args.mode == "resume-step2":
        resume_path = Path(nested(config, "trainer.resume_from_path"))
        if not resume_path.is_absolute() or resume_path.name != "global_step_1":
            mismatches["trainer.resume_from_path"] = {
                "expected": "absolute path ending in global_step_1",
                "actual": str(resume_path),
            }

    report = {
        "status": "pass" if not mismatches else "fail",
        "checks": len(expected) + 3,
        "mismatches": mismatches,
        "derived": {
            "questions_per_step": nested(config, "data.train_batch_size"),
            "trajectories_per_question": nested(config, "env.rollout.n"),
            "maximum_trajectories": nested(config, "data.train_batch_size") * nested(config, "env.rollout.n"),
            "maximum_environment_actions": nested(config, "data.train_batch_size")
            * nested(config, "env.rollout.n")
            * nested(config, "env.max_steps"),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
