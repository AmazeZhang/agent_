"""WP6: Agentic RL (GRPO) smoke on rllm's verl backend.

Real minimal GRPO run: Qwen2.5-Coder-3B-Instruct + LoRA(rank 8), 4 SWE
tasks x n=4 rollouts, shaped hidden-test reward from WP2 eval repos, one
GRPO update, checkpoint saved. Run in tmux with CUDA_VISIBLE_DEVICES on a
free GPU (1-7, never 0):

    CUDA_VISIBLE_DEVICES=1 python scripts/smoke_rl.py

Run 10 (warm start, spec WP6 "WP3 可信轨迹初始化策略后"): initialize the
policy from the SFT-merged model in a FRESH checkpoint dir so verl doesn't
resume the base-init run 9/8 ckpt (verl auto-resumes default_local_dir):

    CUDA_VISIBLE_DEVICES=1 P2_GRPO_MODEL=/media/imc/data/yzy/agent/project2/models/qwen25-coder-3b-sft-merged P2_GRPO_CKPT=/media/imc/data/yzy/agent/project2/checkpoints/smoke-grpo2 python scripts/smoke_rl.py

Acceptance (spec A3/A4): reward genuinely enters training (console shows
non-trivial reward stats), parameters update (logprobs/advantages change),
checkpoint written under checkpoints/smoke-grpo2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# rllm is vendored (not pip-installed); scripts dir holds our workflow/reward.
VENDOR = Path("/home/imc/yzy/agent/project2-coding-agent-rl/vendor/rllm")
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(SCRIPTS))

# Env overrides (run 10 warm-start from the SFT-merged model; run 11 uses the
# SFT-seen task pool via P2_GRPO_DATASET):
#   P2_GRPO_MODEL=<hf model dir>   P2_GRPO_CKPT=<fresh ckpt dir>   P2_GRPO_DATASET=<name>
MODEL = os.environ.get(
    "P2_GRPO_MODEL",
    "/media/imc/data/yzy/agent/project2/hf-cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5",
)
WORK_ROOT = "/media/imc/data/yzy/agent/project2/grpo-work"
CKPT_DIR = os.environ.get("P2_GRPO_CKPT", "/media/imc/data/yzy/agent/project2/checkpoints/smoke-grpo")
DATASET_NAME = os.environ.get("P2_GRPO_DATASET", "p2_swe_smoke")


def main() -> None:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "0" in gpu.split(",") or not gpu:
        raise SystemExit("REFUSED: set CUDA_VISIBLE_DEVICES to a free GPU (1-7), never 0")

    from hydra import compose, initialize
    from omegaconf import OmegaConf

    from rllm.data.dataset import DatasetRegistry
    from rllm.trainer.agent_trainer import AgentTrainer

    if not DatasetRegistry.dataset_exists(DATASET_NAME, "train"):
        raise SystemExit(f"dataset {DATASET_NAME} missing — run build_grpo_smoke_data.py first")

    train_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "train")
    print(f"[smoke_rl] dataset: {len(train_dataset)} tasks, gpu={gpu}")

    from swe_reward import SweHiddenTestReward
    from swe_workflow import SweSingleTurnWorkflow

    overrides = [
        "data.train_batch_size=4",
        "data.val_batch_size=4",
        "data.max_prompt_length=1024",
        "data.max_response_length=2048",
        f"actor_rollout_ref.model.path={MODEL}",
        "actor_rollout_ref.model.use_remove_padding=True",
        # Top-level FSDP fields, NOT the nested lora.rank block (megatron-only):
        # verl FSDP's _build_lora_module reads model_config.lora_rank; the nested
        # lora.rank is ignored there, leaving lora_rank=0 -> trainer sends plain
        # q_proj.weight while the vLLM worker (LoRA-enabled) expects
        # qkv_proj.base_layer.weight -> KeyError on weight sync.
        "actor_rollout_ref.model.lora_rank=8",
        "actor_rollout_ref.model.lora_alpha=16",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192",
        # _update_actor (verl trainer/ppo/ray_trainer.py) multiplies
        # ppo_mini_batch_size by rollout.n before assigning mini_batch_size to
        # the flattened rollout tensordict: 2 * 4 = 8 rollout samples per
        # mini-batch, 16 % 8 == 0 (4 tasks x n=4). An 8 here would produce
        # mini_batch_size=32 > batch of 16 -> AssertionError 16 % 32 != 0.
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0.0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # Engine strategy MUST be set at the top level: FSDPActorConfig.__post_init__
        # (verl workers/config/actor.py) does `engine.strategy = self.strategy`, so
        # fsdp_config.strategy overrides are clobbered by the top-level field.
        # FSDP1 silently ignores param_offload for the actor ("We force turn off
        # CPUOffload for actor because it causes incorrect results when using grad
        # accumulation" — verl fsdp/transformer_impl.py), keeping the full 3B model
        # GPU-resident. fsdp2 + offload_policy gives a real CPUOffloadPolicy so
        # weights leave the GPU between steps.
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.offload_policy=True",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "actor_rollout_ref.ref.fsdp_config.offload_policy=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.temperature=1.0",
        # Actor+ref models (FSDP, ~13 GiB) are already resident when the
        # vLLM server starts; 0.75×23.5 GiB fails the free-memory gate.
        # Eager mode drops vLLM's torch.compile + CUDA-graph workspace
        # (~7 GiB transient on this stack), leaving room for the KV cache.
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.max_num_seqs=16",
        # n=4 (16 train rollouts) to give the shaped reward a realistic chance
        # of non-zero advantages — the base 3B model's patches rarely apply.
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "algorithm.adv_estimator=grpo",
        "rllm.compact_filtering.enable=False",
        "rllm.compact_filtering.mask_max_prompt_length_exceeded=True",
        "rllm.compact_filtering.mask_max_response_length_exceeded=True",
        "rllm.compact_filtering.mask_max_turns_exceeded=False",
        "rllm.rejection_sample.enable=False",
        "rllm.rejection_sample.multiplier=1.0",
        "rllm.stepwise_advantage.enable=False",
        "rllm.workflow.use_workflow=True",
        "rllm.workflow.n_parallel_tasks=4",
        "trainer.critic_warmup=0",
        "trainer.logger=['console']",
        "trainer.project_name=p2-smoke",
        "trainer.experiment_name=swe-grpo",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=1",
        f"trainer.default_local_dir={CKPT_DIR}",
        "trainer.default_hdfs_dir=null",
        "trainer.total_epochs=1",
    ]

    with initialize(version_base=None, config_path="pkg://rllm.trainer.config", job_name="p2-smoke-grpo"):
        config = compose(config_name="agent_ppo_trainer", overrides=overrides)

    print("[smoke_rl] config resolved:")
    print(OmegaConf.to_yaml(config))

    reward_fn = SweHiddenTestReward(work_root=WORK_ROOT)

    trainer = AgentTrainer(
        workflow_class=SweSingleTurnWorkflow,
        workflow_args={"reward_fn": reward_fn},
        train_dataset=train_dataset,
        # verl always constructs the val dataloader (even with val_before_train=False),
        # so point val_files at the same parquet instead of verl's default gsm8k.
        val_dataset=train_dataset,
        config=config,
        backend="verl",
    )
    trainer.train()
    print("[smoke_rl] DONE")


if __name__ == "__main__":
    main()
