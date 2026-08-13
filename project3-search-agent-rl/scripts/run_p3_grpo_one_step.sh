#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
vendor_dir="${project_dir}/vendor/verl-agent"

expected_upstream="20bd331bdbc9026a5668e11362178e10ab7400c8"
actual_upstream="$(git -C "$vendor_dir" rev-parse HEAD)"
if [[ "$actual_upstream" != "$expected_upstream" ]]; then
  echo "upstream commit mismatch: expected ${expected_upstream}, got ${actual_upstream}" >&2
  exit 10
fi

data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"
python_bin="${project_data}/envs/searchr1-repro-cu124/bin/python"
model_path="${project_data}/models/Qwen2.5-1.5B-Instruct"
dataset_dir="${project_data}/datasets/searchr1-smoke"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-grpo-one-step}"
resume_from="${PROJECT3_RESUME_FROM:-}"
total_training_steps="${PROJECT3_TOTAL_TRAINING_STEPS:-1}"
total_epochs="${PROJECT3_TOTAL_EPOCHS:-1}"

for required_path in "$python_bin" "$model_path" "$dataset_dir/train.parquet" "$dataset_dir/test.parquet"; do
  if [[ ! -e "$required_path" ]]; then
    echo "required path missing: ${required_path}" >&2
    exit 11
  fi
done

if [[ ! "$total_training_steps" =~ ^[1-9][0-9]*$ || ! "$total_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo "training steps and epochs must be positive integers" >&2
  exit 16
fi
if [[ -n "$resume_from" ]]; then
  if [[ "$resume_from" != /* || ! -d "$resume_from/actor" || ! -f "$resume_from/data.pt" ]]; then
    echo "resume checkpoint must be an absolute global_step directory with actor/ and data.pt: ${resume_from}" >&2
    exit 17
  fi
  if [[ ! "$(basename -- "$resume_from")" =~ ^global_step_[0-9]+$ ]]; then
    echo "resume checkpoint basename must be global_step_N: ${resume_from}" >&2
    exit 18
  fi
fi

if [[ ! "$retriever_url" =~ ^http://127\.0\.0\.1:[0-9]{1,5}/retrieve$ ]]; then
  echo "retriever URL must be an IPv4 loopback /retrieve endpoint: ${retriever_url}" >&2
  exit 12
fi

mode="run"
if [[ "${1:-}" == "--print-config" ]]; then
  mode="print-config"
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--print-config]" >&2
  exit 2
fi

overrides=(
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=true"
  "algorithm.use_kl_in_reward=false"
  "data.train_files=${dataset_dir}/train.parquet"
  "data.val_files=${dataset_dir}/test.parquet"
  "data.train_batch_size=8"
  "data.val_batch_size=16"
  "data.max_prompt_length=2048"
  "data.max_response_length=256"
  "data.filter_overlong_prompts=true"
  "data.truncation=left"
  "data.return_raw_chat=true"
  "data.shuffle=false"
  "actor_rollout_ref.model.path=${model_path}"
  "actor_rollout_ref.model.lora_rank=32"
  "actor_rollout_ref.model.lora_alpha=32"
  "actor_rollout_ref.model.target_modules=all-linear"
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.model.use_remove_padding=true"
  "actor_rollout_ref.actor.optim.lr=3e-6"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
  "actor_rollout_ref.actor.ppo_mini_batch_size=8"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=false"
  "actor_rollout_ref.actor.use_kl_loss=true"
  "actor_rollout_ref.actor.kl_loss_coef=0.001"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.use_torch_compile=false"
  "actor_rollout_ref.actor.fsdp_config.param_offload=true"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=sync"
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "actor_rollout_ref.rollout.free_cache_engine=true"
  "actor_rollout_ref.rollout.enable_chunked_prefill=false"
  "actor_rollout_ref.rollout.max_num_batched_tokens=2304"
  "actor_rollout_ref.rollout.max_model_len=2304"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=true"
  "env.env_name=search"
  "env.seed=0"
  "env.max_steps=2"
  "env.history_length=2"
  "env.rollout.n=2"
  "env.search.search_url=${retriever_url}"
  "env.search.topk=3"
  "env.search.timeout=180"
  "env.search.log_requests=true"
  "trainer.logger=['console']"
  "trainer.project_name=search_r1_repro"
  "trainer.experiment_name=p3_grpo_incremental_qwen25_15b_lora32_seed0"
  "trainer.n_gpus_per_node=1"
  "trainer.nnodes=1"
  "trainer.total_epochs=${total_epochs}"
  "trainer.total_training_steps=${total_training_steps}"
  "trainer.val_before_train=false"
  "trainer.test_freq=-1"
  "trainer.save_freq=1"
  "trainer.default_local_dir=${run_dir}/checkpoints"
  "trainer.rollout_data_dir=${run_dir}/rollouts"
  "ray_init.num_cpus=32"
  "hydra.run.dir=${run_dir}/hydra"
)

if [[ -n "$resume_from" ]]; then
  overrides+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${resume_from}")
else
  overrides+=("trainer.resume_mode=disable")
fi

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# This pinned veRL generation documents V1 as an explicit opt-in. Keep the
# baseline on its V0 hybrid-engine path instead of inheriting vLLM defaults.
export VLLM_USE_V1=0
# Ray 2.43.0 crashed in TaskEventBuffer::FlushEvents while the completed GPU
# actor was being terminated. Zero is Ray's own disable value for this buffer.
export RAY_task_events_report_interval_ms=0
export PYTHONPATH="${vendor_dir}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$mode" == "print-config" ]]; then
  exec "$python_bin" -m verl.trainer.main_ppo --cfg job "${overrides[@]}"
fi

if [[ -z "${PROJECT3_RUN_ID:-}" || -z "${PROJECT3_RUN_DIR:-}" ]]; then
  echo "actual training must be launched through scripts/run_managed.sh" >&2
  exit 13
fi

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "P3 one-step gate requires run_managed.sh to expose only physical GPU1" >&2
  exit 14
fi

for patch_file in \
  "${project_dir}/patches/0001-search-retrieval-status-observability.patch" \
  "${project_dir}/patches/0002-structured-rollout-audit.patch"; do
  if ! git -C "$vendor_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "required patch is not applied: $(basename -- "$patch_file")" >&2
    exit 15
  fi
done

"$python_bin" - "$retriever_url" <<'PY'
import json
import sys
from urllib.request import urlopen

retrieve_url = sys.argv[1]
health_url = retrieve_url.rsplit("/", 1)[0] + "/health"
with urlopen(health_url, timeout=5) as response:
    payload = json.load(response)
if payload.get("status") != "ready" or payload.get("vectors") != 21_015_324:
    raise SystemExit(f"retriever health gate failed: {payload}")
print(f"retriever health gate passed: {payload}")
PY

exec "$python_bin" -m verl.trainer.main_ppo "${overrides[@]}"
