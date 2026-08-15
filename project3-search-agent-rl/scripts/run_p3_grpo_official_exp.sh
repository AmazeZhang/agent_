#!/usr/bin/env bash
# P3 phase-2 official-loose training entry (docs/P3_PHASE2_OFFICIAL_TRAIN_DESIGN_2026-08-15.md).
#
# Official-loose semantics on the upstream (official) training set:
#   - env.projection=official  (patch 0005: raw actions passthrough, valids all True)
#   - actor_rollout_ref.actor.use_invalid_action_penalty=false  (config-only)
#   - format_score=0.1 env reward (patch 0004, shared with evaluation)
#   - Qwen2.5-3B FULL-PARAM FSDP (no lora_rank); LoRA is only the fallback path
#   - 6 physical GPUs 1,2,3,4,6,7 (GPU0/5 forbidden by gate and gpu_guard)
#   - GRPO: group_n via env.rollout.n=5 (fork hard constraint rollout.n==1);
#     samples/step = train_batch_size x 5 (formal default 66 -> 330; target 132 -> 660)
#   - data.shuffle=true with fixed data.seed=1234 and trainer.seed=1234
#   - save_freq=50 (formal; checkpoints align to Step 50/100/300)
#
# Profiles (PROJECT3_TRAIN_PROFILE):
#   formal (default): total_training_steps=50 (PROJECT3_TOTAL_TRAINING_STEPS
#     override for 100/300 continuation), save_freq=50,
#     gpu_memory_utilization=0.45 (PROJECT3_GPU_MEM_UTIL override).
#   smoke: total_training_steps=1, save_freq=1 (produces a checkpoint for the
#     resume verification), gpu_memory_utilization=0.40 (first VRAM observation
#     point per user review; 0.45 next). Smoke is a verification tool only,
#     never a formal config.
#
# Resume: PROJECT3_RESUME_FROM=<absolute global_step_N dir> (verl native).
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
model_path="${project_data}/models/Qwen2.5-3B"
train_parquet="${project_data}/datasets/searchr1-upstream/train.parquet"
val_dir="${PROJECT3_VAL_DIR:-${project_data}/datasets/searchr1-smoke}"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-grpo-official-exp}"
resume_from="${PROJECT3_RESUME_FROM:-}"

profile="${PROJECT3_TRAIN_PROFILE:-formal}"
case "$profile" in
  smoke|formal) ;;
  *) echo "unknown PROJECT3_TRAIN_PROFILE: ${profile} (smoke|formal)" >&2; exit 20 ;;
esac

train_batch_size="${PROJECT3_TRAIN_BATCH_SIZE:-66}"          # 66 formal default; 132 target
mini_batch_size=$((train_batch_size * 5))                    # group_n=5 samples
total_training_steps="${PROJECT3_TOTAL_TRAINING_STEPS:-50}"  # formal default 50
official_lr="${PROJECT3_OFFICIAL_LR:-1e-5}"                  # verify vs official Search-R1 config before freeze
official_kl="${PROJECT3_OFFICIAL_KL_COEF:-0.001}"             # verify vs official Search-R1 config before freeze
trainer_seed="${PROJECT3_TRAINER_SEED:-1234}"
data_seed="${PROJECT3_DATA_SEED:-1234}"
gpu_memory_utilization="${PROJECT3_GPU_MEM_UTIL:-0.45}"
save_freq="${PROJECT3_SAVE_FREQ:-50}"
if [[ "$profile" == "smoke" ]]; then
  total_training_steps="1"
  save_freq="1"
  gpu_memory_utilization="${PROJECT3_GPU_MEM_UTIL:-0.40}"
fi

for required_path in "$python_bin" "$model_path" "$train_parquet" "$val_dir/test.parquet"; do
  if [[ ! -e "$required_path" ]]; then
    echo "required path missing: ${required_path}" >&2
    exit 11
  fi
done

if [[ ! "$total_training_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "training steps must be a positive integer" >&2
  exit 16
fi
if (( train_batch_size % 6 != 0 )); then
  echo "train_batch_size must be divisible by DP size 6 (FSDP sharding): ${train_batch_size}" >&2
  exit 21
fi
if (( mini_batch_size % 6 != 0 )); then
  echo "ppo_mini_batch_size ${mini_batch_size} not divisible by 6" >&2
  exit 22
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

# Self-check: resolved experimental values on the first stdout line.
echo "[OFFICIAL_EXP] resolved: profile=${profile} train_batch_size=${train_batch_size} mini_batch_size=${mini_batch_size} steps=${total_training_steps} lr=${official_lr} kl=${official_kl} gpu_mem=${gpu_memory_utilization} save_freq=${save_freq} seed=${trainer_seed}/${data_seed}"

overrides=(
  "algorithm.adv_estimator=grpo"
  "algorithm.norm_adv_by_std_in_grpo=true"
  "algorithm.use_kl_in_reward=false"
  "data.train_files=${train_parquet}"
  "data.val_files=${val_dir}/test.parquet"
  "data.train_batch_size=${train_batch_size}"
  "data.val_batch_size=16"
  "data.max_prompt_length=2048"
  "data.max_response_length=256"
  "data.filter_overlong_prompts=true"
  "data.truncation=left"
  "data.return_raw_chat=true"
  # shuffle with fixed seed (user review): cross-epoch prompt order is not
  # sequential; RandomSampler is seeded by data.seed (create_rl_sampler).
  # data.seed / trainer.seed are not struct fields in this fork -> `+` append.
  "data.shuffle=true"
  "+data.seed=${data_seed}"
  "actor_rollout_ref.model.path=${model_path}"
  # FULL-PARAM FSDP mainline: no lora_rank, no offload.
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.model.use_remove_padding=true"
  "actor_rollout_ref.actor.optim.lr=${official_lr}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=false"
  "actor_rollout_ref.actor.use_kl_loss=true"
  "actor_rollout_ref.actor.kl_loss_coef=${official_kl}"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.use_torch_compile=false"
  "actor_rollout_ref.actor.fsdp_config.param_offload=false"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=sync"
  # Fork hard constraint (main_ppo.py:173): actor_rollout_ref.rollout.n==1,
  # GRPO group size comes from env.rollout.n=5 below.
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "actor_rollout_ref.rollout.free_cache_engine=true"
  "actor_rollout_ref.rollout.enable_chunked_prefill=false"
  "actor_rollout_ref.rollout.max_num_batched_tokens=2304"
  "actor_rollout_ref.rollout.max_model_len=2304"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=false"
  # Official-loose semantics (patch 0005 + config-only penalty off).
  "env.env_name=search"
  "env.projection=official"
  "env.seed=0"
  "env.max_steps=2"
  "env.history_length=2"
  "env.rollout.n=5"
  "env.search.search_url=${retriever_url}"
  "env.search.topk=3"
  "env.search.timeout=180"
  "env.search.log_requests=true"
  "actor_rollout_ref.actor.use_invalid_action_penalty=false"
  "trainer.logger=['console']"
  "trainer.project_name=search_r1_repro"
  "trainer.experiment_name=p3_grpo_official_3b_fsdp6_loose_n5_b${train_batch_size}_s0"
  "+trainer.seed=${trainer_seed}"
  "trainer.n_gpus_per_node=6"
  "trainer.nnodes=1"
  "trainer.total_epochs=1"
  "trainer.total_training_steps=${total_training_steps}"
  "trainer.val_before_train=false"
  "trainer.test_freq=-1"
  "trainer.save_freq=${save_freq}"
  "trainer.default_local_dir=${run_dir}/checkpoints"
  "trainer.rollout_data_dir=${run_dir}/rollouts"
  "ray_init.num_cpus=${PROJECT3_RAY_NUM_CPUS:-64}"
  "hydra.run.dir=${run_dir}/hydra"
)

if [[ -n "$resume_from" ]]; then
  overrides+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${resume_from}")
else
  overrides+=("trainer.resume_mode=disable")
fi

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export RAY_task_events_report_interval_ms=0
export PYTHONPATH="${vendor_dir}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
# Proxy purge: retriever traffic is loopback-only and must never enter clash.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost"

if [[ "$mode" == "print-config" ]]; then
  exec "$python_bin" -m verl.trainer.main_ppo --cfg job "${overrides[@]}"
fi

if [[ -z "${PROJECT3_RUN_ID:-}" || -z "${PROJECT3_RUN_DIR:-}" ]]; then
  echo "actual training must be launched through scripts/run_managed.sh" >&2
  exit 13
fi

# 6 physical GPUs 1,2,3,4,6,7 only; GPU0/5 are forbidden by design and the
# run_managed.sh gpu_guard rejects them too (GPU0 = desktop, GPU5 unstable).
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1,2,3,4,6,7" ]]; then
  echo "P3 official-exp gate requires run_managed.sh to expose physical GPUs 1,2,3,4,6,7 (got: ${CUDA_VISIBLE_DEVICES:-<unset>})" >&2
  exit 14
fi

for patch_file in \
  "${project_dir}/patches/0001-search-retrieval-status-observability.patch" \
  "${project_dir}/patches/0002-structured-rollout-audit.patch" \
  "${project_dir}/patches/0003-graceful-ray-shutdown-and-atomic-rollout.patch" \
  "${project_dir}/patches/0004-search-prompt-and-format-reward.patch" \
  "${project_dir}/patches/0005-search-env-loose-projection.patch"; do
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
