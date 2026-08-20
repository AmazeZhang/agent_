#!/usr/bin/env bash
# P3 clean-upstream GRPO vs GiGPO shared training wrapper
# (user directive 2026-08-20: freeze the Search-aware Reward line, switch to a
# clean-upstream algorithm comparison on Qwen2.5-3B-Instruct).
#
# THE ONLY ALGORITHM VARIABLE IS `algorithm.adv_estimator` (grpo | gigpo).
# Everything else is a fixed common config on the PRISTINE upstream tree
# vendor/upstream-20bd331b (20bd331b, patches 0001-0009 NOT applied):
#
#   env.max_steps=4  env.history_length=4  env.rollout.n=5 (group)
#   upstream search_projection / skyrl compute_score / SEARCH_TEMPLATE prompts
#   Qwen2.5-3B-Instruct (starting model, Step0), train_batch_size=66 (DP6),
#   FSDP param+optimizer+ref offload (verified 6x24GB full-param topology),
#   gpu_mem=0.60, max_num_seqs=64, CPU E5 retriever, physical GPUs 1,2,3,4,6,7.
#
# GiGPO-specific overrides (only when PROJECT3_ADV_ESTIMATOR=gigpo):
#   algorithm.adv_estimator=gigpo  algorithm.gamma=0.95
#   algorithm.gigpo.step_advantage_w=1.0  algorithm.gigpo.mode=mean_std_norm
#   algorithm.gigpo.enable_similarity=true  algorithm.gigpo.similarity_thresh=0.9
#
# Profiles: total_training_steps is a plain positive integer.
#   eng-smoke: 1 step (save_freq=1) -- topology + pipeline verification only,
#     1-step EM/advantage must never be read as an algorithm signal.
#   10-step direction runs: 10 steps, save_freq=5 (midpoint checkpoint as a
#     safety net), then scripts/model_merger.py merge + confirm-256 eval.
#
# Usage (must run inside run_managed.sh via start_tmux_run.sh):
#   PROJECT3_ADV_ESTIMATOR=grpo|gigpo
#   PROJECT3_TOTAL_TRAINING_STEPS=<positive int>   (default 1)
#   PROJECT3_MODEL_PATH=<hf model dir>             (default Qwen2.5-3B-Instruct)
#   PROJECT3_SAVE_FREQ=<int>                       (default 1)
#   bash scripts/run_p3_grpo_gigpo_shared.sh [--print-config|--dump-overrides]
#
# Exit codes: 10 clean-tree commit mismatch, 11 missing paths, 12 retriever URL,
# 13 not managed, 14 GPU mapping, 15 clean tree dirty / patch markers present,
# 16 invalid total_training_steps, 17 train_batch_size % 6, 18 mini % 6,
# 19 unknown adv_estimator, 20 CPU memory gate, 21 model weight shards missing.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
# CLEAN line: the pristine upstream worktree (NO patches 0001-0009).
vendor_dir="${project_dir}/vendor/upstream-20bd331b"

expected_upstream="20bd331bdbc9026a5668e11362178e10ab7400c8"
actual_upstream="$(git -C "$vendor_dir" rev-parse HEAD 2>/dev/null || echo MISSING)"
if [[ "$actual_upstream" != "$expected_upstream" ]]; then
  echo "clean-tree commit mismatch: expected ${expected_upstream}, got ${actual_upstream}" >&2
  exit 10
fi

# The clean line MUST stay pristine: no staged/unstaged/untracked changes, and
# no patch marker may be present anywhere in the tree (0007/0008 marker
# "search_aware_step_reward"; absence of patches is the clean-line contract).
if [[ -n "$(git -C "$vendor_dir" status --porcelain)" ]]; then
  echo "clean tree is not pristine (git status --porcelain is non-empty)" >&2
  git -C "$vendor_dir" status --porcelain | head -20 >&2
  exit 15
fi
if grep -rq "search_aware_step_reward" "$vendor_dir" --include="*.py" 2>/dev/null; then
  echo "clean tree contains patch marker 'search_aware_step_reward' (patches must NOT be applied)" >&2
  exit 15
fi

adv_estimator="${PROJECT3_ADV_ESTIMATOR:-}"
case "$adv_estimator" in
  grpo|gigpo) ;;
  *)
    echo "PROJECT3_ADV_ESTIMATOR must be grpo or gigpo, got: ${adv_estimator:-<unset>}" >&2
    exit 19
    ;;
esac

data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"
python_bin="${project_data}/envs/searchr1-repro-cu124/bin/python"
# Starting model = the Qwen2.5-3B-Instruct Step0 (never from gs300, never from
# the patched-line checkpoints). Both GRPO and GiGPO launches start from this
# same model; tokenizer = the Instruct tokenizer (byte-identical to Step0 eval).
model_path="${PROJECT3_MODEL_PATH:-${project_data}/models/Qwen2.5-3B-Instruct}"
train_parquet="${project_data}/datasets/searchr1-upstream/train.parquet"
val_dir="${PROJECT3_VAL_DIR:-${project_data}/datasets/searchr1-smoke}"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-grpo-gigpo-shared}"

mode="run"
if [[ "${1:-}" == "--print-config" ]]; then
  mode="print-config"
  shift
fi
if [[ "${1:-}" == "--dump-overrides" ]]; then
  mode="dump-overrides"
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--print-config|--dump-overrides]" >&2
  exit 2
fi

train_batch_size="${PROJECT3_TRAIN_BATCH_SIZE:-66}"          # DP6-divisible
mini_batch_size=$((train_batch_size * 5))                    # group_n=5 samples
total_training_steps="${PROJECT3_TOTAL_TRAINING_STEPS:-1}"
gpu_memory_utilization="${PROJECT3_GPU_MEM_UTIL:-0.60}"      # verified full-param topology
max_num_seqs="${PROJECT3_MAX_NUM_SEQS:-64}"
save_freq="${PROJECT3_SAVE_FREQ:-1}"
offload_param=true
offload_optimizer=true
offload_ref=true
official_lr="${PROJECT3_OFFICIAL_LR:-1e-6}"
official_kl="${PROJECT3_OFFICIAL_KL_COEF:-0.001}"
official_warmup_ratio="${PROJECT3_OFFICIAL_WARMUP_RATIO:-0.285}"
trainer_seed="${PROJECT3_TRAINER_SEED:-1234}"
data_seed="${PROJECT3_DATA_SEED:-1234}"

for required_path in "$python_bin" "$model_path/config.json" \
  "$model_path/tokenizer_config.json" "$train_parquet" "$val_dir/test.parquet"; do
  if [[ ! -e "$required_path" ]]; then
    echo "required path missing: ${required_path}" >&2
    exit 11
  fi
done
shopt -s nullglob
model_shards=("$model_path"/model-*.safetensors)
shopt -u nullglob
if (( ${#model_shards[@]} < 1 )); then
  echo "model weight shards missing under ${model_path} (model-*.safetensors)" >&2
  exit 21
fi

if [[ ! "$total_training_steps" =~ ^[1-9][0-9]*$ ]]; then
  echo "PROJECT3_TOTAL_TRAINING_STEPS must be a positive integer, got: ${total_training_steps}" >&2
  exit 16
fi
if (( train_batch_size % 6 != 0 )); then
  echo "train_batch_size must be divisible by DP size 6 (FSDP sharding): ${train_batch_size}" >&2
  exit 17
fi
if (( mini_batch_size % 6 != 0 )); then
  echo "ppo_mini_batch_size ${mini_batch_size} not divisible by 6" >&2
  exit 18
fi

if [[ ! "$retriever_url" =~ ^http://127\.0\.0\.1:[0-9]{1,5}/retrieve$ ]]; then
  echo "retriever URL must be an IPv4 loopback /retrieve endpoint: ${retriever_url}" >&2
  exit 12
fi

# Self-check: resolved experimental values on the first stdout line.
echo "[SHARED_EXP] resolved: adv_estimator=${adv_estimator} model=$(basename -- "$model_path") train_batch_size=${train_batch_size} mini_batch_size=${mini_batch_size} total_training_steps=${total_training_steps} lr=${official_lr} lr_warmup_steps_ratio=${official_warmup_ratio} kl=${official_kl} gpu_mem=${gpu_memory_utilization} max_num_seqs=${max_num_seqs} offload_param=${offload_param} offload_optimizer=${offload_optimizer} offload_ref=${offload_ref} save_freq=${save_freq} seed=${trainer_seed}/${data_seed}"

overrides=(
  "algorithm.adv_estimator=${adv_estimator}"
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
  "data.shuffle=true"
  "+data.seed=${data_seed}"
  "actor_rollout_ref.model.path=${model_path}"
  # FULL-PARAM FSDP mainline (verified topology, same as official-loose).
  "actor_rollout_ref.model.enable_gradient_checkpointing=true"
  "actor_rollout_ref.model.use_remove_padding=true"
  "actor_rollout_ref.actor.optim.lr=${official_lr}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_dynamic_bsz=false"
  "actor_rollout_ref.actor.use_kl_loss=true"
  "actor_rollout_ref.actor.kl_loss_coef=${official_kl}"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.use_torch_compile=false"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${offload_param}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload_optimizer}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=sync"
  "actor_rollout_ref.rollout.n=1"           # fork hard constraint (GRPO/GiGPO group via env.rollout.n)
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "actor_rollout_ref.rollout.free_cache_engine=true"
  "actor_rollout_ref.rollout.enable_chunked_prefill=false"
  "actor_rollout_ref.rollout.max_num_batched_tokens=2304"
  "actor_rollout_ref.rollout.max_model_len=2304"
  "actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${offload_ref}"
  # Clean-upstream search env: upstream search_projection, skyrl compute_score
  # terminal reward, SEARCH_TEMPLATE prompts. NO patch-0001..0009 switches.
  "env.env_name=search"
  "env.seed=0"
  "env.max_steps=4"
  "env.history_length=4"
  "env.rollout.n=5"
  "env.search.search_url=${retriever_url}"
  "env.search.topk=3"
  "env.search.timeout=180"
  "env.search.log_requests=true"
  "trainer.logger=['console']"
  "trainer.project_name=search_r1_repro"
  "trainer.experiment_name=p3_grpo_gigpo_shared_${adv_estimator}_3binstruct_fsdp6_n5_b${train_batch_size}_s0"
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
  "actor_rollout_ref.actor.optim.lr_warmup_steps=-1"
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${official_warmup_ratio}"
)

if [[ "$adv_estimator" == "gigpo" ]]; then
  # GiGPO overrides (user directive): discount 0.95, step_advantage_w=1.0,
  # mean_std_norm, similarity-based anchor grouping with threshold 0.9.
  overrides+=(
    "algorithm.gamma=0.95"
    "algorithm.gigpo.step_advantage_w=1.0"
    "algorithm.gigpo.mode=mean_std_norm"
    "algorithm.gigpo.enable_similarity=true"
    "algorithm.gigpo.similarity_thresh=0.9"
  )
else
  # GRPO: episodic advantage, gamma=1.0 explicit (fingerprint stability).
  overrides+=("algorithm.gamma=1.0")
fi

overrides+=("trainer.resume_mode=disable")

# Canonical fingerprint of the resolved overrides (sorted, stable): recorded in
# the run log; the wrapper's own config path is fingerprinted the same way.
config_fp="$(printf '%s\n' "${overrides[@]}" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "[SHARED_EXP] resolved_config_sha256=${config_fp}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export RAY_task_events_report_interval_ms=0
# CLEAN line: the pristine upstream tree must shadow the patched vendor.
export PYTHONPATH="${vendor_dir}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost"

if [[ "$mode" == "print-config" ]]; then
  exec "$python_bin" -m verl.trainer.main_ppo --cfg job "${overrides[@]}"
fi

if [[ "$mode" == "dump-overrides" ]]; then
  printf '%s\n' "${overrides[@]}"
  echo "__config_fp__=${config_fp}"
  exit 0
fi

if [[ -z "${PROJECT3_RUN_ID:-}" || -z "${PROJECT3_RUN_DIR:-}" ]]; then
  echo "actual training must be launched through scripts/run_managed.sh" >&2
  exit 13
fi

# CPU memory / swap gate (identical to official-loose: offload needs tens of
# GiB of CPU RAM; the retriever index is ~64.5GB on disk). MemAvailable >=
# 96GiB -> proceed; 64-96GiB -> pause, report, abort; < 64GiB -> abort.
# Already-noticeable swap (> 2GiB used) -> abort.
"$python_bin" - "$run_dir" <<'PY'
import os
import subprocess
import sys

run_dir = sys.argv[1]

def meminfo():
    values = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0]) * 1024  # kB -> bytes
    return values

def gib(value):
    return value / 1024**3

mem = meminfo()
retriever_rss = 0
try:
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\0", b" ")
        except OSError:
            continue
        if b"serve_p25_cpu_retriever" in cmdline:
            with open(f"/proc/{pid}/status") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        retriever_rss += int(line.split()[1]) * 1024
                        break
except OSError:
    pass
import glob as _glob
ray_tmp_bytes = 0
for path in _glob.glob("/tmp/p3r.*"):
    try:
        result = subprocess.run(["du", "-sb", path], capture_output=True, text=True, check=True)
        ray_tmp_bytes += int(result.stdout.split()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        pass
disk = subprocess.run(["df", "-B1", "--output=avail", run_dir],
                      capture_output=True, text=True).stdout.splitlines()
disk_free = int(disk[1]) if len(disk) > 1 else 0

report = {
    "MemTotal_GiB": round(gib(mem["MemTotal"]), 1),
    "MemAvailable_GiB": round(gib(mem["MemAvailable"]), 1),
    "SwapTotal_GiB": round(gib(mem["SwapTotal"]), 1),
    "SwapFree_GiB": round(gib(mem["SwapFree"]), 1),
    "retriever_rss_GiB": round(gib(retriever_rss), 1),
    "ray_tmp_bytes_GiB": round(gib(ray_tmp_bytes), 1),
    "checkpoint_disk_free_GiB": round(gib(disk_free), 1),
}
print(f"[SHARED_EXP] cpu_memory_gate report: {report}", flush=True)
available = mem["MemAvailable"]
swap_used = mem["SwapTotal"] - mem["SwapFree"]
if available < 64 * 1024**3:
    raise SystemExit("cpu memory gate failed: MemAvailable < 64GiB, aborting")
if available < 96 * 1024**3:
    raise SystemExit("cpu memory gate: MemAvailable in 64-96GiB band, pausing (no auto start)")
if swap_used > 2 * 1024**3:
    raise SystemExit(f"cpu memory gate failed: already swapping ({swap_used/1024**3:.1f}GiB used), aborting")
if disk_free < 100 * 1024**3:
    print(f"[SHARED_EXP] warning: checkpoint disk < 100GiB ({gib(disk_free):.0f}GiB), advisory only", flush=True)
print("[SHARED_EXP] cpu memory gate passed", flush=True)
PY

# 6 physical GPUs 1,2,3,4,6,7 only (GPU0 = desktop, GPU5 unstable); run_managed.sh
# gpu_guard rejects the forbidden pair too.
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1,2,3,4,6,7" ]]; then
  echo "P3 shared gate requires run_managed.sh to expose physical GPUs 1,2,3,4,6,7 (got: ${CUDA_VISIBLE_DEVICES:-<unset>})" >&2
  exit 14
fi

"$python_bin" - "$retriever_url" <<'PY'
import json
import sys
from urllib.request import urlopen

retrieve_url = sys.argv[1]
health_url = retrieve_url.rsplit("/", 1)[0] + "/health"
with urlopen(health_url, timeout=5) as response:
    payload = json.load(response)
# 21,015,324 vectors = the real Wiki-18 index; max_concurrent_queries must be
# the rate-limit config chosen by the stress matrix (threads=8, limit=64).
if (
    payload.get("status") != "ready"
    or payload.get("vectors") != 21_015_324
    or payload.get("max_concurrent_queries") != 64
):
    raise SystemExit(f"retriever health gate failed: {payload}")
print(f"retriever health gate passed: {payload}")
PY

exec "$python_bin" -m verl.trainer.main_ppo "${overrides[@]}"
