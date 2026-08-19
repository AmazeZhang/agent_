#!/usr/bin/env bash
# P3 Phase 4B Search-aware GRPO v1 engineering entry
# (docs/P3_PHASE4_SEARCH_CAUSAL_DIAG_RESULT_2026-08-19.md §9).
#
# Independent configuration/profile from the official-loose baseline: the
# official-loose config, checkpoints, eval scripts and results are NOT touched.
# This wrapper exists to run the frozen v1 reward (patch 0007):
#
#   R = R_answer + 0.15*evidence_hit + 0.30*sce - 0.20*invalid_or_error
#       - 0.45*redundant_search_count - 0.20*new_answer_leak_in_query
#   format_score=0.1 (unchanged), valid_retrieval coefficient alpha=0,
#   use_invalid_action_penalty=false (config-only), adv_estimator=grpo
#   (GiGPO deferred), starting model = Qwen2.5-3B Base (never from gs300).
#
# Step-attribution: R_answer (+ format + sce settled via episode metadata)
# ONLY on the terminal answer step; evidence/invalid/redundant/answer-leak
# shaping on the corresponding search step; Observation tokens never enter the
# policy loss (mask unchanged). All 8 reward components are recorded per
# episode; per-uid component sum == per-record placed score sum (exact cents).
#
# Profiles (PROJECT3_TRAIN_PROFILE):
#   eng-smoke (default): 1 training step (PROJECT3_TOTAL_TRAINING_STEPS must be
#     1 or 2), save_freq=1, gpu_mem=0.60, max_num_seqs=64, full
#     param/optimizer/ref offload -- the verified 6x24GB full-param topology
#     (official-offload-smoke 2026-08-16). Verifies ONLY: VRAM, online v1
#     reward computation (placement + sum assertion), non-zero gradients,
#     checkpoint + resume. NEVER judged by 1-2 step EM (behavior question).
#     Optional resume verification: PROJECT3_RESUME_FROM=<.../global_step_1>
#     with total_training_steps=2 (one new update, stops at global_step_2).
#   behavior-smoke: 5-10 steps (PROJECT3_TOTAL_TRAINING_STEPS in 5..10),
#     save_freq=5; pre-registered behavior observation (search rate /
#     component trigger counts / advantage direction). FAIL-CLOSED: requires
#     PROJECT3_BEHAVIOR_SMOKE_APPROVED=yes -- every GPU action still needs the
#     user's separate approval at launch time (Phase 4B requirement 8).
#
# Every GPU launch must go through scripts/run_managed.sh (PROJECT3_RUN_ID /
# PROJECT3_RUN_DIR required below); physical GPUs 1,2,3,4,6,7 only.
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
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-grpo-v1}"
resume_from="${PROJECT3_RESUME_FROM:-}"

profile="${PROJECT3_TRAIN_PROFILE:-eng-smoke}"
case "$profile" in
  eng-smoke|behavior-smoke) ;;
  *) echo "unknown PROJECT3_TRAIN_PROFILE: ${profile} (eng-smoke|behavior-smoke)" >&2; exit 20 ;;
esac

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

train_batch_size="${PROJECT3_TRAIN_BATCH_SIZE:-66}"          # same default as official-loose
mini_batch_size=$((train_batch_size * 5))                    # group_n=5 samples
total_training_steps="${PROJECT3_TOTAL_TRAINING_STEPS:-1}"   # eng-smoke default: 1
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

if [[ "$profile" == "eng-smoke" ]]; then
  # 1-2 steps only; 1-2 step EM must never be read as an algorithm signal.
  if [[ "$total_training_steps" != "1" && "$total_training_steps" != "2" ]]; then
    echo "fail-closed: eng-smoke total_training_steps must be 1 or 2 (got: ${total_training_steps})" >&2
    exit 25
  fi
  save_freq="1"
fi
if [[ "$profile" == "behavior-smoke" ]]; then
  if [[ "$total_training_steps" != "5" && "$total_training_steps" != "10" ]]; then
    echo "fail-closed: behavior-smoke total_training_steps must be 5 or 10 (got: ${total_training_steps})" >&2
    exit 26
  fi
  if [[ "${PROJECT3_BEHAVIOR_SMOKE_APPROVED:-}" != "yes" ]]; then
    echo "fail-closed: behavior-smoke requires PROJECT3_BEHAVIOR_SMOKE_APPROVED=yes (engineering smoke must pass and the user must separately approve the GPU action first)" >&2
    exit 27
  fi
  save_freq="${PROJECT3_SAVE_FREQ:-5}"
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
  # v1 always starts from Qwen2.5-3B Base unless this is the eng-smoke resume
  # verification, which is pinned to the exact global_step_1 of a v1 eng-smoke.
  if [[ "$mode" != "dump-overrides" ]] && { [[ "$resume_from" != /* || ! -d "$resume_from/actor" || ! -f "$resume_from/data.pt" ]]; }; then
    echo "resume checkpoint must be an absolute global_step directory with actor/ and data.pt: ${resume_from}" >&2
    exit 17
  fi
  if [[ "$resume_from" != /* || ! "$(basename -- "$resume_from")" =~ ^global_step_[0-9]+$ ]]; then
    echo "resume checkpoint basename must be global_step_N: ${resume_from}" >&2
    exit 18
  fi
fi
if [[ -n "$resume_from" && "$(basename -- "$resume_from")" != "global_step_1" ]]; then
  echo "fail-closed: v1 resume is only the eng-smoke resume verification from global_step_1 (got: ${resume_from})" >&2
  exit 28
fi
if [[ -n "$resume_from" && "$total_training_steps" != "2" ]]; then
  echo "fail-closed: resume verification requires total_training_steps=2 (one new update, stops at global_step_2; got: ${total_training_steps})" >&2
  exit 29
fi

if [[ ! "$retriever_url" =~ ^http://127\.0\.0\.1:[0-9]{1,5}/retrieve$ ]]; then
  echo "retriever URL must be an IPv4 loopback /retrieve endpoint: ${retriever_url}" >&2
  exit 12
fi

# Self-check: resolved experimental values on the first stdout line.
echo "[V1_EXP] resolved: profile=${profile} train_batch_size=${train_batch_size} mini_batch_size=${mini_batch_size} total_training_steps=${total_training_steps} lr=${official_lr} lr_warmup_steps_ratio=${official_warmup_ratio} kl=${official_kl} gpu_mem=${gpu_memory_utilization} max_num_seqs=${max_num_seqs} offload_param=${offload_param} offload_optimizer=${offload_optimizer} offload_ref=${offload_ref} save_freq=${save_freq} seed=${trainer_seed}/${data_seed} resume_from=${resume_from:-<none>} search_aware_step_reward=true(env+reward_model) use_invalid_action_penalty=false adv_estimator=grpo"

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
  "actor_rollout_ref.rollout.n=1"           # fork hard constraint (GRPO group via env.rollout.n)
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "actor_rollout_ref.rollout.free_cache_engine=true"
  "actor_rollout_ref.rollout.enable_chunked_prefill=false"
  "actor_rollout_ref.rollout.max_num_batched_tokens=2304"
  "actor_rollout_ref.rollout.max_model_len=2304"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${offload_ref}"
  # v1 uses the official-loose projection (patch 0005) as its base semantics.
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
  # Phase 4B v1 switches (patch 0007): env computes per-step shaping, the
  # reward manager places them step-attributed and asserts sum consistency.
  # Both must be true together; the manager fails closed on mismatch.
  "+env.search_aware_step_reward=true"
  "+reward_model.search_aware_step_reward=true"
  # Phase 4B.1 (patch 0008): GRPO normalizes TRAJECTORY returns (sum of the
  # trajectory's step records) per uid instead of per record; the trajectory
  # advantage is broadcast back to all its records. main_ppo fails closed if
  # this is on without search_aware_step_reward.
  "+algorithm.search_v1_trajectory_return=true"
  # config-only penalty OFF: the env's own invalid/error penalty already lives
  # on the search step; ray_trainer's post-hoc subtraction would double-count.
  "actor_rollout_ref.actor.use_invalid_action_penalty=false"
  "trainer.logger=['console']"
  "trainer.project_name=search_r1_repro"
  "trainer.experiment_name=p3_grpo_v1_3b_fsdp6_n5_b${train_batch_size}_s0"
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
  # smoke profiles: warmup via ratio (verl default delegation).
  "actor_rollout_ref.actor.optim.lr_warmup_steps=-1"
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${official_warmup_ratio}"
)

if [[ -n "$resume_from" ]]; then
  overrides+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${resume_from}")
else
  overrides+=("trainer.resume_mode=disable")
fi

# Canonical fingerprint of the resolved overrides (sorted, stable): recorded in
# the run log; the wrapper's own config path is fingerprinted the same way.
config_fp="$(printf '%s\n' "${overrides[@]}" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "[V1_EXP] resolved_config_sha256=${config_fp}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export RAY_task_events_report_interval_ms=0
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
print(f"[V1_EXP] cpu_memory_gate report: {report}", flush=True)
available = mem["MemAvailable"]
swap_used = mem["SwapTotal"] - mem["SwapFree"]
if available < 64 * 1024**3:
    raise SystemExit("cpu memory gate failed: MemAvailable < 64GiB, aborting")
if available < 96 * 1024**3:
    raise SystemExit("cpu memory gate: MemAvailable in 64-96GiB band, pausing (no auto start)")
if swap_used > 2 * 1024**3:
    raise SystemExit(f"cpu memory gate failed: already swapping ({swap_used/1024**3:.1f}GiB used), aborting")
if disk_free < 100 * 1024**3:
    print(f"[V1_EXP] warning: checkpoint disk < 100GiB ({gib(disk_free):.0f}GiB), advisory only", flush=True)
print("[V1_EXP] cpu memory gate passed", flush=True)
PY

# 6 physical GPUs 1,2,3,4,6,7 only (GPU0 = desktop, GPU5 unstable); run_managed.sh
# gpu_guard rejects the forbidden pair too.
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1,2,3,4,6,7" ]]; then
  echo "P3 v1 gate requires run_managed.sh to expose physical GPUs 1,2,3,4,6,7 (got: ${CUDA_VISIBLE_DEVICES:-<unset>})" >&2
  exit 14
fi

for patch_file in \
  "${project_dir}/patches/0001-search-retrieval-status-observability.patch" \
  "${project_dir}/patches/0002-structured-rollout-audit.patch" \
  "${project_dir}/patches/0003-graceful-ray-shutdown-and-atomic-rollout.patch" \
  "${project_dir}/patches/0004-search-prompt-and-format-reward.patch" \
  "${project_dir}/patches/0005-search-env-loose-projection.patch" \
  "${project_dir}/patches/0006-segment-stop-step-decoupled-schedule-horizon.patch" \
  "${project_dir}/patches/0007-search-aware-step-reward.patch"; do
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
