#!/usr/bin/env bash
# P3 Search-aware clean v2 training entry (2026-08-22 user directive,
# docs/P3_SEARCH_AWARE_GRPO10_RESULT_REPORT_2026-08-22.md -- v1 judged FAILED
# direction C; THIS is the clean-baseline-reset v2 line).
#
# Baseline contract: the v2 tree (vendor/verl-agent-v2 worktree @20bd331b) is
# pristine upstream + ONLY the v2 patch series patches/v2/v2-0001..0007. The
# v1 semantics (0004 prompt rewrite, 0005 loose projection, 0007/0008 v1
# reward) are NOT applied. Clean protocol restored byte-for-byte:
#
#   upstream SEARCH_TEMPLATE_NO_HIS / SEARCH_TEMPLATE prompts (prompts/search.py)
#   upstream search_projection (projection.py, unmodified)
#   upstream skyrl compute_score format_score=0.0 (utils.py, unmodified)
#   SearchEnv termination / max_steps=4 / history_length=4 (clean GRPO10 line)
#   Qwen2.5-3B-Instruct starting model (fresh start, same as clean Step0)
#
# v2 switches ON (reward/advantage/audit ONLY, never the protocol):
#   R = R_answer + 0.15*first_evidence_hit + 0.30*sce
#       - 0.20*invalid_or_error - 0.20*true_redundant_search
#       - 0.20*new_answer_leak_in_query     (format_score stays 0.0)
#   TRUE redundancy (frozen): duplicate normalized query OR no new document ID
#   (content-hash fallback) -- the first search and new-evidence searches are
#   NEVER redundant (v1's "2nd+ search always redundant" is gone).
#   Step attribution: shaping on the search step, R_answer+sce on the terminal
#   step; Observation tokens never enter the policy loss (mask unchanged);
#   8-component per-trajectory audit == placed sum == trajectory return
#   (integer cents, fail-closed); GRPO normalizes the 5 trajectory returns per
#   uid and broadcasts the trajectory advantage to all its records.
#   use_invalid_action_penalty=false (config-only, like v1): the env's own
#   invalid penalty already lives on the search step; the post-hoc config
#   subtraction would break the sum-consistency invariant.
#
# Profiles (PROJECT3_TRAIN_PROFILE):
#   eng-smoke: exactly 1-2 steps (PROJECT3_TOTAL_TRAINING_STEPS in 1..2),
#     save_freq=1 -- engineering verification ONLY (VRAM, online v2 reward
#     computation + sum assertion, search trajectories, checkpoint). NEVER
#     judged by 1-2 step EM.
#   behavior: exactly 5 steps (the v2 5-step behavior experiment), save_freq=1
#     so every step's checkpoint + rollout audit are preserved. Fresh start
#     from Step0 -- must NOT resume from the eng-smoke run.
#   ten-step: exactly 10 steps (the v2 GRPO10 continuation, 2026-08-23
#     directive, gated on the counterfactual evidence-usage gate PASS).
#     save_freq=5 (checkpoints at gs5 + gs10), fresh from Qwen2.5-3B-Instruct
#     Step0 (resume forbidden), config fingerprint fixed to the v2 Step5
#     reference (same optimizer/env/reward/v2 switches), warmup ratio 0.285
#     (actual warmup steps recorded per step in the rollout audit). Requires
#     PROJECT3_TEN_STEP_APPROVED=yes (fail-closed approval env).
#
# Every GPU launch must go through scripts/run_managed.sh (PROJECT3_RUN_ID /
# PROJECT3_RUN_DIR required below); physical GPUs 1,2,3,4,6,7 only.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
# v2 line: the vendored worktree at pristine 20bd331b + ONLY patches/v2/*.
# (v2-0001..0006 from the 2026-08-22 directive; v2-0007 = duplicate-record
# source fix 2026-08-23.)
vendor_dir="${project_dir}/vendor/verl-agent-v2"

expected_upstream="20bd331bdbc9026a5668e11362178e10ab7400c8"
actual_upstream="$(git -C "$vendor_dir" rev-parse HEAD 2>/dev/null || echo MISSING)"
if [[ "$actual_upstream" != "$expected_upstream" ]]; then
  echo "v2-tree commit mismatch: expected ${expected_upstream}, got ${actual_upstream}" >&2
  exit 10
fi

data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"
python_bin="${project_data}/envs/searchr1-repro-cu124/bin/python"
# Starting model: Qwen2.5-3B-Instruct (the clean Step0 / GRPO10 / GiGPO10
# baseline start; v2 must start from the SAME Step0 for the protocol gate).
model_path="${PROJECT3_MODEL_PATH:-${project_data}/models/Qwen2.5-3B-Instruct}"
train_parquet="${project_data}/datasets/searchr1-upstream/train.parquet"
val_dir="${PROJECT3_VAL_DIR:-${project_data}/datasets/searchr1-smoke}"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-grpo-v2}"
resume_from="${PROJECT3_RESUME_FROM:-}"

profile="${PROJECT3_TRAIN_PROFILE:-eng-smoke}"
case "$profile" in
  eng-smoke|behavior|ten-step) ;;
  *) echo "unknown PROJECT3_TRAIN_PROFILE: ${profile} (eng-smoke|behavior|ten-step)" >&2; exit 20 ;;
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

train_batch_size="${PROJECT3_TRAIN_BATCH_SIZE:-66}"          # clean GRPO10 default
mini_batch_size=$((train_batch_size * 5))                    # env.rollout.n=5
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
# Clean protocol (identical to the clean GRPO10/GiGPO10 line).
env_max_steps="${PROJECT3_ENV_MAX_STEPS:-4}"
env_history_length="${PROJECT3_ENV_HISTORY_LENGTH:-4}"

if [[ "$profile" == "eng-smoke" ]]; then
  if [[ "$total_training_steps" != "1" && "$total_training_steps" != "2" ]]; then
    echo "fail-closed: eng-smoke total_training_steps must be 1 or 2 (got: ${total_training_steps})" >&2
    exit 25
  fi
  save_freq="1"
fi
if [[ "$profile" == "behavior" ]]; then
  if [[ "$total_training_steps" != "5" ]]; then
    echo "fail-closed: behavior (5-step experiment) total_training_steps must be 5 (got: ${total_training_steps})" >&2
    exit 26
  fi
  if [[ "${PROJECT3_BEHAVIOR_APPROVED:-}" != "yes" ]]; then
    echo "fail-closed: behavior profile requires PROJECT3_BEHAVIOR_APPROVED=yes (eng-smoke must pass and the user must separately approve the 5-step GPU action first)" >&2
    exit 27
  fi
  save_freq="1"  # keep every step's checkpoint + rollout audit
fi
if [[ "$profile" == "ten-step" ]]; then
  if [[ "$total_training_steps" != "10" ]]; then
    echo "fail-closed: ten-step (v2 GRPO10) total_training_steps must be 10 (got: ${total_training_steps})" >&2
    exit 31
  fi
  if [[ "${PROJECT3_TEN_STEP_APPROVED:-}" != "yes" ]]; then
    echo "fail-closed: ten-step profile requires PROJECT3_TEN_STEP_APPROVED=yes (counterfactual evidence-usage gate must PASS and the user must approve the 10-step GPU action)" >&2
    exit 30
  fi
  if [[ "$save_freq" != "5" ]]; then
    echo "fail-closed: ten-step save_freq must be 5 (checkpoints at gs5 + gs10), got: ${save_freq}" >&2
    exit 33
  fi
  if [[ "$resume_from" != "" ]]; then
    echo "fail-closed: ten-step starts FRESH from Step0 (resume forbidden), got: ${resume_from}" >&2
    exit 32
  fi
  # model fixed to the clean Step0 start (the SAME model the 5-step run and
  # the clean GRPO10 line started from); the scheduler horizon differs from
  # the behavior run's, so resuming from a behavior checkpoint is out.
  reference_model="${project_data}/models/Qwen2.5-3B-Instruct"
  if [[ "$(readlink -f "$model_path")" != "$(readlink -f "$reference_model")" ]]; then
    echo "fail-closed: ten-step must start from Qwen2.5-3B-Instruct Step0 (got: ${model_path})" >&2
    exit 34
  fi
  # config fingerprint fixed to the v2 Step5 reference values: identical
  # optimizer / env / reward / v2 switches; any deviation aborts.
  if [[ "$train_batch_size" != "66" || "$gpu_memory_utilization" != "0.60" \
        || "$max_num_seqs" != "64" || "$official_lr" != "1e-6" \
        || "$official_kl" != "0.001" || "$official_warmup_ratio" != "0.285" \
        || "$env_max_steps" != "4" || "$env_history_length" != "4" ]]; then
    echo "fail-closed: ten-step config must equal the v2 Step5 reference except for an explicitly approved seed (train_batch_size=66, gpu_mem=0.60, max_num_seqs=64, lr=1e-6, kl=0.001, warmup=0.285, max_steps=4, history=4)" >&2
    exit 35
  fi
  if [[ "$trainer_seed" != "1234" || "$data_seed" != "1234" ]]; then
    if [[ "${PROJECT3_MULTI_SEED_APPROVED:-}" != "yes" ]]; then
      echo "fail-closed: non-reference ten-step seed requires PROJECT3_MULTI_SEED_APPROVED=yes" >&2
      exit 36
    fi
    if [[ ! "$trainer_seed" =~ ^[0-9]+$ || "$trainer_seed" != "$data_seed" ]]; then
      echo "fail-closed: approved multi-seed run requires one non-negative integer shared by trainer/data (got ${trainer_seed}/${data_seed})" >&2
      exit 37
    fi
  fi
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
  echo "fail-closed: the v2 5-step experiment starts FRESH from Step0; resume is not part of the v2 protocol (got: ${resume_from})" >&2
  exit 28
fi

if [[ ! "$retriever_url" =~ ^http://127\.0\.0\.1:[0-9]{1,5}/retrieve$ ]]; then
  echo "retriever URL must be an IPv4 loopback /retrieve endpoint: ${retriever_url}" >&2
  exit 12
fi

# Self-check: resolved experimental values on the first stdout line.
echo "[V2_EXP] resolved: profile=${profile} model_path=${model_path} train_batch_size=${train_batch_size} mini_batch_size=${mini_batch_size} total_training_steps=${total_training_steps} env_max_steps=${env_max_steps} env_history_length=${env_history_length} lr=${official_lr} lr_warmup_steps_ratio=${official_warmup_ratio} kl=${official_kl} gpu_mem=${gpu_memory_utilization} max_num_seqs=${max_num_seqs} offload_param=${offload_param} offload_optimizer=${offload_optimizer} offload_ref=${offload_ref} save_freq=${save_freq} seed=${trainer_seed}/${data_seed} resume_from=${resume_from:-<none>} search_aware_step_reward=true(env+reward_model) search_v1_trajectory_return=true use_invalid_action_penalty=false adv_estimator=grpo gamma=1.0 format_score=0.0(clean) projection=clean search_projection prompt=clean SEARCH_TEMPLATE"

overrides=(
  "algorithm.adv_estimator=grpo"
  "algorithm.gamma=1.0"          # explicit: identical to the clean GRPO10 line
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
  # FULL-PARAM FSDP mainline (verified topology, same as clean GRPO10).
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
  "actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=2304"
  "actor_rollout_ref.rollout.max_model_len=2304"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${offload_ref}"
  # Clean-upstream search env: upstream search_projection, skyrl compute_score
  # format_score=0.0, SEARCH_TEMPLATE prompts. NO env.projection override and
  # NO prompt/format patches -- the clean protocol is the v2 baseline.
  "env.env_name=search"
  "env.seed=0"
  "env.max_steps=${env_max_steps}"
  "env.history_length=${env_history_length}"
  "env.rollout.n=5"
  "env.search.search_url=${retriever_url}"
  "env.search.topk=3"
  "env.search.timeout=180"
  "env.search.log_requests=true"
  # v2 switches (patch v2-0004/v2-0005/v2-0006): env computes the per-step v2
  # shaping components; the reward manager places them step-attributed and
  # asserts sum-consistency (fail-closed); GRPO normalizes trajectory returns.
  # All three must be on together; main_ppo fails closed on mismatch.
  # v2-0006 pre-declares reward_model/algorithm keys (default false) so those
  # are plain assignments; env.search_aware_step_reward is a NEW top-level key
  # (the env factory propagates it into every per-env config), so it needs "+".
  "+env.search_aware_step_reward=true"
  "reward_model.search_aware_step_reward=true"
  "algorithm.search_v1_trajectory_return=true"
  # config-only penalty OFF: the env's own invalid/error penalty already lives
  # on the search step; ray_trainer's post-hoc subtraction would double-count
  # and break the 8-component sum == placed sum == trajectory return invariant.
  "actor_rollout_ref.actor.use_invalid_action_penalty=false"
  "trainer.logger=['console']"
  "trainer.project_name=search_r1_repro"
  "trainer.experiment_name=p3_search_aware_clean_v2_${profile}_fsdp6_n5_b${train_batch_size}_s${trainer_seed}"
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
  # smoke/behavior profiles: warmup via ratio (verl default delegation).
  "actor_rollout_ref.actor.optim.lr_warmup_steps=-1"
  "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${official_warmup_ratio}"
)

overrides+=("trainer.resume_mode=disable")

# Canonical fingerprint of the resolved overrides (sorted, stable): recorded in
# the run log; the wrapper's own config path is fingerprinted the same way.
config_fp="$(printf '%s\n' "${overrides[@]}" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "[V2_EXP] resolved_config_sha256=${config_fp}"

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export RAY_task_events_report_interval_ms=0
# v2 line: the vendored worktree must shadow the patched official vendor.
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
print(f"[V2_EXP] cpu_memory_gate report: {report}", flush=True)
available = mem["MemAvailable"]
swap_used = mem["SwapTotal"] - mem["SwapFree"]
if available < 64 * 1024**3:
    raise SystemExit("cpu memory gate failed: MemAvailable < 64GiB, aborting")
if available < 96 * 1024**3:
    raise SystemExit("cpu memory gate: MemAvailable in 64-96GiB band, pausing (no auto start)")
if swap_used > 2 * 1024**3:
    raise SystemExit(f"cpu memory gate failed: already swapping ({swap_used/1024**3:.1f}GiB used), aborting")
if disk_free < 100 * 1024**3:
    print(f"[V2_EXP] warning: checkpoint disk < 100GiB ({gib(disk_free):.0f}GiB), advisory only", flush=True)
print("[V2_EXP] cpu memory gate passed", flush=True)
PY

# 6 physical GPUs 1,2,3,4,6,7 only (GPU0 = desktop, GPU5 unstable); run_managed.sh
# gpu_guard rejects the forbidden pair too.
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1,2,3,4,6,7" ]]; then
  echo "P3 v2 gate requires run_managed.sh to expose physical GPUs 1,2,3,4,6,7 (got: ${CUDA_VISIBLE_DEVICES:-<unset>})" >&2
  exit 14
fi

# --- patch set gate (final-state rebuild verification) ---
# Rebuild pristine 20bd331b (git archive HEAD == pristine: the worktree's
# changes are uncommitted) + patches/v2/v2-0001..0007 in a scratch tree and
# diff the final state against the vendor worktree. This is the deterministic
# rebuild proof the v2 line requires (user directive: patches must rebuild the
# v2 tree from 20bd331b; chain reverse-check is not decidable, final-state
# rebuild is).
verify_scratch="$(mktemp -d /tmp/p3v2patch.XXXXXX)"

cleanup_patch_verify() {
  if [[ -n "${verify_scratch:-}" &&
        "$verify_scratch" == /tmp/p3v2patch.* &&
        -d "$verify_scratch" ]]; then
    rm -rf -- "$verify_scratch"
  fi
}
trap cleanup_patch_verify EXIT

patch_names=(
  v2-0001-search-retrieval-status-observability
  v2-0002-structured-rollout-audit
  v2-0003-graceful-ray-shutdown-and-atomic-rollout
  v2-0004-search-aware-clean-v2-step-reward
  v2-0005-v2-trajectory-return-and-question-passthrough
  v2-0006-v2-config-schema
  v2-0007-duplicate-record-source-fix
)

git -C "$vendor_dir" archive HEAD | tar -x -C "$verify_scratch" || {
  echo "failed to reconstruct pristine 20bd331b tree" >&2
  exit 15
}

for patch_name in "${patch_names[@]}"; do
  patch_file="$project_dir/patches/v2/${patch_name}.patch"
  [[ -f "$patch_file" ]] || {
    echo "missing required patch: $patch_file" >&2
    exit 15
  }

  (
    cd "$verify_scratch"
    git apply --check "$patch_file" &&
    git apply "$patch_file"
  ) || {
    echo "failed to replay required patch: $patch_name" >&2
    exit 15
  }
done

if ! diff -qr \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  "$verify_scratch" "$vendor_dir" >/dev/null; then
  echo "v2 vendor worktree does not match pristine 20bd331b + patches/v2/v2-0001..0007" >&2
  diff -qr \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='*.egg-info' \
    "$verify_scratch" "$vendor_dir" >&2 || true
  exit 15
fi

cleanup_patch_verify
trap - EXIT
unset verify_scratch
# --- end patch set gate ---

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

# --- nvidia-smi per-GPU PHYSICAL peak sampler (2s cadence, training GPUs).
# P3 v2 §8: physical per-GPU peaks are recorded here; the verl log's
# max_memory_reserved_gb values are worker-aggregated torch-allocator views and
# must never be reported as per-GPU physical peaks.
peak_file="${run_dir}/peak_memory_nvidia_smi.json"
"$python_bin" - "$peak_file" <<'PY' &
import json
import os
import signal
import subprocess
import sys
import time

peak_file = sys.argv[1]
visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "1,2,3,4,6,7").split(",") if d]
peaks = {}
totals = {}
started = time.monotonic()
stop_requested = False

def request_stop(_signum, _frame):
    global stop_requested
    stop_requested = True

signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
try:
    while not stop_requested and time.monotonic() - started < 14400:  # 4h cap
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            time.sleep(2)
            continue
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            index, used, total = parts[0], int(parts[1]), int(parts[2])
            if index in visible:
                peaks[index] = max(peaks.get(index, 0), used)
                totals[index] = total
        time.sleep(2)
except Exception as exc:  # never mask the training's own exit code
    with open(peak_file, "w") as handle:
        json.dump({"error": str(exc), "peaks_mib": peaks}, handle)
    raise SystemExit(0)
finally:
    with open(peak_file, "w") as handle:
        json.dump(
            {
                "source": "nvidia-smi sampler (2s cadence)",
                "per_gpu_physical_peak_mib": peaks,
                "gpu_total_mib": totals,
                "note": "nvidia-smi physical peaks; verl log max_memory_reserved_gb is a worker-aggregated torch-allocator view, NOT a physical peak",
            },
            handle,
        )
    sys.exit(0)
PY
sampler_pid=$!

"$python_bin" -m verl.trainer.main_ppo "${overrides[@]}"
status=$?

# stop the nvidia-smi sampler (final write lands in its finally block)
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true

exit "$status"
