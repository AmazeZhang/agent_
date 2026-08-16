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
#   formal (default): schedule horizon FIXED at total_training_steps=300
#     (patch 0006 decouples the 300-step LR schedule from segment stops;
#     SS10 of the design doc is resolved). Segment stop point REQUIRED via
#     PROJECT3_SEGMENT_STOP_STEP=50|100|300 with fail-closed resume rules:
#     stop=50 -> no resume (fresh start from Qwen2.5-3B Base); stop=100 ->
#     resume from global_step_50; stop=300 -> resume from global_step_100;
#     every other combination exits with an error (see fail-closed block).
#     Architecture pinned to the verified form (official-offload-smoke
#     2026-08-16): gpu_memory_utilization=0.60, max_num_seqs=64,
#     actor/optimizer/ref offload=true, save_freq=50, lr_warmup_steps=85
#     (== int(0.285 x 300)); the 0.45/no-offload form was empirically rejected
#     and must not be reintroduced here.
#   smoke: total_training_steps=1, save_freq=1 (produces a checkpoint for the
#     resume verification), gpu_memory_utilization=0.40 (first VRAM observation
#     point per user review; 0.45 next). Smoke is a verification tool only,
#     never a formal config.
#   official-offload-smoke: the verified 0.60 + offload + max_num_seqs=64
#     architecture, 1 step (see PROGRESS_SYNC 2026-08-16).
#   official-offload-resume-smoke: resume verification from global_step_1;
#     total_training_steps=2 (resume at global step 1 -> exactly one new update,
#     final global_step_2; never continues to step 3), save_freq=1,
#     PROJECT3_RESUME_FROM is REQUIRED and must point at .../global_step_1.
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
  smoke|formal|official-offload-smoke|official-offload-resume-smoke) ;;
  *) echo "unknown PROJECT3_TRAIN_PROFILE: ${profile} (smoke|formal|official-offload-smoke|official-offload-resume-smoke)" >&2; exit 20 ;;
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

train_batch_size="${PROJECT3_TRAIN_BATCH_SIZE:-66}"          # 66 formal default; 132 target
mini_batch_size=$((train_batch_size * 5))                    # group_n=5 samples
total_training_steps="${PROJECT3_TOTAL_TRAINING_STEPS:-300}"  # formal default: 300-step schedule horizon (fixed)
# Official Search-R1 hyperparams (reference commit 598e61bd1d36895726d28a8d06b3a15bed19f5d3,
# train_grpo.sh blob 119d348e90d7c082c3b635eee7e022c941d14a57, file sha256
# 203098948565e60caf90da93aeaab74759d8ad9df1449649c8f03425e64f7c66): lr=1e-6,
# lr_warmup_steps_ratio=0.285, kl low_var_kl coef 0.001. lr_warmup_steps must
# stay -1 (verl default) so the ratio takes effect; a concrete step count
# would shadow the ratio.
official_lr="${PROJECT3_OFFICIAL_LR:-1e-6}"
official_kl="${PROJECT3_OFFICIAL_KL_COEF:-0.001}"
official_warmup_ratio="${PROJECT3_OFFICIAL_WARMUP_RATIO:-0.285}"
trainer_seed="${PROJECT3_TRAINER_SEED:-1234}"
data_seed="${PROJECT3_DATA_SEED:-1234}"
# formal defaults = the verified-successful architecture (official-offload-smoke,
# 2026-08-16): 0.60 vLLM budget, max_num_seqs=64, full param/optimizer/ref offload.
gpu_memory_utilization="${PROJECT3_GPU_MEM_UTIL:-0.60}"
save_freq="${PROJECT3_SAVE_FREQ:-50}"
offload_param=true
offload_optimizer=true
offload_ref=true
max_num_seqs="${PROJECT3_MAX_NUM_SEQS:-64}"
if [[ "$profile" == "smoke" ]]; then
  total_training_steps="1"
  save_freq="1"
  gpu_memory_utilization="${PROJECT3_GPU_MEM_UTIL:-0.40}"
fi
if [[ "$profile" == "official-offload-smoke" ]]; then
  # Official-architecture adaptation smoke (independent line from the 0.40/0.45
  # single-variable experiments; see PROGRESS_SYNC 2026-08-16): actor params +
  # optimizer and ref params offloaded to CPU (fork path:
  # verl/workers/fsdp_workers.py:142-146, 452, 553-560 via
  # verl/utils/fsdp_utils.py offload_fsdp_model_to_cpu / offload_fsdp_optimizer,
  # FSDP1 manual offload; no grad_offload field exists in this fork), vLLM gets
  # 0.60, max_num_seqs capped at 64 for 6x24GB x 330 samples.
  total_training_steps="1"
  save_freq="1"
  gpu_memory_utilization="${PROJECT3_GPU_MEM_UTIL:-0.60}"
  max_num_seqs="${PROJECT3_MAX_NUM_SEQS:-64}"
  offload_param=true
  offload_optimizer=true
  offload_ref=true
fi
if [[ "$profile" == "official-offload-resume-smoke" ]]; then
  # Resume verification (design sequence step 4): same verified architecture as
  # official-offload-smoke, but total_training_steps=2 so that resuming from
  # global_step_1 performs exactly ONE new update and stops at global_step_2
  # (no automatic continuation to step 3). PROJECT3_RESUME_FROM is required and
  # pinned to the source global_step_1 directory (validated below).
  total_training_steps="2"
  save_freq="1"
  gpu_memory_utilization="${PROJECT3_GPU_MEM_UTIL:-0.60}"
  max_num_seqs="${PROJECT3_MAX_NUM_SEQS:-64}"
  offload_param=true
  offload_optimizer=true
  offload_ref=true
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
  # --dump-overrides is freeze tooling: the fail-closed stop/resume rules still
  # apply, but the source checkpoint may not exist yet (segments resume from
  # checkpoints produced by the PREVIOUS segment), so only the format is checked.
  if [[ "$mode" != "dump-overrides" ]] && { [[ "$resume_from" != /* || ! -d "$resume_from/actor" || ! -f "$resume_from/data.pt" ]]; }; then
    echo "resume checkpoint must be an absolute global_step directory with actor/ and data.pt: ${resume_from}" >&2
    exit 17
  fi
  if [[ "$resume_from" != /* || ! "$(basename -- "$resume_from")" =~ ^global_step_[0-9]+$ ]]; then
    echo "resume checkpoint basename must be global_step_N: ${resume_from}" >&2
    exit 18
  fi
fi
if [[ "$profile" == "official-offload-resume-smoke" ]]; then
  # This profile is a resume-verification tool: the source must be the exact
  # global_step_1 directory of the official-offload-smoke run (absolute path,
  # read-only; the wrapper never writes into it -- default_local_dir is the new
  # run's own directory).
  if [[ -z "$resume_from" ]]; then
    echo "profile official-offload-resume-smoke requires PROJECT3_RESUME_FROM=<absolute global_step_1 dir>" >&2
    exit 23
  fi
  if [[ "$(basename -- "$resume_from")" != "global_step_1" ]]; then
    echo "profile official-offload-resume-smoke requires resume source .../global_step_1 (got: ${resume_from})" >&2
    exit 24
  fi
fi

# ---- formal segment-stop fail-closed validation (patch 0006) ----
# Segment stops (50/100/300) are decoupled from the schedule horizon (300, fixed):
# trainer.segment_stop_step only ends this segment after saving a checkpoint with
# DataLoader/Optimizer/Scheduler/RNG state; the LR scheduler is always built for
# 300 steps with warmup 85. Every combination other than the fixed table below
# exits with an error -- formal never silently changes its stop/resume semantics.
segment_stop_step="${PROJECT3_SEGMENT_STOP_STEP:-}"
resume_step=""
if [[ -n "$resume_from" ]]; then
  resume_step="$(basename -- "$resume_from")"
  resume_step="${resume_step#global_step_}"
fi
if [[ "$profile" == "formal" ]]; then
  if [[ -z "$segment_stop_step" ]]; then
    echo "fail-closed: formal requires PROJECT3_SEGMENT_STOP_STEP=50|100|300 (segment stop point; schedule horizon fixed at 300)" >&2
    exit 30
  fi
  case "$segment_stop_step" in
    50|100|300) ;;
    *) echo "fail-closed: PROJECT3_SEGMENT_STOP_STEP must be one of 50|100|300 (got: ${segment_stop_step})" >&2; exit 31 ;;
  esac
  if [[ "$segment_stop_step" == "50" ]]; then
    if [[ -n "$resume_from" ]]; then
      echo "fail-closed: segment stop 50 forbids resume (formal Step 0-50 starts from Qwen2.5-3B Base; got PROJECT3_RESUME_FROM=${resume_from})" >&2
      exit 32
    fi
  elif [[ "$segment_stop_step" == "100" ]]; then
    if [[ -z "$resume_from" ]]; then
      echo "fail-closed: segment stop 100 requires resume from global_step_50 (got: no PROJECT3_RESUME_FROM)" >&2
      exit 33
    fi
    if [[ "$resume_step" != "50" ]]; then
      echo "fail-closed: segment stop 100 requires resume from global_step_50 (got: ${resume_from})" >&2
      exit 34
    fi
  elif [[ "$segment_stop_step" == "300" ]]; then
    if [[ -z "$resume_from" ]]; then
      echo "fail-closed: segment stop 300 requires resume from global_step_100 (got: no PROJECT3_RESUME_FROM)" >&2
      exit 35
    fi
    if [[ "$resume_step" != "100" ]]; then
      echo "fail-closed: segment stop 300 requires resume from global_step_100 (got: ${resume_from})" >&2
      exit 36
    fi
  fi
  # generic: resume step must be strictly before the stop point
  if [[ -n "$resume_step" ]] && (( 10#$resume_step >= 10#$segment_stop_step )); then
    echo "fail-closed: resume step ${resume_step} >= segment stop ${segment_stop_step} (no training left in this segment)" >&2
    exit 37
  fi
  # schedule horizon pinned to 300 (segment control is segment_stop_step only)
  if [[ "$total_training_steps" != "300" ]]; then
    echo "fail-closed: formal schedule horizon pinned to total_training_steps=300 (got PROJECT3_TOTAL_TRAINING_STEPS=${total_training_steps}); segment stops are controlled by PROJECT3_SEGMENT_STOP_STEP only" >&2
    exit 38
  fi
  # architecture pinned to the verified form (official-offload-smoke 2026-08-16)
  if [[ "$gpu_memory_utilization" != "0.60" || "$max_num_seqs" != "64" ]]; then
    echo "fail-closed: formal architecture pinned to gpu_mem=0.60 + max_num_seqs=64 (got ${gpu_memory_utilization}/${max_num_seqs})" >&2
    exit 39
  fi
  if [[ "$offload_param" != "true" || "$offload_optimizer" != "true" || "$offload_ref" != "true" ]]; then
    echo "fail-closed: formal architecture pinned to param/optimizer/ref offload=true (got ${offload_param}/${offload_optimizer}/${offload_ref})" >&2
    exit 40
  fi
  if [[ "$save_freq" != "50" ]]; then
    echo "fail-closed: formal save_freq pinned to 50 (checkpoints align to segment stops; got ${save_freq})" >&2
    exit 41
  fi
elif [[ -n "$segment_stop_step" ]]; then
  echo "fail-closed: PROJECT3_SEGMENT_STOP_STEP is formal-only (profile=${profile}); smoke profiles stop via total_training_steps" >&2
  exit 42
fi

if [[ ! "$retriever_url" =~ ^http://127\.0\.0\.1:[0-9]{1,5}/retrieve$ ]]; then
  echo "retriever URL must be an IPv4 loopback /retrieve endpoint: ${retriever_url}" >&2
  exit 12
fi

# Self-check: resolved experimental values on the first stdout line.
if [[ "$profile" == "formal" ]]; then
  warmup_display="85"     # formal: fixed == int(0.285 x 300)
  ratio_display="n/a"
else
  warmup_display="-1"
  ratio_display="${official_warmup_ratio}"
fi
echo "[OFFICIAL_EXP] resolved: profile=${profile} train_batch_size=${train_batch_size} mini_batch_size=${mini_batch_size} schedule_total_steps=${total_training_steps} segment_stop_step=${segment_stop_step:-<none>} lr=${official_lr} lr_warmup_steps=${warmup_display} lr_warmup_steps_ratio=${ratio_display} kl=${official_kl} gpu_mem=${gpu_memory_utilization} max_num_seqs=${max_num_seqs:--} offload_param=${offload_param} offload_optimizer=${offload_optimizer} offload_ref=${offload_ref} save_freq=${save_freq} seed=${trainer_seed}/${data_seed} resume_from=${resume_from:-<none>}"

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
  "actor_rollout_ref.ref.fsdp_config.param_offload=${offload_ref}"
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

if [[ -n "$max_num_seqs" ]]; then
  overrides+=("actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs}")
fi

if [[ "$profile" == "formal" ]]; then
  # formal: warmup fixed at 85 steps (== int(0.285 x 300)); a positive
  # lr_warmup_steps takes priority in fsdp_workers.py:371-383, so the ratio is
  # never consulted and the LR schedule is identical across the 0-50/50-100/100-300
  # segments regardless of the segment stop point.
  overrides+=("actor_rollout_ref.actor.optim.lr_warmup_steps=85")
  # segment stop point (patch 0006): trainer stops after this global step,
  # saving checkpoint (DataLoader/Optimizer/Scheduler/RNG) and returning normally.
  overrides+=("trainer.segment_stop_step=${segment_stop_step}")
else
  # smoke profiles: warmup via ratio (verl default delegation); total_training_steps
  # still bounds these short engineering runs directly.
  overrides+=(
    "actor_rollout_ref.actor.optim.lr_warmup_steps=-1"
    "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${official_warmup_ratio}"
  )
fi

if [[ -n "$resume_from" ]]; then
  overrides+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${resume_from}")
else
  overrides+=("trainer.resume_mode=disable")
fi

# Canonical fingerprint of the resolved overrides (sorted, so it is stable
# independent of insertion order); recorded in the run log for the report.
config_fp="$(printf '%s\n' "${overrides[@]}" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)"
echo "[OFFICIAL_EXP] resolved_config_sha256=${config_fp}"

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

if [[ "$mode" == "dump-overrides" ]]; then
  # Machine-readable override list + canonical fingerprint (same computation as
  # the run path); used by the freeze tooling to derive full and
  # training-invariant SHAs from this script's own code path.
  printf '%s\n' "${overrides[@]}"
  echo "__config_fp__=${config_fp}"
  exit 0
fi

if [[ -z "${PROJECT3_RUN_ID:-}" || -z "${PROJECT3_RUN_DIR:-}" ]]; then
  echo "actual training must be launched through scripts/run_managed.sh" >&2
  exit 13
fi

# CPU memory / swap gate (offload needs tens of GiB of CPU RAM; the retriever
# index is ~64.5GB on disk and its process RSS must not push the machine into
# swap). MemAvailable >= 96GiB -> proceed; 64-96GiB -> pause, report, abort;
# < 64GiB -> abort. Already-noticeable swap (> 2GiB used) -> abort.
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
print(f"[OFFICIAL_EXP] cpu_memory_gate report: {report}", flush=True)
available = mem["MemAvailable"]
swap_used = mem["SwapTotal"] - mem["SwapFree"]
if available < 64 * 1024**3:
    raise SystemExit("cpu memory gate failed: MemAvailable < 64GiB, aborting")
if available < 96 * 1024**3:
    raise SystemExit("cpu memory gate: MemAvailable in 64-96GiB band, pausing (no auto start)")
if swap_used > 2 * 1024**3:
    raise SystemExit(f"cpu memory gate failed: already swapping ({swap_used/1024**3:.1f}GiB used), aborting")
if disk_free < 100 * 1024**3:
    print(f"[OFFICIAL_EXP] warning: checkpoint disk < 100GiB ({gib(disk_free):.0f}GiB), advisory only", flush=True)
print("[OFFICIAL_EXP] cpu memory gate passed", flush=True)
PY

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
  "${project_dir}/patches/0005-search-env-loose-projection.patch" \
  "${project_dir}/patches/0006-segment-stop-step-decoupled-schedule-horizon.patch"; do
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
# the rate-limit config chosen by the stress matrix (threads=8, limit=64), the
# only config under which the 330-env burst stays inside the 180s timeout.
if (
    payload.get("status") != "ready"
    or payload.get("vectors") != 21_015_324
    or payload.get("max_concurrent_queries") != 64
):
    raise SystemExit(f"retriever health gate failed: {payload}")
print(f"retriever health gate passed: {payload}")
PY

exec "$python_bin" -m verl.trainer.main_ppo "${overrides[@]}"
