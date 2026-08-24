#!/usr/bin/env bash
# Managed pure-evaluation wrapper for the P3 Search-aware clean v2 line
# (pristine 20bd331b + ONLY patches/v2/v2-0001..0007; vLLM-native backend).
# Independent script on purpose: the clean/official-line wrappers must never
# grow switches (line-split convention).
#
# v2 evaluation runs the CLEAN protocol (SEARCH_TEMPLATE prompts, upstream
# search_projection, skyrl compute_score format_score=0.0, max_steps=4,
# history_length=4) -- identical to run_p3_eval_upstream_clean.sh semantics,
# because the v2 patch series must not change prompt/projection/reward. The
# tree gate (--v2-dir + --pristine-dir) verifies the v2 tree is imported and
# the four protocol-critical files are byte-identical to pristine.
#
# Data modes:
#   smoke                  16-question pipeline gate (never a quality claim)
#   official-confirm256-v1 confirm-256 evaluation (the clean Step0 / GRPO10
#                           comparison set; used for the Step0 protocol-
#                           equivalence gate and the Step5 main evaluation)
#   dev64                  fixed 64-question behaviour-diagnosis set
#                           (searchr1-p3-dev64-v1); main eval greedy + 5-rollout
#                           sampling diagnosis use this set
#
# Usage (must run inside run_managed.sh via start_tmux_run.sh):
#   PROJECT3_EVAL_DATA=smoke|official-confirm256-v1|dev64
#   PROJECT3_EVAL_MODEL=<absolute path to full model dir>
#   PROJECT3_EVAL_TOKENIZER=<absolute path to tokenizer>  (default Qwen2.5-3B BASE)
#   PROJECT3_EVAL_TEMPERATURE=<0.0 greedy main | >0 diagnosis>  (default 0.0)
#   PROJECT3_EVAL_NUM_ROLLOUTS=<1 main | 5 diagnosis>           (default 1)
#   PROJECT3_EVAL_RETRIEVAL_CONDITION=<real|shuffled|no-evidence>  (default real)
#       Counterfactual evidence conditions (main mode only, confirm256 only):
#         shuffled    -- model's own query retrieved for real first; on success
#                        the evidence is replaced by the REAL docs of the fixed
#                        mapping (i + PROJECT3_EVAL_SHUFFLE_STEP) mod 256.
#                        Errors/empty/no-results of the real call are kept
#                        verbatim (never remapped); the run is pre-registered
#                        with the fixed mapping + SHA before any episode runs.
#         no-evidence -- every successful search returns the fixed neutral
#                        envelope; no retriever call; no invalid/error.
#   PROJECT3_EVAL_SHUFFLE_STEP=<int>    (default 17, fixed mapping offset)
#   PROJECT3_EVAL_NO_EVIDENCE_DOCS=<int> (default 3, neutral docs in envelope)
#   bash scripts/run_p3_eval_v2.sh
#
# Exit codes: 10 v2-tree commit mismatch, 11 missing paths, 12 retriever URL,
# 13 not managed, 14 GPU mapping, 15 v2 tree dirty / rebuild mismatch,
# 16 merged-model verification failed, 19 unknown eval data,
# 20 counterfactual condition outside its allowed scope (confirm256 main mode).
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
# v2 line: the vendored worktree (pristine 20bd331b + ONLY patches/v2/*).
vendor_dir="${project_dir}/vendor/verl-agent-v2"
# Pristine reference for the protocol byte-equality gate.
pristine_dir="${project_dir}/vendor/upstream-20bd331b"

expected_upstream="20bd331bdbc9026a5668e11362178e10ab7400c8"
actual_upstream="$(git -C "$vendor_dir" rev-parse HEAD 2>/dev/null || echo MISSING)"
if [[ "$actual_upstream" != "$expected_upstream" ]]; then
  echo "v2-tree commit mismatch: expected ${expected_upstream}, got ${actual_upstream}" >&2
  exit 10
fi

data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"
python_bin="${project_data}/envs/searchr1-repro-cu124/bin/python"
model_path="${PROJECT3_EVAL_MODEL:-${project_data}/models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo}"
tokenizer_path="${PROJECT3_EVAL_TOKENIZER:-${project_data}/models/Qwen2.5-3B}"
eval_data="${PROJECT3_EVAL_DATA:-smoke}"
eval_temperature="${PROJECT3_EVAL_TEMPERATURE:-0.0}"
eval_num_rollouts="${PROJECT3_EVAL_NUM_ROLLOUTS:-1}"
eval_retrieval_condition="${PROJECT3_EVAL_RETRIEVAL_CONDITION:-real}"
eval_shuffle_step="${PROJECT3_EVAL_SHUFFLE_STEP:-17}"
eval_no_evidence_docs="${PROJECT3_EVAL_NO_EVIDENCE_DOCS:-3}"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-eval-v2}"

case "$eval_data" in
  smoke)
    data_files="${project_data}/datasets/searchr1-smoke/test.parquet"
    manifest_path="${project_data}/datasets/searchr1-smoke/manifest.json"
    manifest_key="test"
    ;;
  official-confirm256-v1)
    data_files="${project_data}/datasets/searchr1-official-confirm256-v1/heldout.parquet"
    manifest_path="${project_data}/datasets/searchr1-official-confirm256-v1/manifest.json"
    manifest_key="heldout"
    ;;
  dev64)
    data_files="${project_data}/datasets/searchr1-p3-dev64-v1/dev64.parquet"
    manifest_path="${project_data}/datasets/searchr1-p3-dev64-v1/manifest.json"
    manifest_key="dev64"
    ;;
  *)
    echo "PROJECT3_EVAL_DATA must be smoke, official-confirm256-v1 or dev64, got: ${eval_data}" >&2
    exit 19
    ;;
esac

# Counterfactual conditions are defined ONLY for the confirm-256 greedy main
# pass (same question set, same model, strict pairing). Fail closed anywhere
# else; the python side independently re-checks main mode.
case "${eval_retrieval_condition}" in
  real|shuffled|no-evidence) ;;
  *)
    echo "PROJECT3_EVAL_RETRIEVAL_CONDITION must be real, shuffled or no-evidence, got: ${eval_retrieval_condition}" >&2
    exit 20
    ;;
esac
if [[ "${eval_retrieval_condition}" != "real" ]]; then
  if [[ "${eval_data}" != "official-confirm256-v1" ]]; then
    echo "counterfactual condition ${eval_retrieval_condition} requires PROJECT3_EVAL_DATA=official-confirm256-v1 (got ${eval_data})" >&2
    exit 20
  fi
  if [[ "${eval_temperature}" != "0.0" || "${eval_num_rollouts}" != "1" ]]; then
    echo "counterfactual condition ${eval_retrieval_condition} requires greedy main mode (temperature=0.0, num_rollouts=1), got ${eval_temperature}/${eval_num_rollouts}" >&2
    exit 20
  fi
fi

leakage_reference="${project_data}/datasets/searchr1-smoke/train.parquet"

for required_path in "$python_bin" "$model_path" "$tokenizer_path" "$data_files" "$manifest_path" "$leakage_reference"; do
  if [[ ! -e "$required_path" ]]; then
    echo "required path missing: ${required_path}" >&2
    exit 11
  fi
done

if [[ ! "$retriever_url" =~ ^http://127\.0\.0\.1:[0-9]{1,5}/retrieve$ ]]; then
  echo "retriever URL must be an IPv4 loopback /retrieve endpoint: ${retriever_url}" >&2
  exit 12
fi

if [[ -z "${PROJECT3_RUN_ID:-}" || -z "${PROJECT3_RUN_DIR:-}" ]]; then
  echo "evaluation must be launched through scripts/run_managed.sh" >&2
  exit 13
fi

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "P3 v2 eval gate requires run_managed.sh to expose only physical GPU1" >&2
  exit 14
fi

# --- v2 tree gate: rebuild pristine + patches/v2/v2-0001..0007 and diff the
# final state against the vendor worktree (deterministic rebuild proof).
verify_scratch="$(mktemp -d /tmp/p3v2evalpatch.XXXXXX)"
cleanup_patch_verify() {
  if [[ -n "${verify_scratch:-}" &&
        "$verify_scratch" == /tmp/p3v2evalpatch.* &&
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
# --- end v2 tree gate ---

# Merged-model gate: must be a complete, loadable, NaN-free merged model.
verify_output="$("$python_bin" "${script_dir}/verify_p3_merged_model.py" --merged-dir "$model_path" 2>&1)" || {
  echo "merged-model verification failed: ${verify_output}" >&2
  exit 16
}
if [[ "$verify_output" != *"VERIFY_MERGED: PASS"* ]]; then
  echo "merged-model gate did not print VERIFY_MERGED: PASS: ${verify_output}" >&2
  exit 16
fi

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

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# Same vLLM engine path as the training rollout (V0 engine).
export VLLM_USE_V1=0
# Local-only stack: model files, dataset and retriever all live on loopback.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
# v2 line: the vendored worktree shadows the patched official vendor.
export PYTHONPATH="${vendor_dir}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

# --- nvidia-smi per-GPU PHYSICAL peak sampler (2s cadence, one logical GPU).
# This is the authoritative physical-peak number; the script's torch values
# are torch-allocator views and must be reported separately (never conflated).
peak_file="${PROJECT3_RUN_DIR}/peak_memory_nvidia_smi.json"
"$python_bin" - "$peak_file" <<'PY' &
import json
import signal
import subprocess
import sys
import time

peak_file = sys.argv[1]
peaks = {}
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
            if index != "1":
                continue  # eval is pinned to physical GPU1
            peaks[index] = max(peaks.get(index, 0), used)
        time.sleep(2)
except Exception as exc:  # never mask the evaluation's own exit code
    with open(peak_file, "w") as handle:
        json.dump({"error": str(exc), "peaks_mib": peaks}, handle)
    raise SystemExit(0)
finally:
    with open(peak_file, "w") as handle:
        json.dump(
            {
                "source": "nvidia-smi sampler (2s cadence)",
                "per_gpu_physical_peak_mib": peaks,
                "gpu_total_mib": total if 'total' in dir() else None,
                "note": "nvidia-smi physical peaks; torch.cuda.max_memory_* in results.json are torch-allocator views, NOT physical peaks",
            },
            handle,
        )
    sys.exit(0)
PY
sampler_pid=$!

"$python_bin" "${script_dir}/run_p3_eval_v2.py" \
  --v2-dir "$vendor_dir" \
  --pristine-dir "$pristine_dir" \
  --model "$model_path" \
  --tokenizer "$tokenizer_path" \
  --data-files "$data_files" \
  --manifest "$manifest_path" \
  --manifest-key "$manifest_key" \
  --leakage-reference "$leakage_reference" \
  --search-url "$retriever_url" \
  --max-steps 4 --history-length 4 --topk 3 --timeout 180 \
  --max-input-tokens 3072 --max-new-tokens 256 --seed 0 \
  --temperature "$eval_temperature" --num-rollouts "$eval_num_rollouts" \
  --retrieval-condition "$eval_retrieval_condition" \
  --shuffle-step "$eval_shuffle_step" --no-evidence-docs "$eval_no_evidence_docs" \
  --max-envs-per-batch 24
status=$?

# stop the nvidia-smi sampler (final write lands in its finally block)
kill "$sampler_pid" 2>/dev/null || true
wait "$sampler_pid" 2>/dev/null || true

exit "$status"
