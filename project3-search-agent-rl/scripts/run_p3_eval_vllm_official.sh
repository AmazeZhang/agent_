#!/usr/bin/env bash
# Managed pure-evaluation wrapper for the P3 OFFICIAL-LOOSE line
# (vLLM-native greedy backend). Independent script on purpose: the strict-fork
# line (run_p3_eval_vllm.sh) must never grow switches (line-split convention,
# docs/P3_EXPERIMENT_LINES_2026-08-15.md section 3.3).
#
# Official Search-R1 semantics: raw action straight into the vendored skyrl
# SearchEnv (no projection, no invalid penalty, format_score=0.1).
#
# Usage (must run inside run_managed.sh via start_tmux_run.sh):
#   PROJECT3_EVAL_DATA=smoke|official-confirm256-v1|final-confirm512
#   PROJECT3_EVAL_MODEL=<absolute path to full model dir>   (default: official GRPO checkpoint)
#   PROJECT3_EVAL_TOKENIZER=<absolute path to tokenizer>    (default: Qwen2.5-3B BASE tokenizer)
#   bash scripts/run_p3_eval_vllm_official.sh
#
# --tokenizer is pinned to the Qwen2.5-3B BASE tokenizer so that Base and the
# official GRPO checkpoint receive byte-identical input token ids (the official
# checkpoint's own tokenizer_config embeds a tools-flavoured chat_template).
#
# Exit codes: 10 commit mismatch, 11 missing paths, 12 retriever URL,
# 13 not managed, 14 GPU mapping, 15 patch not applied, 19 unknown eval data.
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
model_path="${PROJECT3_EVAL_MODEL:-${project_data}/models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo}"
# Fixed: the BASE tokenizer renders inputs for BOTH models (see above).
tokenizer_path="${PROJECT3_EVAL_TOKENIZER:-${project_data}/models/Qwen2.5-3B}"
eval_data="${PROJECT3_EVAL_DATA:-smoke}"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-heldout-eval-vllm-official}"

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
  final-confirm512)
    data_files="${project_data}/datasets/searchr1-final-confirm512/heldout.parquet"
    manifest_path="${project_data}/datasets/searchr1-final-confirm512/manifest.json"
    manifest_key="heldout"
    ;;
  *)
    echo "PROJECT3_EVAL_DATA must be smoke, official-confirm256-v1 or final-confirm512, got: ${eval_data}" >&2
    exit 19
    ;;
esac
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
  echo "P3 eval gate requires run_managed.sh to expose only physical GPU1" >&2
  exit 14
fi

# Patch set gate (final-state rebuild verification, same as the training
# wrappers): per-patch `git apply --reverse --check` is unsound for a CHAIN
# of patches (0007/0008 edit the same files as 0001/0002/0004 and shift their
# hunks, so reverse-check misreports applied patches). Rebuild upstream +
# 0001..0009 in a scratch tree and diff against the vendor worktree instead.
verify_scratch="$(mktemp -d /tmp/p3patch.XXXXXX)"
cleanup_patch_verify() {
  if [[ -n "${verify_scratch:-}" &&
        "$verify_scratch" == /tmp/p3patch.* &&
        -d "$verify_scratch" ]]; then
    rm -rf -- "$verify_scratch"
  fi
}
trap cleanup_patch_verify EXIT

patch_names=(
  0001-search-retrieval-status-observability
  0002-structured-rollout-audit
  0003-graceful-ray-shutdown-and-atomic-rollout
  0004-search-prompt-and-format-reward
  0005-search-env-loose-projection
  0006-segment-stop-step-decoupled-schedule-horizon
  0007-search-aware-step-reward
  0008-v1-trajectory-return-and-traj-audit
  0009-search-aware-config-schema
)

git -C "$vendor_dir" archive HEAD | tar -x -C "$verify_scratch" || {
  echo "failed to reconstruct vendor upstream tree" >&2
  exit 15
}

for patch_name in "${patch_names[@]}"; do
  patch_file="$project_dir/patches/${patch_name}.patch"
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
  echo "vendor worktree does not match upstream + patches 0001..0009" >&2
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
if payload.get("status") != "ready" or payload.get("vectors") != 21_015_324:
    raise SystemExit(f"retriever health gate failed: {payload}")
print(f"retriever health gate passed: {payload}")
PY

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# Same vLLM engine path as the training rollout (run_p3_grpo_fix_exp.sh).
export VLLM_USE_V1=0
# Local-only stack: model files, dataset and retriever all live on loopback.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export PYTHONPATH="${vendor_dir}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$python_bin" "${script_dir}/run_p3_eval_vllm_official.py" \
  --model "$model_path" \
  --tokenizer "$tokenizer_path" \
  --data-files "$data_files" \
  --manifest "$manifest_path" \
  --manifest-key "$manifest_key" \
  --leakage-reference "$leakage_reference" \
  --search-url "$retriever_url" \
  --max-steps 2 --history-length 2 --topk 3 --timeout 180 \
  --max-input-tokens 2048 --max-new-tokens 256 --seed 0 \
  --max-envs-per-batch 32
