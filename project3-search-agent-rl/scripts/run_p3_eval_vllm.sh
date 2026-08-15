#!/usr/bin/env bash
# Managed pure-evaluation wrapper for P3 (vLLM-native greedy backend).
# Inference only: the underlying script constructs no optimizer, no scheduler,
# no backward, no Ray. All gates mirror run_p3_eval_heldout.sh, and the engine
# rides the training-rollout vLLM path (VLLM_USE_V1=0, bfloat16,
# gpu_memory_utilization 0.6, enforce_eager, max_model_len 2304).
#
# Usage (must run inside run_managed.sh via start_tmux_run.sh):
#   PROJECT3_EVAL_DATA=smoke|heldout32
#   PROJECT3_EVAL_ADAPTER=<absolute path to lora_adapter dir>   (unset => base model)
#   bash scripts/run_p3_eval_vllm.sh
#
# Exit codes: 10 commit mismatch, 11 missing paths, 12 retriever URL,
# 13 not managed, 14 GPU mapping, 15 patch not applied, 19 unknown eval data,
# 20 invalid adapter.
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
model_path="${PROJECT3_EVAL_MODEL:-${project_data}/models/Qwen2.5-1.5B-Instruct}"
adapter_dir="${PROJECT3_EVAL_ADAPTER:-}"
eval_data="${PROJECT3_EVAL_DATA:-smoke}"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-heldout-eval-vllm}"

case "$eval_data" in
  smoke)
    data_files="${project_data}/datasets/searchr1-smoke/test.parquet"
    manifest_path="${project_data}/datasets/searchr1-smoke/manifest.json"
    manifest_key="test"
    ;;
  heldout32)
    data_files="${project_data}/datasets/searchr1-heldout32/heldout.parquet"
    manifest_path="${project_data}/datasets/searchr1-heldout32/manifest.json"
    manifest_key="heldout"
    ;;
  confirm256)
    data_files="${project_data}/datasets/searchr1-confirm256/heldout.parquet"
    manifest_path="${project_data}/datasets/searchr1-confirm256/manifest.json"
    manifest_key="heldout"
    ;;
  *)
    echo "PROJECT3_EVAL_DATA must be smoke, heldout32 or confirm256, got: ${eval_data}" >&2
    exit 19
    ;;
esac
leakage_reference="${project_data}/datasets/searchr1-smoke/train.parquet"

for required_path in "$python_bin" "$model_path" "$data_files" "$manifest_path" "$leakage_reference"; do
  if [[ ! -e "$required_path" ]]; then
    echo "required path missing: ${required_path}" >&2
    exit 11
  fi
done

if [[ -n "$adapter_dir" ]]; then
  if [[ "$adapter_dir" != /* || ! -f "$adapter_dir/adapter_config.json" || ! -f "$adapter_dir/adapter_model.safetensors" ]]; then
    echo "eval adapter must be an absolute PEFT lora_adapter directory: ${adapter_dir}" >&2
    exit 20
  fi
fi

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

for patch_file in \
  "${project_dir}/patches/0001-search-retrieval-status-observability.patch" \
  "${project_dir}/patches/0002-structured-rollout-audit.patch" \
  "${project_dir}/patches/0003-graceful-ray-shutdown-and-atomic-rollout.patch" \
  "${project_dir}/patches/0004-search-prompt-and-format-reward.patch"; do
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

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
# Same vLLM engine path as the training rollout (run_p3_grpo_fix_exp.sh);
# the eval script aborts unless VLLM_USE_V1 is exactly "0".
export VLLM_USE_V1=0
# Local-only stack: model files, dataset and retriever all live on loopback.
# The tmux server env may carry http(s)_proxy (e.g. a system proxy on 7890);
# requests/httpx then route loopback traffic through the proxy and every
# search times out (observed 2026-08-15 on confirm-256). Managed evals must
# never traverse a proxy, independent of the launching shell's environment.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export PYTHONPATH="${vendor_dir}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  --model "$model_path"
  --data-files "$data_files"
  --manifest "$manifest_path"
  --manifest-key "$manifest_key"
  --leakage-reference "$leakage_reference"
  --search-url "$retriever_url"
  --max-steps 2 --history-length 2 --topk 3 --timeout 180
  --max-input-tokens 2048 --max-new-tokens 256 --seed 0
)
if [[ -n "$adapter_dir" ]]; then
  args+=(--adapter "$adapter_dir")
fi

exec "$python_bin" "${script_dir}/run_p3_eval_vllm.py" "${args[@]}"
