#!/usr/bin/env bash
# Managed pure-evaluation wrapper for the P3 CLEAN-UPSTREAM line
# (20bd331b, patches 0001-0008 NOT applied; vLLM-native greedy backend).
# Independent script on purpose: the official-line wrapper must never grow
# switches (line-split convention, docs/P3_EXPERIMENT_LINES_2026-08-15.md).
#
# Clean-upstream semantics: SearchEnvironmentManager (question + search history
# via SearchMemory) + upstream search_projection + single-layer official Search
# prompt (SEARCH_TEMPLATE_NO_HIS / SEARCH_TEMPLATE). max_steps=4, history_length=4,
# topk=3. Terminal reward = upstream skyrl compute_score format_score=0.0;
# EM = reward >= 1.0. No custom Reward, no trajectory-return modifications.
#
# Usage (must run inside run_managed.sh via start_tmux_run.sh):
#   PROJECT3_EVAL_DATA=smoke|official-confirm256-v1
#   PROJECT3_EVAL_MODEL=<absolute path to full model dir>   (default: official 3B GRPO merged checkpoint)
#   PROJECT3_EVAL_TOKENIZER=<absolute path to tokenizer>    (default: Qwen2.5-3B BASE tokenizer)
#   bash scripts/run_p3_eval_upstream_clean.sh
#
# --tokenizer is pinned to the Qwen2.5-3B BASE tokenizer so that Base and the
# official GRPO checkpoint receive byte-identical input token ids (the official
# checkpoint's own tokenizer_config embeds a tools-flavoured chat_template).
#
# Exit codes: 10 clean-tree commit mismatch, 11 missing paths, 12 retriever URL,
# 13 not managed, 14 GPU mapping, 15 clean tree dirty / patch markers present,
# 16 merged-model verification failed, 19 unknown eval data.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
# CLEAN line: the pristine upstream worktree (NO patches 0001-0008).
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

data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"
python_bin="${project_data}/envs/searchr1-repro-cu124/bin/python"
model_path="${PROJECT3_EVAL_MODEL:-${project_data}/models/p3-formal-segment-100-300-gs300-merged-20260817b}"
# Fixed: the BASE tokenizer renders inputs for BOTH models (see above).
tokenizer_path="${PROJECT3_EVAL_TOKENIZER:-${project_data}/models/Qwen2.5-3B}"
eval_data="${PROJECT3_EVAL_DATA:-smoke}"
retriever_url="${PROJECT3_RETRIEVER_URL:-http://127.0.0.1:18080/retrieve}"
run_dir="${PROJECT3_RUN_DIR:-${project_data}/dry-run/p3-eval-upstream-clean}"

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
  *)
    echo "PROJECT3_EVAL_DATA must be smoke or official-confirm256-v1, got: ${eval_data}" >&2
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

# Merged-model gate: the official 3B GRPO checkpoint must be a complete,
# loadable, NaN-free merged model (verify_p3_merged_model.py prints
# "VERIFY_MERGED: PASS").
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
# CLEAN line: the pristine upstream tree must shadow the patched vendor.
export PYTHONPATH="${vendor_dir}:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$python_bin" "${script_dir}/run_p3_eval_upstream_clean.py" \
  --upstream-dir "$vendor_dir" \
  --model "$model_path" \
  --tokenizer "$tokenizer_path" \
  --data-files "$data_files" \
  --manifest "$manifest_path" \
  --manifest-key "$manifest_key" \
  --leakage-reference "$leakage_reference" \
  --search-url "$retriever_url" \
  --max-steps 4 --history-length 4 --topk 3 --timeout 180 \
  --max-input-tokens 3072 --max-new-tokens 256 --seed 0 \
  --max-envs-per-batch 24
