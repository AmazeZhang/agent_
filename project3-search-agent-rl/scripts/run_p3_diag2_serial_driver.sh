#!/usr/bin/env bash
# Phase 4A diagnostic 2 + Step300 backfill driver: run serially on GPU1.
#
# 12 counterfactual runs (3 models x 4 conditions, one GPU1 managed tmux run
# each) + Step300 dev256 standard eval backfill. Each run launches through
# start_tmux_run.sh (managed gates: GPU1-only, upstream pin, patches,
# retriever health); the driver waits for each tmux session to exit before
# launching the next. Requires the shared evidence cache
# (diag_cache/dev256_top10_docs.json) to exist -- build it first with:
#   python scripts/run_p3_eval_counterfactual.py --cache-only
#
# Usage: bash scripts/run_p3_diag2_serial_driver.sh
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"
cache_file="${project_data}/diag_cache/dev256_top10_docs.json"

if [[ ! -f "$cache_file" ]]; then
  echo "evidence cache missing: ${cache_file} (build it first with --cache-only)" >&2
  exit 11
fi

BASE_MODEL="${project_data}/models/Qwen2.5-3B"
GS300_MODEL="${project_data}/models/p3-formal-segment-100-300-gs300-merged-20260817b"
OFFICIAL_MODEL="${project_data}/models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo"
for p in "$BASE_MODEL" "$GS300_MODEL" "$OFFICIAL_MODEL"; do
  [[ -e "$p" ]] || { echo "model missing: ${p}" >&2; exit 11; }
done

# usage: launch_and_wait <run-id> <wrapper-script> [KEY=VALUE ...]
launch_and_wait() {
  local run_id="$1"
  local target="$2"
  shift 2
  if tmux has-session -t "p3-${run_id}" 2>/dev/null; then
    echo "session p3-${run_id} already exists; skipping launch" >&2
    exit 3
  fi
  echo "[driver] launching ${run_id}"
  PROJECT3_DATA_ROOT="${data_root}" \
    bash "${script_dir}/start_tmux_run.sh" "${run_id}" 1 -- env "$@" bash "${target}" >/dev/null
  while tmux has-session -t "p3-${run_id}" 2>/dev/null; do sleep 10; done
  echo "[driver] finished ${run_id}"
}

declare -A MODELS=(
  [base]="${BASE_MODEL}"
  [gs300]="${GS300_MODEL}"
  [searchr1]="${OFFICIAL_MODEL}"
)
CONDITIONS="no-evidence real-top3 oracle shuffled"

for m in base gs300 searchr1; do
  for c in ${CONDITIONS}; do
    run_id="p3-eval-counterfactual-${m}-${c}-20260819a"
    launch_and_wait "${run_id}" "${script_dir}/run_p3_eval_counterfactual.sh" \
      PROJECT3_CF_MODEL="${MODELS[$m]}" \
      PROJECT3_CF_CONDITION="$c"
  done
done

# Step300 dev256 standard eval backfill (diag 1B / diag 3 input)
launch_and_wait "p3-eval-official-confirm256-gs300-20260819a" "${script_dir}/run_p3_eval_vllm_official.sh" \
  PROJECT3_EVAL_DATA=official-confirm256-v1 \
  PROJECT3_EVAL_MODEL="${GS300_MODEL}"

echo "[driver] all 13 runs finished"
