#!/usr/bin/env bash
# Serial GPU1-only confirm256 evaluation for the preregistered seed2026 pair.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"

clean_model="${project_data}/models/p3-clean-grpo10-seed2026-gs10-merged-20260824a"
aware_model="${project_data}/models/p3-aware-v2-grpo10-seed2026-gs10-merged-20260824a"
tokenizer="${project_data}/models/Qwen2.5-3B"
clean_run="p3-eval-clean-grpo10-seed2026-gs10-confirm256-20260824a"
aware_run="p3-eval-aware-v2-seed2026-gs10-confirm256-20260824a"

for path in "${clean_model}" "${aware_model}" "${tokenizer}"; do
  [[ -e "${path}" ]] || { echo "required model/tokenizer missing: ${path}" >&2; exit 11; }
done

launch_and_accept() {
  local run_id="$1" model="$2"
  local run_dir="${project_data}/runs/${run_id}"
  local session="p3-${run_id}"

  [[ ! -e "${run_dir}" ]] || { echo "refusing to reuse run directory: ${run_dir}" >&2; return 12; }
  ! tmux has-session -t "${session}" 2>/dev/null || {
    echo "refusing to reuse tmux session: ${session}" >&2
    return 13
  }

  echo "[pair-eval] launch run=${run_id} model=${model}"
  PROJECT3_DATA_ROOT="${data_root}" \
  PROJECT3_EVAL_DATA=official-confirm256-v1 \
  PROJECT3_EVAL_MODEL="${model}" \
  PROJECT3_EVAL_TOKENIZER="${tokenizer}" \
  PROJECT3_EVAL_TEMPERATURE=0.0 \
  PROJECT3_EVAL_NUM_ROLLOUTS=1 \
  PROJECT3_EVAL_RETRIEVAL_CONDITION=real \
    bash "${script_dir}/start_tmux_run.sh" "${run_id}" 1 -- \
      bash "${script_dir}/run_p3_eval_v2.sh"

  local waited=0
  while (( waited < 1080 )); do  # 3h timeout, 10s cadence
    if [[ -f "${run_dir}/metadata.env" ]] && grep -q '^exit_code=' "${run_dir}/metadata.env"; then
      break
    fi
    sleep 10
    waited=$((waited + 1))
  done

  grep -q '^exit_code=0$' "${run_dir}/metadata.env" || {
    echo "${run_id}: missing successful managed exit" >&2
    return 14
  }
  [[ -s "${run_dir}/results.json" && -s "${run_dir}/episodes.jsonl" ]] || {
    echo "${run_id}: missing evaluation artifacts" >&2
    return 15
  }
  [[ "$(wc -l <"${run_dir}/episodes.jsonl")" == 256 ]] || {
    echo "${run_id}: episodes.jsonl does not contain 256 rows" >&2
    return 16
  }
  ! find "${run_dir}" -type f \( -name '*.partial' -o -name '*.incomplete' \) -print -quit | grep -q . || {
    echo "${run_id}: partial artifact found" >&2
    return 17
  }
  grep -q '^physical_gpu=1 compute_processes=none$' "${run_dir}/cleanup.log" || {
    echo "${run_id}: GPU1 cleanup gate failed" >&2
    return 18
  }
  [[ -s "${run_dir}/peak_memory_nvidia_smi.json" ]] || {
    echo "${run_id}: physical peak memory file missing" >&2
    return 19
  }
  if grep -Eiq 'Traceback|out of memory|CUDA error|NCCL[^[:space:]]* error|Xid|SYSTEM_ERROR|RAY_WORKER_FAILURE' \
      "${run_dir}/stdout.log" "${run_dir}/stderr.log"; then
    echo "${run_id}: fatal error signature found" >&2
    return 20
  fi
  echo "[pair-eval] accepted run=${run_id}"
}

launch_and_accept "${clean_run}" "${clean_model}"
launch_and_accept "${aware_run}" "${aware_model}"
echo "PAIR_EVAL_RUNS_PASS clean=${clean_run} aware=${aware_run}"
