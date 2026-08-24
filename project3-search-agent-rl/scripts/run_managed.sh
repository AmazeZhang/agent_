#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu_guard.sh
source "${script_dir}/gpu_guard.sh"

usage() {
  echo "usage: PROJECT3_DATA_ROOT=/mounted/data $0 <run-id> <physical-gpu-ids> -- <command> [args...]"
}

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
  usage >&2
  exit 2
fi

run_id="$1"
gpu_ids="$2"
shift 3

if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  echo "invalid run ID; use 1-80 letters, numbers, dots, underscores, or dashes" >&2
  exit 2
fi

project3_validate_gpu_ids "$gpu_ids"
project3_require_known_gpus "$gpu_ids"
project3_require_idle_gpus "$gpu_ids"
data_root="$(project3_resolve_data_root)"
project3_require_disk_space "$data_root"

project_data_dir="${data_root}/project3-search-agent-rl"
mkdir -p -- "$project_data_dir"
lock_fds=()
IFS=',' read -r -a gpu_array <<<"$gpu_ids"
for gpu_id in "${gpu_array[@]}"; do
  lock_path="${project_data_dir}/gpu-${gpu_id}.lock"
  exec {lock_fd}>"$lock_path"
  if ! flock -n "$lock_fd"; then
    echo "physical GPU ${gpu_id} is locked by another managed project3 run" >&2
    exit 6
  fi
  lock_fds+=("$lock_fd")
done

run_root="${project_data_dir}/runs"
run_dir="${run_root}/${run_id}"
mkdir -p -- "$run_root"
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing run directory: ${run_dir}" >&2
  exit 5
fi
mkdir -- "$run_dir"

# Ray appends long session and socket names. Keep its live temp prefix short so
# AF_UNIX's 107-byte path limit is not coupled to the descriptive run ID. The
# directory is moved into the immutable run directory during cleanup.
ray_tmp_dir="$(mktemp -d /tmp/p3r.XXXXXX)"

command_file="${run_dir}/command.txt"
printf '%q ' "$@" >"$command_file"
printf '\n' >>"$command_file"
{
  echo "run_id=${run_id}"
  echo "physical_gpu_ids=${gpu_ids}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "user=$(id -un)"
  echo "working_directory=$(pwd)"
  echo "data_root=${data_root}"
  echo "ray_tmp_dir=${ray_tmp_dir}"
} >"${run_dir}/metadata.env"
child_pid=""
cleanup_started=0

stop_process_group() {
  local signal_name="$1"
  if [[ -n "$child_pid" ]] && ps -eo sid=,pgid= | awk -v id="$child_pid" '$1 == id && $2 == id {found=1} END {exit !found}'; then
    kill -s "$signal_name" -- "-${child_pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local reason="$1"
  local attempt
  local session_alive=0
  if (( cleanup_started )); then
    return
  fi
  cleanup_started=1
  trap - EXIT INT TERM HUP

  if [[ -n "$child_pid" ]] && ps -eo sid= | awk -v id="$child_pid" '$1 == id {found=1} END {exit !found}'; then
    session_alive=1
  fi
  if [[ "$reason" != "normal" || "$session_alive" == "1" ]]; then
    if [[ "$reason" == "normal" ]]; then
      echo "command exited but child processes remain; stopping only managed process group ${child_pid}" >&2
    else
      echo "managed run received ${reason}; stopping only process group ${child_pid}" >&2
    fi
    stop_process_group TERM
    for attempt in $(seq 1 30); do
      if ! ps -eo sid= | awk -v id="$child_pid" '$1 == id {found=1} END {exit !found}'; then
        break
      fi
      sleep 1
    done
    if ps -eo sid= | awk -v id="$child_pid" '$1 == id {found=1} END {exit !found}'; then
      echo "process group did not exit after 30 seconds; sending KILL to that group only" >&2
      stop_process_group KILL
    fi
  fi

  project3_report_gpu_processes "$gpu_ids" | tee -a "${run_dir}/cleanup.log"
  if [[ -d "$ray_tmp_dir" && ! -e "${run_dir}/ray" ]]; then
    mv -- "$ray_tmp_dir" "${run_dir}/ray"
    echo "ray_temp_archived=${run_dir}/ray" >>"${run_dir}/metadata.env"
  elif [[ -d "$ray_tmp_dir" ]]; then
    echo "warning: preserving live Ray temp because archive target exists: ${ray_tmp_dir}" | tee -a "${run_dir}/cleanup.log" >&2
  fi
  echo "finished_at=$(date --iso-8601=seconds)" >>"${run_dir}/metadata.env"
  generate_training_curves
}

generate_training_curves() {
  local curve_script="${script_dir}/generate_training_curves.py"
  local curve_python="${project_data_dir}/envs/searchr1-repro-cu124/bin/python"
  local curve_log="${run_dir}/curve_generation.log"

  # Evaluation/diagnostic runs do not emit verl optimizer-step metrics.  Skip
  # them without creating an empty artifact directory.
  if ! grep -q 'training/global_step:' "${run_dir}/stdout.log" 2>/dev/null; then
    echo "curve_generation=skipped_no_training_metrics" >>"${run_dir}/metadata.env"
    return 0
  fi
  if [[ ! -x "$curve_python" || ! -f "$curve_script" ]]; then
    echo "curve generation skipped: fixed Python or generator missing" >"$curve_log"
    echo "curve_generation=skipped_generator_missing" >>"${run_dir}/metadata.env"
    return 0
  fi

  # Derived plots are CPU-only and fail-open with respect to the immutable
  # training result: a plotting bug must never rewrite the command exit code.
  if CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 "$curve_python" "$curve_script" \
      --run-dir "$run_dir" >>"$curve_log" 2>&1; then
    echo "curve_generation=generated" >>"${run_dir}/metadata.env"
  else
    echo "warning: training curve generation failed; see ${curve_log}" >&2
    echo "curve_generation=failed" >>"${run_dir}/metadata.env"
  fi
}

trap 'cleanup EXIT; exit 130' INT
trap 'cleanup TERM; exit 143' TERM
trap 'cleanup HUP; exit 129' HUP
trap 'cleanup EXIT' EXIT

echo "starting managed run ${run_id} on physical GPU(s) ${gpu_ids}"
setsid env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu_ids" \
  PROJECT3_RUN_ID="$run_id" \
  PROJECT3_RUN_DIR="$run_dir" \
  RAY_TMPDIR="$ray_tmp_dir" \
  "$@" >"${run_dir}/stdout.log" 2>"${run_dir}/stderr.log" &
child_pid=$!
echo "$child_pid" >"${run_dir}/session_id"
echo "session_id=${child_pid}" >>"${run_dir}/metadata.env"

set +e
wait "$child_pid"
exit_code=$?
set -e
echo "exit_code=${exit_code}" >>"${run_dir}/metadata.env"
cleanup normal
exit "$exit_code"
