#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
# shellcheck source=gpu_guard.sh
source "${script_dir}/gpu_guard.sh"

usage() {
  echo "usage: PROJECT4_DATA_ROOT=/media/imc/data/yzy/agent $0 <run-id> <physical-gpu-ids> -- <command> [args...]"
}

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
  usage >&2
  exit 2
fi

run_id="$1"
gpu_ids="$2"
shift 3

if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  echo "invalid Run ID; use 1-80 letters, numbers, dots, underscores or dashes" >&2
  exit 2
fi

project4_validate_gpu_ids "$gpu_ids"
project4_require_known_gpus "$gpu_ids"
project4_require_idle_gpus "$gpu_ids"
data_root="$(project4_resolve_data_root)"
project4_require_disk_space "$data_root"
project4_require_repo_state "$repo_root"

project_data_dir="${data_root}/project4-opensearch-vl-rl"
run_root="${project_data_dir}/runs"
lock_root="${data_root}/.gpu-locks"
mkdir -p -- "$run_root" "$lock_root"

lock_fds=()
IFS=',' read -r -a gpu_array <<<"$gpu_ids"
for gpu_id in "${gpu_array[@]}"; do
  lock_path="${lock_root}/physical-gpu-${gpu_id}.lock"
  exec {lock_fd}>"$lock_path"
  if ! flock -n "$lock_fd"; then
    echo "physical GPU ${gpu_id} is locked by another managed run" >&2
    exit 6
  fi
  lock_fds+=("$lock_fd")
done

run_dir="${run_root}/${run_id}"
if [[ -e "$run_dir" ]]; then
  echo "refusing to overwrite existing Run directory: ${run_dir}" >&2
  exit 5
fi
mkdir -- "$run_dir"

ray_tmp_dir="$(mktemp -d /tmp/p4r.XXXXXX)"
identity_token="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
command_file="${run_dir}/command.txt"
printf '%q ' "$@" >"$command_file"
printf '\n' >>"$command_file"

project4_snapshot_gpus "$gpu_ids" "${run_dir}/gpu_before.csv"
{
  echo "run_id=${run_id}"
  echo "run_identity_token=${identity_token}"
  echo "physical_gpu_ids=${gpu_ids}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "user=$(id -un)"
  echo "working_directory=$(pwd)"
  echo "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"
  echo "opensearch_vl_commit=$(git -C "${repo_root}/project4-opensearch-vl-rl/vendor/OpenSearch-VL" rev-parse HEAD)"
  echo "data_root=${data_root}"
  echo "ray_tmp_dir=${ray_tmp_dir}"
  echo "proxy_policy=${PROJECT4_ALLOW_PROXY:-0}"
} >"${run_dir}/metadata.env"

child_pid=""
cleanup_started=0

process_session_alive() {
  [[ -n "$child_pid" ]] && ps -eo sid= | awk -v id="$child_pid" '$1 == id {found=1} END {exit !found}'
}

stop_process_group() {
  local signal_name="$1"
  if [[ -n "$child_pid" ]]; then
    kill -s "$signal_name" -- "-${child_pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local reason="$1"
  local attempt
  if (( cleanup_started )); then
    return
  fi
  cleanup_started=1
  trap - EXIT INT TERM HUP

  if [[ "$reason" != "normal" ]] || process_session_alive; then
    echo "managed Run cleanup reason=${reason}; stopping only process group ${child_pid}" >&2
    stop_process_group TERM
    for attempt in $(seq 1 30); do
      if ! process_session_alive; then
        break
      fi
      sleep 1
    done
    if process_session_alive; then
      echo "managed process group did not exit after 30 seconds; sending KILL to that group only" >&2
      stop_process_group KILL
    fi
  fi

  project4_report_gpu_processes "$gpu_ids" | tee -a "${run_dir}/cleanup.log"
  project4_snapshot_gpus "$gpu_ids" "${run_dir}/gpu_after.csv" || true
  if [[ -d "$ray_tmp_dir" && ! -e "${run_dir}/ray" ]]; then
    mv -- "$ray_tmp_dir" "${run_dir}/ray"
    echo "ray_temp_archived=${run_dir}/ray" >>"${run_dir}/metadata.env"
  elif [[ -d "$ray_tmp_dir" ]]; then
    echo "warning: preserving Ray temp because archive target exists: ${ray_tmp_dir}" | tee -a "${run_dir}/cleanup.log" >&2
  fi
  echo "finished_at=$(date --iso-8601=seconds)" >>"${run_dir}/metadata.env"
}

trap 'cleanup INT; exit 130' INT
trap 'cleanup TERM; exit 143' TERM
trap 'cleanup HUP; exit 129' HUP
trap 'cleanup EXIT' EXIT

echo "starting managed Run ${run_id} on physical GPU(s) ${gpu_ids}"

proxy_env=()
if [[ "${PROJECT4_ALLOW_PROXY:-0}" != "1" ]]; then
  proxy_env+=(
    -u http_proxy -u https_proxy -u all_proxy -u ftp_proxy
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u FTP_PROXY
  )
fi

setsid env "${proxy_env[@]}" \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$gpu_ids" \
  PROJECT4_RUN_ID="$run_id" \
  PROJECT4_RUN_DIR="$run_dir" \
  PROJECT4_RUN_TOKEN="$identity_token" \
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
