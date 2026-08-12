#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: PROJECT3_DATA_ROOT=/mounted/data $0 <run-id>"
}

if (( $# != 1 )); then
  usage >&2
  exit 2
fi

run_id="$1"
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  echo "invalid run ID" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu_guard.sh
source "${script_dir}/gpu_guard.sh"
data_root="$(project3_resolve_data_root)"
run_dir="${data_root}/project3-search-agent-rl/runs/${run_id}"
session_file="${run_dir}/session_id"

if [[ ! -f "$session_file" ]]; then
  echo "managed run session file not found: ${session_file}" >&2
  exit 2
fi
session_id="$(<"$session_file")"
if [[ ! "$session_id" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid recorded session ID" >&2
  exit 2
fi

mapfile -t session_processes < <(ps -eo pid=,sid=,pgid=,user=,args= | awk -v id="$session_id" '$2 == id')
if (( ${#session_processes[@]} == 0 )); then
  echo "managed run ${run_id} is not running"
  exit 0
fi
if printf '%s\n' "${session_processes[@]}" | awk -v user="$(id -un)" '$4 != user {bad=1} END {exit !bad}'; then
  echo "refusing stop: session contains a process owned by another user" >&2
  exit 3
fi

# A stale session ID could eventually be reused. Require every target process to
# carry the exact run-directory token inherited from run_managed.sh before kill.
for process_line in "${session_processes[@]}"; do
  read -r process_pid _ <<<"$process_line"
  run_token_found=0
  while IFS= read -r -d '' env_entry; do
    if [[ "$env_entry" == "PROJECT3_RUN_DIR=${run_dir}" ]]; then
      run_token_found=1
      break
    fi
  done <"/proc/${process_pid}/environ"
  if (( ! run_token_found )); then
    echo "refusing stop: PID ${process_pid} does not carry this run's identity token" >&2
    exit 3
  fi
done

echo "processes in managed session ${session_id}:"
printf '%s\n' "${session_processes[@]}"
echo "sending TERM to managed process group ${session_id}"
kill -TERM -- "-${session_id}"

for _ in $(seq 1 30); do
  if ! ps -eo sid= | awk -v id="$session_id" '$1 == id {found=1} END {exit !found}'; then
    echo "managed run ${run_id} stopped"
    exit 0
  fi
  sleep 1
done

echo "managed session did not exit after 30 seconds; sending KILL to that group only" >&2
kill -KILL -- "-${session_id}"
