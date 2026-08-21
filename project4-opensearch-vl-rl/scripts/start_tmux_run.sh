#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
  echo "usage: PROJECT4_DATA_ROOT=/media/imc/data/yzy/agent $0 <run-id> <physical-gpu-ids> -- <command> [args...]" >&2
  exit 2
fi

run_id="$1"
gpu_ids="$2"
shift 3
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  echo "invalid Run ID" >&2
  exit 2
fi

session_name="p4-${run_id}"
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 3
fi

data_root="${PROJECT4_DATA_ROOT:-/media/imc/data/yzy/agent}"
min_free_gib="${PROJECT4_MIN_FREE_GIB:-300}"
extra_env=()
while IFS='=' read -r -d '' key value; do
  case "$key" in
    PROJECT4_DATA_ROOT|PROJECT4_MIN_FREE_GIB|PROJECT4_RUN_ID|PROJECT4_RUN_DIR|PROJECT4_RUN_TOKEN) ;;
    PROJECT4_*) extra_env+=("${key}=${value}") ;;
  esac
done < <(env -0)

printf -v managed_command '%q ' \
  env \
  -u http_proxy -u https_proxy -u all_proxy -u ftp_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u FTP_PROXY \
  "PROJECT4_DATA_ROOT=${data_root}" \
  "PROJECT4_MIN_FREE_GIB=${min_free_gib}" \
  "${extra_env[@]}" \
  bash "${script_dir}/run_managed.sh" "$run_id" "$gpu_ids" -- "$@"

tmux new-session -d -s "$session_name" -c "$PWD" "$managed_command"
tmux set-option -t "${session_name}:0" remain-on-exit on

echo "started tmux session: ${session_name}"
echo "attach: tmux attach -t ${session_name}"
echo "detach: press Ctrl-b, then d"
echo "logs: ${data_root}/project4-opensearch-vl-rl/runs/${run_id}/"

