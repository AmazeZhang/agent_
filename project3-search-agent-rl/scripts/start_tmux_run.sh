#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
  echo "usage: PROJECT3_DATA_ROOT=/media/imc/data $0 <run-id> <physical-gpu-ids> -- <command> [args...]" >&2
  exit 2
fi

run_id="$1"
gpu_ids="$2"
shift 3
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]]; then
  echo "invalid run ID" >&2
  exit 2
fi

session_name="p3-${run_id}"
if tmux has-session -t "$session_name" 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  exit 3
fi

data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
min_free_gib="${PROJECT3_MIN_FREE_GIB:-150}"
# Pass through every PROJECT3_* variable (eval model/tokenizer/data/temperature/
# num-rollouts, training adv_estimator/total-steps/profile, ...): the tmux
# server reuses the environment of its FIRST session, so ad-hoc prefix env
# vars would otherwise be lost. PROJECT3_DATA_ROOT / PROJECT3_MIN_FREE_GIB are
# re-exported explicitly below and excluded from the passthrough.
eval_extra_env=()
while IFS='=' read -r -d '' k v; do
  case "$k" in
    PROJECT3_DATA_ROOT|PROJECT3_MIN_FREE_GIB) ;;
    PROJECT3_*) eval_extra_env+=("${k}=${v}") ;;
  esac
done < <(env -0)
printf -v managed_command '%q ' \
  env \
  "PROJECT3_DATA_ROOT=${data_root}" \
  "PROJECT3_MIN_FREE_GIB=${min_free_gib}" \
  "${eval_extra_env[@]}" \
  bash "${script_dir}/run_managed.sh" "$run_id" "$gpu_ids" -- "$@"

tmux new-session -d -s "$session_name" -c "$PWD" "$managed_command"
tmux set-option -t "${session_name}:0" remain-on-exit on

echo "started tmux session: ${session_name}"
echo "attach:  tmux attach -t ${session_name}"
echo "detach:  press Ctrl-b, then d"
echo "status:  tmux list-sessions"
echo "output:  tmux capture-pane -pt ${session_name}:0 -S -100"
echo "logs:    ${data_root}/project3-search-agent-rl/runs/${run_id}/"
