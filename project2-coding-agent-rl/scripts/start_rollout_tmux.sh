#!/usr/bin/env bash
# Start one SWE-agent rollout in a tmux session (guarded, DeepSeek API, no GPU).
# Usage: bash scripts/start_rollout_tmux.sh <run_id> [session_name]
# The config must already exist at local-instances/<run_id>.config.yaml
set -euo pipefail

PILOT_ROOT=/media/imc/data/yzy/agent/project2/swesmith-pilot20
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${1:?usage: start_rollout_tmux.sh <run_id> [session_name]}"
SESSION="${2:-agent-p2-$RUN_ID}"
CONFIG="$PILOT_ROOT/local-instances/$RUN_ID.config.yaml"
LOG_DIR="$PILOT_ROOT/runs/$RUN_ID"

if [[ ! -f "$CONFIG" ]]; then
  echo "REFUSED: config not found: $CONFIG" >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REFUSED: tmux session '$SESSION' already exists" >&2
  exit 3
fi

mkdir -p "$LOG_DIR"
command="set -a; source /home/imc/yzy/agent/.secrets/deepseek.env; set +a; unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy ftp_proxy FTP_PROXY; cd '$PROJECT_ROOT' && sg docker -c '.venvs/swe-tools/bin/sweagent run-batch --config $CONFIG' > '$LOG_DIR/run_batch.log' 2>&1"
tmux new-session -d -s "$SESSION" "$command"
echo "Started tmux session: $SESSION"
echo "Log: $LOG_DIR/run_batch.log"
echo "Config: $CONFIG"
