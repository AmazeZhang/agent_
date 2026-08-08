#!/usr/bin/env bash
# WP7: run one holdout SWE-agent batch (local vLLM endpoint, no API key).
# Usage: bash scripts/start_holdout_tmux.sh <short>-<variant> [session_name]
#   e.g. bash scripts/start_holdout_tmux.sh funcy-lookuper-3y0j7te5-base
# Config must exist at holdout-eval/configs/<short>-<variant>.config.yaml.
set -euo pipefail

PILOT_ROOT=/media/imc/data/yzy/agent/project2/swesmith-pilot20
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?usage: start_holdout_tmux.sh <short>-<variant> [session_name]}"
SESSION="${2:-holdout-$NAME}"
CONFIG="$PILOT_ROOT/holdout-eval/configs/$NAME.config.yaml"
LOG_DIR="$PILOT_ROOT/holdout-eval/runs/$NAME"

if [[ ! -f "$CONFIG" ]]; then
  echo "REFUSED: config not found: $CONFIG" >&2
  exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "REFUSED: tmux session '$SESSION' already exists" >&2
  exit 3
fi

mkdir -p "$LOG_DIR"
command="unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy ftp_proxy FTP_PROXY; cd '$PROJECT_ROOT' && sg docker -c '.venvs/swe-tools/bin/sweagent run-batch --config $CONFIG' > '$LOG_DIR/run_batch.log' 2>&1"
tmux new-session -d -s "$SESSION" "$command"
echo "Started tmux session: $SESSION"
echo "Log: $LOG_DIR/run_batch.log"
