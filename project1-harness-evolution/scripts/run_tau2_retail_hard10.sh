#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$project_root/.." && pwd)"

set -a
source "$workspace_root/.secrets/deepseek.env"
set +a

model="openai/${DEEPSEEK_MODEL:-deepseek-v4-flash}"
llm_args='{"api_base":"https://api.deepseek.com","max_tokens":8192,"temperature":0.0,"extra_body":{"thinking":{"type":"disabled"}}}'

exec "$project_root/.venvs/tau2/bin/python" \
  "$project_root/scripts/tau2_deepseek_cli.py" run \
  --domain retail \
  --task-ids 16 21 23 28 30 32 42 54 55 104 \
  --num-trials 1 \
  --agent llm_agent \
  --agent-llm "$model" \
  --agent-llm-args "$llm_args" \
  --user user_simulator \
  --user-llm "$model" \
  --user-llm-args "$llm_args" \
  --max-concurrency 2 \
  --max-steps 100 \
  --timeout 300 \
  --max-retries 1 \
  --seed 302 \
  --save-to p1-retail-hard10-deepseek-v4-flash-20260806 \
  --log-level INFO
