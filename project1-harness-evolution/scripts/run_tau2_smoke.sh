#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace="$(cd "$project_dir/.." && pwd)"
tau2_repo="$project_dir/vendor/tau2-bench"
python_bin="$project_dir/.venvs/tau2/bin/python"
tau2_adapter="$project_dir/scripts/tau2_deepseek_cli.py"
secrets_file="$workspace/.secrets/deepseek.env"
run_id="${1:-p1-smoke-deepseek-v4-flash-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -r "$secrets_file" ]]; then
  echo "Missing readable secrets file: $secrets_file" >&2
  exit 2
fi

set -a
source "$secrets_file"
set +a

cd "$tau2_repo"
"$python_bin" "$tau2_adapter" run \
  --domain mock \
  --agent-llm "openai/$DEEPSEEK_MODEL" \
  --user-llm "openai/$DEEPSEEK_MODEL" \
  --agent-llm-args "{\"api_base\":\"$OPENAI_BASE_URL\",\"max_tokens\":4096,\"temperature\":0}" \
  --user-llm-args "{\"api_base\":\"$OPENAI_BASE_URL\",\"max_tokens\":4096,\"temperature\":0}" \
  --num-trials 1 \
  --num-tasks 1 \
  --max-concurrency 1 \
  --max-steps 30 \
  --timeout 180 \
  --seed 300 \
  --save-to "$run_id" \
  --verbose-logs \
  --log-level INFO
