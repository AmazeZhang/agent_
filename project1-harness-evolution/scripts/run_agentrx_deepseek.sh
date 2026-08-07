#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "$project_root/.." && pwd)"

set -a
source "$workspace_root/.secrets/deepseek.env"
set +a

exec "$project_root/.venvs/agentrx/bin/python" \
  "$project_root/scripts/agentrx_deepseek_cli.py" "$@"
