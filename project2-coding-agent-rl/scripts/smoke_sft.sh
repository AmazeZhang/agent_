#!/usr/bin/env bash
# WP5: LoRA SFT smoke in tmux on one free GPU (1-7, never 0).
# Usage: bash scripts/smoke_sft.sh <gpu>
set -euo pipefail

GPU="${1:?usage: smoke_sft.sh <gpu 1-7>}"
bash /home/imc/yzy/agent/shared/scripts/check_gpu.sh "$GPU"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=$PROJECT_ROOT/.venvs/rllm-base

# build smoke data if missing
if [[ ! -f /media/imc/data/yzy/agent/project2/smoke-data/sft-train.parquet ]]; then
  $PROJECT_ROOT/.venvs/swe-tools/bin/python "$PROJECT_ROOT/scripts/build_smoke_sft_data.py"
fi

SESSION="p2-smoke-sft"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "export CUDA_VISIBLE_DEVICES=$GPU; \
  export WANDB_MODE=disabled; \
  $VENV/bin/python $PROJECT_ROOT/scripts/smoke_sft.py 2>&1 | tee /media/imc/data/yzy/agent/project2/smoke-data/sft.log"
echo "started tmux session $SESSION (gpu $GPU); log: smoke-data/sft.log"
