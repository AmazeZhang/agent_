#!/usr/bin/env bash
# WP6: Agentic RL (GRPO) smoke in tmux on one free GPU (1-7, never 0).
# Usage: bash scripts/smoke_rl.sh <gpu>
set -euo pipefail

GPU="${1:?usage: smoke_rl.sh <gpu 1-7>}"
bash /home/imc/yzy/agent/shared/scripts/check_gpu.sh "$GPU"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=$PROJECT_ROOT/.venvs/rllm-base

# build GRPO smoke data if missing
$VENV/bin/python "$PROJECT_ROOT/scripts/build_grpo_smoke_data.py" --force || true

SESSION="p2-smoke-rl"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$GPU; \
   export PYTHONUNBUFFERED=1; \
   export WANDB_MODE=disabled; \
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
   export VLLM_ATTENTION_BACKEND=FLASH_ATTN; \
   export VLLM_USE_V1=1; \
   export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1; \
   export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000; \
   $VENV/bin/python $PROJECT_ROOT/scripts/smoke_rl.py 2>&1 | tee /media/imc/data/yzy/agent/project2/smoke-data/grpo.log"
echo "started tmux session $SESSION (gpu $GPU); log: smoke-data/grpo.log"
