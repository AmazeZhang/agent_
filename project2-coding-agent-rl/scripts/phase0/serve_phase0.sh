#!/usr/bin/env bash
# Phase 0: serve SWE-Master-4B-RL on one vLLM OpenAI server (no LoRA).
# Usage: bash scripts/phase0/serve_phase0.sh <gpu 1-7> <model-name> <port> [session]
# Tool parser: swe_command (SWE-agent style <command> tags); override with TOOL_PARSER=...
set -euo pipefail

GPU="${1:?usage: serve_phase0.sh <gpu 1-7> <model-name> <port> [session]}"
MODEL_NAME="${2:-swe-master-4b-rl}"
PORT="${3:-8012}"
SESSION="${4:-p2-phase0-vllm}"
TOOL_PARSER="${TOOL_PARSER:-swe_command}"
bash /home/imc/yzy/agent/shared/scripts/check_gpu.sh "$GPU"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV=$PROJECT_ROOT/.venvs/rllm-base
MODEL_DIR=${MODEL_DIR:-/media/imc/data/yzy/agent/project2/models/SWE-Master-4B-RL}
LOG=/media/imc/data/yzy/agent/project2/phase0/serve-phase0.log
mkdir -p /media/imc/data/yzy/agent/project2/phase0

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$GPU; \
   export PATH=$VENV/bin:\$PATH; \
   export VLLM_USE_V1=1; export VLLM_ATTENTION_BACKEND=FLASH_ATTN; \
   $VENV/bin/python -m vllm.entrypoints.openai.api_server \
     --model $MODEL_DIR \
     --served-model-name $MODEL_NAME \
     --enable-auto-tool-choice \
     --tool-call-parser $TOOL_PARSER \
     --max-model-len 49152 \
     --gpu-memory-utilization 0.70 \
     --enforce-eager \
     --port $PORT 2>&1 | tee $LOG"
echo "started tmux session $SESSION (gpu $GPU, port $PORT, model $MODEL_DIR); log: $LOG"
