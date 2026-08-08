#!/usr/bin/env bash
# WP7: serve base + SFT + GRPO LoRA variants on one vLLM OpenAI server.
# Usage: bash scripts/serve_wp7.sh <gpu 1-7> port (default 8001)
set -euo pipefail

GPU="${1:?usage: serve_wp7.sh <gpu 1-7> port (default 8001)}"
PORT="${2:-8011}"
bash /home/imc/yzy/agent/shared/scripts/check_gpu.sh "$GPU"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=$PROJECT_ROOT/.venvs/rllm-base
BASE=/media/imc/data/yzy/agent/project2/hf-cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5
ADAPTERS=/media/imc/data/yzy/agent/project2/adapters

SESSION="p2-wp7-vllm"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "export CUDA_VISIBLE_DEVICES=$GPU; \
   export PATH=$VENV/bin:\$PATH; \
   export VLLM_USE_V1=1; export VLLM_ATTENTION_BACKEND=FLASH_ATTN; \
   $VENV/bin/python -m vllm.entrypoints.openai.api_server \
     --model $BASE \
     --served-model-name qwen25-coder-3b-base \
     --enable-lora \
     --lora-modules sft=$ADAPTERS/sft2-gs3 grpo=$ADAPTERS/grpo-gs1-r12 \
     --max-model-len 32768 \
     --max-lora-rank 8 \
     --gpu-memory-utilization 0.55 \
     --enforce-eager \
     --port $PORT 2>&1 | tee /media/imc/data/yzy/agent/project2/smoke-data/wp7-vllm.log"
echo "started tmux session $SESSION (gpu $GPU, port $PORT); log: smoke-data/wp7-vllm.log"
