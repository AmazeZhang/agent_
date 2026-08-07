#!/usr/bin/env bash
set -euo pipefail

gpu_id="${1:?usage: bash shared/scripts/check_gpu.sh <physical-gpu-id>}"

if [[ "$gpu_id" == "0" ]]; then
  echo "REFUSED: physical GPU 0 is reserved and must not be used." >&2
  exit 2
fi

if ! [[ "$gpu_id" =~ ^[1-7]$ ]]; then
  echo "REFUSED: GPU id must be an integer from 1 to 7." >&2
  exit 2
fi

echo "Selected physical GPU: $gpu_id"
nvidia-smi -i "$gpu_id" \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv

echo "All active compute processes:"
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader || true

