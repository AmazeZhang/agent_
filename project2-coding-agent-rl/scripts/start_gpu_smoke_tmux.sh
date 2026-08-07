#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
gpu_id="${1:-1}"
session="${2:-agent-p2-gpu-smoke}"
log_dir="$workspace/project2-coding-agent-rl/runs/smoke"
log_file="$log_dir/gpu-${gpu_id}.log"

if [[ "$gpu_id" == "0" ]]; then
  echo "REFUSED: physical GPU 0 is reserved and must not be used." >&2
  exit 2
fi
if ! [[ "$gpu_id" =~ ^[1-7]$ ]]; then
  echo "REFUSED: GPU id must be an integer from 1 to 7." >&2
  exit 2
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "REFUSED: tmux session '$session' already exists; no session was replaced." >&2
  exit 3
fi

bash "$workspace/shared/scripts/check_gpu.sh" "$gpu_id"

read -r used_mb free_mb utilization < <(
  nvidia-smi -i "$gpu_id" \
    --query-gpu=memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ','
)

if (( used_mb > 1024 )); then
  echo "REFUSED: GPU $gpu_id is using ${used_mb}MiB (>1024MiB threshold)." >&2
  exit 4
fi

mkdir -p "$log_dir"
nvidia-smi -i "$gpu_id" > "$log_dir/gpu-${gpu_id}-before.txt"

command="cd '$workspace' && PHYSICAL_GPU_ID='$gpu_id' CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='$gpu_id' '$workspace/project2-coding-agent-rl/.venvs/rllm-base/bin/python' '$workspace/project2-coding-agent-rl/scripts/gpu_smoke.py' > '$log_file' 2>&1"
tmux new-session -d -s "$session" "$command"

echo "Started in tmux session: $session"
echo "Log: $log_file"

