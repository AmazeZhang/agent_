#!/usr/bin/env bash
# Host launcher for the isolated four-GPU ZeRO-3 fused/reference smoke.
set -euo pipefail

[ -n "${TMUX:-}" ] || {
  echo "REFUSED: container ZeRO-3 smoke must run inside tmux." >&2
  exit 2
}

ROOT=/home/imc/yzy/agent
EXPECTED_UUIDS=GPU-b8d12ea9-892d-6e5f-5db3-1784f847830d,GPU-ba5ab136-89eb-3b82-55a1-17e92e46ec05,GPU-0d86baf6-7b43-a8cf-34ca-8473d5e48a5d,GPU-b6a16d76-c59c-a1f0-3557-2026e1aca68c
CONTAINER_IMAGE=${P2_Z3_CONTAINER_IMAGE:-my-gpt-sovits:fixed-cu126}
TRITON_CACHE="$ROOT/project2-coding-agent-rl/runs/phase1/triton_cache"

mkdir -p "$TRITON_CACHE"

for gpu_id in 2 4 6 7; do
  bash "$ROOT/shared/scripts/check_gpu.sh" "$gpu_id"
done

docker run --rm --runtime=nvidia --entrypoint bash \
  --shm-size=16g \
  -e NVIDIA_VISIBLE_DEVICES="$EXPECTED_UUIDS" \
  -e EXPECTED_GPU_UUIDS="$EXPECTED_UUIDS" \
  -v "$ROOT:$ROOT" \
  -v /media/imc/data:/media/imc/data \
  -v /usr/local/cuda-12.6:/usr/local/cuda:ro \
  -v "$TRITON_CACHE:/root/.triton/cache" \
  -w "$ROOT/project2-coding-agent-rl" \
  "$CONTAINER_IMAGE" \
  scripts/phase1/container_z3_entrypoint.sh
