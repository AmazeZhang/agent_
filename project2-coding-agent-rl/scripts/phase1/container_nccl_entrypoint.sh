#!/usr/bin/env bash
# Runs inside an NVIDIA-runtime container that exposes only physical GPU2/4/6/7.
set -euo pipefail

EXPECTED_UUIDS="${EXPECTED_GPU_UUIDS:?EXPECTED_GPU_UUIDS is required}"
mapfile -t uuid_lines < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
ACTUAL_UUIDS=$(IFS=,; echo "${uuid_lines[*]}")
if [[ "$ACTUAL_UUIDS" != "$EXPECTED_UUIDS" ]]; then
  echo "REFUSED: isolated GPU UUID mismatch: actual=$ACTUAL_UUIDS expected=$EXPECTED_UUIDS" >&2
  exit 2
fi
if [[ "${#uuid_lines[@]}" -ne 4 ]]; then
  echo "REFUSED: expected exactly four isolated GPUs, got ${#uuid_lines[@]}" >&2
  exit 2
fi

# Logical 0..3 are safe only because the UUID check above proves they map to
# physical 2/4/6/7. The Python preflight reports both logical and physical IDs.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PHYSICAL_GPU_IDS=2,4,6,7
VENV=/home/imc/yzy/agent/project2-coding-agent-rl/.venvs/phase1-openrlhf
ROOT=/home/imc/yzy/agent

"$VENV/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$ROOT/project2-coding-agent-rl/scripts/phase1/nccl_preflight.py"
