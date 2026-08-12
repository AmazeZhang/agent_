#!/usr/bin/env bash
# Runs the short ZeRO-3 fused/reference comparison inside the isolated container.
set -euo pipefail

EXPECTED_UUIDS="${EXPECTED_GPU_UUIDS:?EXPECTED_GPU_UUIDS is required}"
mapfile -t uuid_lines < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
ACTUAL_UUIDS=$(IFS=,; echo "${uuid_lines[*]}")
if [[ "$ACTUAL_UUIDS" != "$EXPECTED_UUIDS" ]] || [[ "${#uuid_lines[@]}" -ne 4 ]]; then
  echo "REFUSED: isolated GPU UUID mismatch: actual=$ACTUAL_UUIDS expected=$EXPECTED_UUIDS" >&2
  exit 2
fi

unset CUDA_VISIBLE_DEVICES
export OPENRLHF_FUSED_CE_CHUNK_SIZE=512
export CC=/root/conda/bin/gcc
export CXX=/root/conda/bin/g++
VENV=/home/imc/yzy/agent/project2-coding-agent-rl/.venvs/phase1-openrlhf
ROOT=/home/imc/yzy/agent
export PATH="$VENV/bin:$PATH"

test -x "$CC"
test -x "$CXX"

"$VENV/bin/deepspeed" --include localhost:0,1,2,3 \
  "$ROOT/project2-coding-agent-rl/scripts/phase1/test_fused_z3.py" \
  --seqlen 128 --torch-optimizer --disable-gradient-checkpointing
