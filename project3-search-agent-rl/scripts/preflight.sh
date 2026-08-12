#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <comma-separated-physical-gpu-ids>"
  echo "recommended on the current server: $0 1,2,3,4,6,7"
  echo "GPU 0 is forbidden; GPU 5 requires ALLOW_UNSTABLE_GPU5=1"
  exit 2
fi

gpu_ids="$1"
if [[ ",${gpu_ids}," == *",0,"* ]]; then
  echo "refusing to use physical GPU 0"
  exit 2
fi

if [[ ",${gpu_ids}," == *",5,"* ]] && [[ "${ALLOW_UNSTABLE_GPU5:-0}" != "1" ]]; then
  echo "refusing to use unstable physical GPU 5 by default"
  echo "set ALLOW_UNSTABLE_GPU5=1 only for an explicitly supervised run"
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$gpu_ids"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python3 --version
python3 - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"visible_cuda_devices={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    gib = props.total_memory / 1024**3
    print(f"logical_gpu={index} name={props.name} vram_gib={gib:.1f}")
PY

available_kib=$(df -Pk . | awk 'NR==2 {print $4}')
available_gib=$((available_kib / 1024 / 1024))
echo "workspace_free_disk_gib=${available_gib}"

if (( available_gib < 150 )); then
  echo "warning: less than 150 GiB free; even the reduced Search-R1 workflow may run out of disk"
fi

echo "preflight completed; this script does not start training"
