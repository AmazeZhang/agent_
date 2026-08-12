#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=gpu_guard.sh
source "${script_dir}/gpu_guard.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <comma-separated-physical-gpu-ids>"
  echo "recommended on the current server: $0 1,2,3,4,6,7"
  echo "GPU 0 is forbidden; GPU 5 requires ALLOW_UNSTABLE_GPU5=1"
  exit 2
fi

gpu_ids="$1"
project3_validate_gpu_ids "$gpu_ids"
project3_require_known_gpus "$gpu_ids"
project3_require_idle_gpus "$gpu_ids"
data_root="$(project3_resolve_data_root)"
project3_require_disk_space "$data_root"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_ids"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --id="$gpu_ids" \
  --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv
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

echo "preflight completed; this script does not start training"
