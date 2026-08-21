#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
# shellcheck source=gpu_guard.sh
source "${script_dir}/gpu_guard.sh"

gpu_ids="${1:?usage: PROJECT4_DATA_ROOT=/media/imc/data/yzy/agent $0 <physical-gpu-ids>}"

project4_validate_gpu_ids "$gpu_ids"
project4_require_known_gpus "$gpu_ids"
project4_require_idle_gpus "$gpu_ids"
data_root="$(project4_resolve_data_root)"
project4_require_disk_space "$data_root"
project4_require_repo_state "$repo_root"

echo "repository_root=${repo_root}"
echo "repository_commit=$(git -C "$repo_root" rev-parse HEAD)"
echo "opensearch_vl_commit=$(git -C "${repo_root}/project4-opensearch-vl-rl/vendor/OpenSearch-VL" rev-parse HEAD)"
echo "physical_gpu_ids=${gpu_ids}"
project4_snapshot_gpus "$gpu_ids" /dev/stdout
echo "preflight=pass"

