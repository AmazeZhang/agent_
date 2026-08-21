#!/usr/bin/env bash

# Side-effect-free validation helpers for Project 4 launch scripts.
# This file is sourced by other scripts and never changes GPU state itself.

project4_validate_gpu_ids() {
  local gpu_ids="${1:-}"
  local gpu_id
  local -A seen=()

  if [[ ! "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid GPU list: expected comma-separated physical IDs" >&2
    return 2
  fi

  IFS=',' read -r -a project4_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project4_gpu_array[@]}"; do
    if [[ -n "${seen[$gpu_id]:-}" ]]; then
      echo "invalid GPU list: duplicate physical GPU ${gpu_id}" >&2
      return 2
    fi
    seen[$gpu_id]=1

    if [[ "$gpu_id" == "0" ]]; then
      echo "refusing physical GPU 0: reserved for the Linux desktop" >&2
      return 2
    fi
    if (( gpu_id < 1 || gpu_id > 7 )); then
      echo "invalid physical GPU ${gpu_id}: expected an ID from 1 to 7" >&2
      return 2
    fi
    if [[ "$gpu_id" == "5" && "${ALLOW_UNSTABLE_GPU5:-0}" != "1" ]]; then
      echo "refusing historically unstable physical GPU 5 by default" >&2
      echo "Project 4 requires explicit ALLOW_UNSTABLE_GPU5=1 and a supervised health-gated run" >&2
      return 2
    fi
  done
}

project4_require_known_gpus() {
  local gpu_ids="$1"
  local gpu_count
  local gpu_id

  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is not available" >&2
    return 2
  }
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
  IFS=',' read -r -a project4_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project4_gpu_array[@]}"; do
    if (( gpu_id < 0 || gpu_id >= gpu_count )); then
      echo "physical GPU ${gpu_id} does not exist; detected ${gpu_count} GPUs" >&2
      return 2
    fi
  done
}

project4_require_idle_gpus() {
  local gpu_ids="$1"
  local gpu_id
  local processes

  IFS=',' read -r -a project4_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project4_gpu_array[@]}"; do
    processes="$(nvidia-smi --id="$gpu_id" \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$processes" ]]; then
      echo "physical GPU ${gpu_id} already has compute processes:" >&2
      echo "$processes" >&2
      return 3
    fi
  done
}

project4_snapshot_gpus() {
  local gpu_ids="$1"
  local output_file="$2"
  local gpu_id

  if [[ "$output_file" != "/dev/stdout" ]]; then
    : >"$output_file"
  fi
  IFS=',' read -r -a project4_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project4_gpu_array[@]}"; do
    nvidia-smi --id="$gpu_id" \
      --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,driver_version \
      --format=csv,noheader,nounits >>"$output_file"
  done
}

project4_report_gpu_processes() {
  local gpu_ids="$1"
  local gpu_id
  local processes

  IFS=',' read -r -a project4_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project4_gpu_array[@]}"; do
    processes="$(nvidia-smi --id="$gpu_id" \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$processes" ]]; then
      echo "warning: physical GPU ${gpu_id} still has compute processes after the run:" >&2
      echo "$processes" >&2
    else
      echo "physical_gpu=${gpu_id} compute_processes=none"
    fi
  done
}

project4_resolve_data_root() {
  local data_root="${PROJECT4_DATA_ROOT:-}"
  local expected_prefix="/media/imc/data/yzy/agent"

  if [[ -z "$data_root" ]]; then
    echo "PROJECT4_DATA_ROOT is required" >&2
    return 2
  fi
  if [[ ! -d "$data_root" || ! -w "$data_root" ]]; then
    echo "PROJECT4_DATA_ROOT must be an existing writable directory: ${data_root}" >&2
    return 2
  fi
  data_root="$(readlink -f -- "$data_root")"
  if [[ "${PROJECT4_ALLOW_TEST_DATA_ROOT:-0}" != "1" && "$data_root" != "$expected_prefix" ]]; then
    echo "PROJECT4_DATA_ROOT must be ${expected_prefix}; got ${data_root}" >&2
    return 2
  fi
  printf '%s\n' "$data_root"
}

project4_require_disk_space() {
  local data_root="$1"
  local min_free_gib="${PROJECT4_MIN_FREE_GIB:-300}"
  local available_kib
  local available_gib

  if [[ ! "$min_free_gib" =~ ^[0-9]+$ ]]; then
    echo "PROJECT4_MIN_FREE_GIB must be a non-negative integer" >&2
    return 2
  fi
  available_kib="$(df -Pk -- "$data_root" | awk 'NR==2 {print $4}')"
  available_gib=$((available_kib / 1024 / 1024))
  echo "data_root=${data_root}"
  echo "data_root_free_disk_gib=${available_gib}"
  if (( available_gib < min_free_gib )); then
    echo "refusing launch: data root has less than ${min_free_gib} GiB free" >&2
    return 4
  fi
}

project4_require_repo_state() {
  local repo_root="$1"
  local expected_submodule="c5c02a49780e26ae9cb6f1fb56731d1e594d59f0"
  local actual_submodule

  if [[ ! -d "${repo_root}/.git" ]]; then
    echo "repository metadata not found under ${repo_root}" >&2
    return 2
  fi
  actual_submodule="$(git -C "${repo_root}/project4-opensearch-vl-rl/vendor/OpenSearch-VL" rev-parse HEAD)"
  if [[ "$actual_submodule" != "$expected_submodule" ]]; then
    echo "OpenSearch-VL submodule mismatch: expected ${expected_submodule}, got ${actual_submodule}" >&2
    return 5
  fi
}
