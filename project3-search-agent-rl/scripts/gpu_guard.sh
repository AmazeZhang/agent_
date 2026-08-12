#!/usr/bin/env bash

# Shared, side-effect-free GPU validation helpers for project launch scripts.
# This file is meant to be sourced; it never changes GPU state by itself.

project3_validate_gpu_ids() {
  local gpu_ids="${1:-}"
  local gpu_id
  local -A seen=()

  if [[ ! "$gpu_ids" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid GPU list: expected comma-separated physical IDs" >&2
    return 2
  fi

  IFS=',' read -r -a project3_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project3_gpu_array[@]}"; do
    if [[ -n "${seen[$gpu_id]:-}" ]]; then
      echo "invalid GPU list: duplicate physical GPU ${gpu_id}" >&2
      return 2
    fi
    seen[$gpu_id]=1

    if [[ "$gpu_id" == "0" ]]; then
      echo "refusing physical GPU 0: it is reserved for the Linux desktop" >&2
      return 2
    fi
    if [[ "$gpu_id" == "5" && "${ALLOW_UNSTABLE_GPU5:-0}" != "1" ]]; then
      echo "refusing unstable physical GPU 5 by default" >&2
      echo "set ALLOW_UNSTABLE_GPU5=1 only for an explicitly supervised run" >&2
      return 2
    fi
  done
}

project3_require_known_gpus() {
  local gpu_ids="$1"
  local gpu_count
  local gpu_id

  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is not available" >&2
    return 2
  }
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
  IFS=',' read -r -a project3_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project3_gpu_array[@]}"; do
    if (( gpu_id < 0 || gpu_id >= gpu_count )); then
      echo "physical GPU ${gpu_id} does not exist; detected ${gpu_count} GPUs" >&2
      return 2
    fi
  done
}

project3_require_idle_gpus() {
  local gpu_ids="$1"
  local gpu_id
  local processes

  IFS=',' read -r -a project3_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project3_gpu_array[@]}"; do
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

project3_report_gpu_processes() {
  local gpu_ids="$1"
  local gpu_id
  local processes

  IFS=',' read -r -a project3_gpu_array <<<"$gpu_ids"
  for gpu_id in "${project3_gpu_array[@]}"; do
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

project3_resolve_data_root() {
  local data_root="${PROJECT3_DATA_ROOT:-}"

  if [[ -z "$data_root" ]]; then
    echo "PROJECT3_DATA_ROOT is required" >&2
    echo "mount the data disk, then point PROJECT3_DATA_ROOT at a writable directory on it" >&2
    return 2
  fi
  if [[ ! -d "$data_root" || ! -w "$data_root" ]]; then
    echo "PROJECT3_DATA_ROOT must be an existing writable directory: ${data_root}" >&2
    return 2
  fi
  readlink -f -- "$data_root"
}

project3_require_disk_space() {
  local data_root="$1"
  local min_free_gib="${PROJECT3_MIN_FREE_GIB:-150}"
  local available_kib
  local available_gib

  if [[ ! "$min_free_gib" =~ ^[0-9]+$ ]]; then
    echo "PROJECT3_MIN_FREE_GIB must be a non-negative integer" >&2
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
