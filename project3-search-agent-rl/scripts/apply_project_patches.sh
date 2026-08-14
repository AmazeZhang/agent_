#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
vendor_dir="${project_dir}/vendor/verl-agent"
patch_files=(
  "${project_dir}/patches/0001-search-retrieval-status-observability.patch"
  "${project_dir}/patches/0002-structured-rollout-audit.patch"
  "${project_dir}/patches/0003-graceful-ray-shutdown-and-atomic-rollout.patch"
  "${project_dir}/patches/0004-search-prompt-and-format-reward.patch"
)

for patch_file in "${patch_files[@]}"; do
  if git -C "$vendor_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
    echo "patch already applied: $(basename -- "$patch_file")"
    continue
  fi
  git -C "$vendor_dir" apply --check "$patch_file"
  git -C "$vendor_dir" apply "$patch_file"
  echo "patch applied: $(basename -- "$patch_file")"
done
