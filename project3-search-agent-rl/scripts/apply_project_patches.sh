#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
vendor_dir="${project_dir}/vendor/verl-agent"
patch_file="${project_dir}/patches/0001-search-retrieval-status-observability.patch"

if git -C "$vendor_dir" apply --reverse --check "$patch_file" 2>/dev/null; then
  echo "patch already applied: $(basename -- "$patch_file")"
  exit 0
fi

git -C "$vendor_dir" apply --check "$patch_file"
git -C "$vendor_dir" apply "$patch_file"
echo "patch applied: $(basename -- "$patch_file")"
