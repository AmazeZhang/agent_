#!/usr/bin/env bash
set -euo pipefail

run_id="${1:?usage: PROJECT4_DATA_ROOT=/media/imc/data/yzy/agent $0 <exact-run-id>}"
data_root="${PROJECT4_DATA_ROOT:?PROJECT4_DATA_ROOT is required}"
data_root="$(readlink -f -- "$data_root")"
run_dir="${data_root}/project4-opensearch-vl-rl/runs/${run_id}"

if [[ ! -d "$run_dir" ]]; then
  echo "Run directory does not exist: ${run_dir}" >&2
  exit 2
fi
if [[ ! -f "${run_dir}/metadata.env" || ! -f "${run_dir}/session_id" ]]; then
  echo "Run identity files are incomplete: ${run_dir}" >&2
  exit 3
fi

metadata_run_id="$(sed -n 's/^run_id=//p' "${run_dir}/metadata.env")"
identity_token="$(sed -n 's/^run_identity_token=//p' "${run_dir}/metadata.env")"
session_id="$(cat "${run_dir}/session_id")"

if [[ "$metadata_run_id" != "$run_id" || ! "$session_id" =~ ^[0-9]+$ || ${#identity_token} -lt 20 ]]; then
  echo "Run identity validation failed; refusing to signal anything" >&2
  exit 4
fi

if ! ps -eo sid=,pgid= | awk -v id="$session_id" '$1 == id && $2 == id {found=1} END {exit !found}'; then
  echo "managed process session ${session_id} is not running"
  exit 0
fi

if [[ ! -r "/proc/${session_id}/environ" ]] || \
   ! tr '\0' '\n' <"/proc/${session_id}/environ" | grep -Fxq "PROJECT4_RUN_TOKEN=${identity_token}"; then
  echo "live process does not carry the exact Run identity token; refusing to signal" >&2
  exit 5
fi

echo "sending TERM only to managed process group ${session_id} for Run ${run_id}"
kill -TERM -- "-${session_id}"
for _ in $(seq 1 30); do
  if ! ps -eo sid= | awk -v id="$session_id" '$1 == id {found=1} END {exit !found}'; then
    echo "managed Run stopped cleanly"
    exit 0
  fi
  sleep 1
done

echo "managed process group did not stop in 30 seconds; sending KILL to the same group" >&2
kill -KILL -- "-${session_id}" 2>/dev/null || true

