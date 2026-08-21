#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
test_root="$(mktemp -d /tmp/p4-safety-test.XXXXXX)"
fake_bin="${project_dir}/tests/fake_bin"
data_root="${test_root}/data"
mkdir -p "$data_root"

unrelated_pid=""
cleanup() {
  if [[ -n "$unrelated_pid" ]] && kill -0 "$unrelated_pid" 2>/dev/null; then
    kill "$unrelated_pid" 2>/dev/null || true
    wait "$unrelated_pid" 2>/dev/null || true
  fi
  rm -rf -- "$test_root"
}
trap cleanup EXIT

export PATH="${fake_bin}:${PATH}"
export PROJECT4_DATA_ROOT="$data_root"
export PROJECT4_ALLOW_TEST_DATA_ROOT=1
export PROJECT4_MIN_FREE_GIB=0

# Shell syntax is part of the safety gate.
for script in "${project_dir}"/scripts/*.sh; do
  bash -n "$script"
done

# Physical GPU0 must always fail.
if bash -c "source '${project_dir}/scripts/gpu_guard.sh'; project4_validate_gpu_ids 0"; then
  echo "GPU0 guard unexpectedly passed" >&2
  exit 1
fi

# GPU5 needs explicit project-scoped authorization.
if bash -c "source '${project_dir}/scripts/gpu_guard.sh'; project4_validate_gpu_ids 5"; then
  echo "GPU5 guard unexpectedly passed without authorization" >&2
  exit 1
fi
ALLOW_UNSTABLE_GPU5=1 bash -c \
  "source '${project_dir}/scripts/gpu_guard.sh'; project4_validate_gpu_ids 5"

# Unknown compute work must block the target GPU.
if FAKE_BUSY_GPU=1 bash "${project_dir}/scripts/preflight.sh" 1; then
  echo "busy GPU preflight unexpectedly passed" >&2
  exit 1
fi

# A normal managed Run records identity, clears inherited proxies and exits cleanly.
http_proxy=http://127.0.0.1:7890 https_proxy=http://127.0.0.1:7890 \
  bash "${project_dir}/scripts/run_managed.sh" test-ok 1 -- \
  bash -c 'printf "cuda=%s proxy=%s run=%s\n" "$CUDA_VISIBLE_DEVICES" "${http_proxy-unset}" "$PROJECT4_RUN_ID"'

run_dir="${data_root}/project4-opensearch-vl-rl/runs/test-ok"
grep -Fxq 'cuda=1 proxy=unset run=test-ok' "${run_dir}/stdout.log"
grep -Fxq 'exit_code=0' "${run_dir}/metadata.env"
grep -Fxq 'physical_gpu=1 compute_processes=none' "${run_dir}/cleanup.log"

# Existing Run directories are immutable.
if bash "${project_dir}/scripts/run_managed.sh" test-ok 1 -- true; then
  echo "existing Run overwrite unexpectedly passed" >&2
  exit 1
fi

# stop_managed must stop only the exact identified process group.
sleep 60 &
unrelated_pid=$!
bash "${project_dir}/scripts/run_managed.sh" test-stop 1 -- \
  bash -c 'trap "exit 0" TERM; while true; do sleep 1; done' &
managed_wrapper_pid=$!
stop_run_dir="${data_root}/project4-opensearch-vl-rl/runs/test-stop"
for _ in $(seq 1 100); do
  [[ -s "${stop_run_dir}/session_id" ]] && break
  sleep 0.05
done
[[ -s "${stop_run_dir}/session_id" ]]
bash "${project_dir}/scripts/stop_managed.sh" test-stop
wait "$managed_wrapper_pid"
kill -0 "$unrelated_pid"

echo "P0 safety tests: PASS"
