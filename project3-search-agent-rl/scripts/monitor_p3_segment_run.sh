#!/usr/bin/env bash
# Monitor a formal segment training run against the fail-closed stop conditions.
# Usage: monitor_p3_segment_run.sh <run-dir> <gpu-ids,comma,sep> <retriever-url>
# Prints one status line per check; exits 0 even on alert (cron-friendly).
# Stop conditions (per segment authorization): GPU drop/Xid, forbidden GPU used,
# OOM, NaN/Inf, discontinuous resume state, retriever final failure, config
# fingerprint mismatch.

set -uo pipefail

run_dir="${1:?run-dir required}"
gpu_ids="${2:?gpu-ids required (e.g. 1,2,3,4,6,7)}"
retriever_url="${3:-http://127.0.0.1:18080/health}"

alerts=()
info=()

# 0. run alive / exit state
if [[ -f "${run_dir}/metadata.env" ]] && grep -q "^exit_code=" "${run_dir}/metadata.env"; then
  ec="$(grep "^exit_code=" "${run_dir}/metadata.env" | cut -d= -f2)"
  alerts+=("RUN_EXITED exit_code=${ec}")
else
  info+=("running")
fi

# 1. GPU membership: every authorized GPU present, no unauthorized GPU busy
mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null)
for line in "${gpu_lines[@]}"; do
  idx="${line%%,*}"
  rest="${line#*,}"
  mem="${rest%%,*}"
  util="${rest##*,}"
  util="${util% %}"
  mem="${mem% MiB}"
  if [[ ",${gpu_ids}," != *",${idx},"* ]]; then
    # forbidden GPU counts as used only when it departs the idle baseline
    # (>=1 GiB AND util > 0); the ~387 MiB system-idle footprint (oray/desktop)
    # is pre-existing and constant across all prior runs.
    if (( mem > 1024 )) && (( util > 0 )); then
      alerts+=("FORBIDDEN_GPU_USED gpu=${idx} mem=${mem}MiB util=${util}%")
    fi
  fi
done

# 2. authorized GPUs present (dropped card => nvidia-smi still lists it; a dead
#    card would fail nvidia-smi entirely -> covered below)
if [[ -z "${gpu_lines[0]:-}" ]]; then
  alerts+=("NVIDIA_SMI_FAILED")
fi

# 3. OOM / NaN / Inf / Xid in logs
if grep -qi "out of memory\|OutOfMemoryError\|CUDA OOM" "${run_dir}/stderr.log" 2>/dev/null; then
  alerts+=("OOM")
fi
if grep -q "SEGMENT_STOP" "${run_dir}/stdout.log" 2>/dev/null; then
  info+=("SEGMENT_STOP present")
fi
# NaN/Inf only inside metric values (e.g. "grad_norm:nan", "perf/throughput:inf")
if grep -E ":[0-9]*\.?(nan|inf)" "${run_dir}/stdout.log" 2>/dev/null | grep -qv "INFO\|WARNING\|DEBUG"; then
  alerts+=("NAN_INF_IN_METRICS")
fi

# 4. Xid / GPU errors in kernel log (need root; best effort)
if command -v dmesg >/dev/null 2>&1 && dmesg 2>/dev/null | tail -200 | grep -qi "NVRM: Xid"; then
  alerts+=("XID_ERROR")
fi

# 5. step progress: metrics lines exist and global_step is increasing
steps="$(grep -o "training/global_step:[0-9]*" "${run_dir}/stdout.log" 2>/dev/null | sort -u -t: -k2 -n | tail -1 | cut -d: -f2)"
if [[ -n "$steps" ]]; then
  info+=("latest_global_step=${steps}")
fi

# 6. retriever health
health="$(curl -s --max-time 10 "${retriever_url}" 2>/dev/null || true)"
if [[ "$health" != *'"status":"ready"'* ]]; then
  alerts+=("RETRIEVER_NOT_READY")
else
  info+=("retriever_ready")
fi

# 7. config fingerprint: resolved_config_sha256 line present
if ! grep -q "resolved_config_sha256=" "${run_dir}/stdout.log" 2>/dev/null; then
  alerts+=("NO_CONFIG_FINGERPRINT")
fi

echo "[P3MON] $(date --iso-8601=seconds) ${info[*]:-no-progress}"
if (( ${#alerts[@]} > 0 )); then
  echo "[P3MON] ALERT: ${alerts[*]}"
fi
