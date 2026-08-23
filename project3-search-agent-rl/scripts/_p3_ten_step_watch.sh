#!/bin/bash
# P3 v2 ten-step: background watcher (Phase 5).
#  1. every 120s runs the per-step audit (10 abort conditions) on the live run
#  2. writes a VIOLATION marker and exits immediately if any_violation=true
#  3. samples nvidia-smi per-GPU memory peaks every 2s for Phase-7 reporting
#  4. exits when the training tmux session dies / stdout shows run completion
set -u

PROJECT=/home/imc/yzy/agent/project3-search-agent-rl
RUN=/media/imc/data/project3-search-agent-rl/runs/p3-search-aware-clean-v2-grpo10-fsdp6-b66-n5-s0-20260823a
SESSION=p3-p3-search-aware-clean-v2-grpo10-fsdp6-b66-n5-s0-20260823a
STATE=$PROJECT/gates/p3_ten_step_audit_20260823a.json
PEAKS=$PROJECT/gates/p3_ten_step_gpu_peaks_20260823a.json
EXPECTED_SHA=d727b64f7c1c235e1d070637d9af498a02b1b89868bce088afdc19b814358402
LOG=$PROJECT/gates/p3_ten_step_watch_20260823a.log

mkdir -p "$(dirname "$STATE")"
: > "$LOG"
echo "[watch $(date '+%F %T')] started" >> "$LOG"

# nvidia-smi peak sampler (2s cadence, atomic write, per-GPU max used_memory)
(
  declare -A PEAK
  N=0
  while true; do
    SNAP=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    while IFS=',' read -r idx mem; do
      idx=$(echo "$idx" | xargs); mem=$(echo "$mem" | xargs)
      [ -z "$mem" ] && continue
      cur=${PEAK[$idx]:-0}
      if [ "$mem" -gt "$cur" ]; then PEAK[$idx]=$mem; fi
    done <<< "$SNAP"
    N=$((N+1))
    {
      echo "{"
      echo "  \"kind\": \"p3-ten-step-gpu-peaks\","
      echo "  \"run_id\": \"$(basename "$RUN")\","
      echo "  \"samples\": $N,"
      echo "  \"peak_used_mib_by_index\": {"
      first=1
      for idx in 0 1 2 3 4 5 6 7; do
        if [ $first -eq 0 ]; then echo -n ", "; else echo -n ""; fi
        first=0
        echo -n "\"$idx\": ${PEAK[$idx]:-0}"
      done
      echo ""
      echo "  }"
      echo "}"
    } > "${PEAKS}.partial"
    mv "${PEAKS}.partial" "$PEAKS"
    sleep 2
  done
) &
SAMPLER_PID=$!
echo "[watch $(date '+%F %T')] sampler pid=$SAMPLER_PID" >> "$LOG"

# audit loop
while true; do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[watch $(date '+%F %T')] training session gone -> stopping" >> "$LOG"
    break
  fi
  cd "$PROJECT"
  OUT=$(CUDA_VISIBLE_DEVICES='' python3 scripts/audit_p3_ten_step.py \
      --run "$RUN" --state "$STATE" --expected-config-sha "$EXPECTED_SHA" 2>&1)
  RC=$?
  VIOL=$(echo "$OUT" | grep -c 'any_violation=True' || true)
  echo "[watch $(date '+%F %T')] audit rc=$RC violation=$VIOL" >> "$LOG"
  echo "$OUT" >> "$LOG"
  if [ "$VIOL" -gt 0 ]; then
    echo "[watch $(date '+%F %T')] VIOLATION -> abort marker" >> "$LOG"
    echo "$(date '+%F %T') VIOLATION" > "$PROJECT/gates/p3_ten_step_VIOLATION_20260823a.marker"
    break
  fi
  sleep 120
done

kill "$SAMPLER_PID" 2>/dev/null
echo "[watch $(date '+%F %T')] watcher stopped (sampler killed)" >> "$LOG"
