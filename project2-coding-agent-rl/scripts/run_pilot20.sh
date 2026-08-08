#!/usr/bin/env bash
# One-click driver for the 20-task SWE-smith pilot pipeline.
#
#   prepare | launch | monitor | summary
#
#   prepare  -- sanitize all registry tasks whose snapshot is missing
#   launch   -- start rollout tmux sessions for runs whose config exists and
#               whose run dir has no submitted result yet (max 2 concurrent)
#   monitor  -- poll running rollouts; evaluate each completed run
#               (integrity gate -> hidden-test eval -> summary)
#   summary  -- rebuild evaluations/summary.json from artifacts
#
# Usage: bash scripts/run_pilot20.sh <phase> [--max-concurrent N] [--dry-run]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PILOT_ROOT=/media/imc/data/yzy/agent/project2/swesmith-pilot20
REGISTRY="$PROJECT_ROOT/scripts/tasks-registry.json"
SCRIPTS="$PROJECT_ROOT/scripts"
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/swe-tools/bin/python}

PHASE="${1:?usage: run_pilot20.sh <prepare|launch|monitor|summary> [--dry-run]}"
MAX_CONCURRENT=${MAX_CONCURRENT:-2}

list_registry_tasks() {
  "$PYTHON_BIN" - "$REGISTRY" <<'PYEOF'
import json, sys
reg = json.load(open(sys.argv[1]))
for short, t in reg["tasks"].items():
    print(f"{short}\t{t['instance_id']}\t{t['repo_dir']}")
PYEOF
}

run_count() {  # <run_id> -> count of submitted exit statuses
  # YAML list items are indented ("    - instance"), so match a dash at any
  # leading whitespace, not just column 0.
  local status_file="$PILOT_ROOT/runs/$1/run_batch_exit_statuses.yaml"
  [[ -f "$status_file" ]] && grep -cE "^[[:space:]]*- " "$status_file" || echo 0
}

case "$PHASE" in
  prepare)
    echo "== prepare: sanitize snapshots for all registry tasks =="
    while IFS=$'\t' read -r short instance repo; do
      bash "$SCRIPTS/pipeline_sanitize.sh" "$short" "$instance" "$repo"
    done < <(list_registry_tasks)
    ;;

  launch)
    echo "== launch: start pending rollouts (max $MAX_CONCURRENT concurrent) =="
    local running=0
    for cfg in "$PILOT_ROOT/local-instances/"deepseek-v4-flash-run*.config.yaml; do
      [[ -e "$cfg" ]] || continue
      run_id=$(basename "$cfg" .config.yaml)
      if [[ "$(run_count "$run_id")" -ge 1 ]]; then
        echo "skip $run_id (already has submission)"
        continue
      fi
      if tmux has-session -t "agent-p2-$run_id" 2>/dev/null; then
        echo "skip $run_id (tmux session alive)"
        continue
      fi
      if [[ "${DRY_RUN:-}" == "1" ]]; then
        echo "[dry] would launch $run_id"
        continue
      fi
      bash "$SCRIPTS/start_rollout_tmux.sh" "$run_id"
      running=$((running + 1))
      [[ $running -ge $MAX_CONCURRENT ]] && { echo "concurrency cap reached ($MAX_CONCURRENT)"; break; }
    done
    echo "launch done"
    ;;

  monitor)
    echo "== monitor: evaluate completed rollouts =="
    for cfg in "$PILOT_ROOT/local-instances/"deepseek-v4-flash-run*.config.yaml; do
      [[ -e "$cfg" ]] || continue
      run_id=$(basename "$cfg" .config.yaml)
      if tmux has-session -t "agent-p2-$run_id" 2>/dev/null; then
        continue  # still running
      fi
      [[ "$(run_count "$run_id")" -ge 1 ]] || { echo "skip $run_id (no submission yet)"; continue; }
      # map run_id -> task_short via the run config's instance id
      task_json=$("$PYTHON_BIN" - "$REGISTRY" "$PILOT_ROOT/local-instances/$run_id.config.yaml" <<'PYEOF'
import json, sys, re
reg = json.load(open(sys.argv[1]))
cfg_text = open(sys.argv[2]).read()
m = re.search(r"instance_id\s*:\s*(\S+)", cfg_text)
if not m:
    raise SystemExit("instance_id not found in config")
for short, t in reg["tasks"].items():
    if t["instance_id"] == m.group(1):
        print(short); break
PYEOF
)
      if [[ -n "$task_json" ]] && [[ "${DRY_RUN:-}" == "1" ]]; then
        echo "[dry] would evaluate $run_id ($task_json)"
        continue
      fi
      echo "== evaluate $run_id ($task_json) =="
      bash "$SCRIPTS/pipeline_integrity.sh" "$run_id" "$task_json"
      bash "$SCRIPTS/pipeline_evaluate.sh" "$run_id" "$task_json"
    done
    "$PYTHON_BIN" "$SCRIPTS/pipeline_summary.py" --write
    ;;

  summary)
    "$PYTHON_BIN" "$SCRIPTS/pipeline_summary.py" --write
    ;;

  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac
