#!/usr/bin/env bash
# Integrity gate for one rollout: no parent history, no hidden-test exposure,
# no test-file edits. Writes candidate-evals/<task_short>-<run_id>/integrity.json
# Usage: bash scripts/pipeline_integrity.sh <run_id> <task_short>
set -euo pipefail

PILOT_ROOT=/media/imc/data/yzy/agent/project2/swesmith-pilot20
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="$PROJECT_ROOT/scripts/tasks-registry.json"
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/swe-tools/bin/python}

RUN_ID="${1:?usage: pipeline_integrity.sh <run_id> <task_short>}"
TASK_SHORT="${2:?usage: pipeline_integrity.sh <run_id> <task_short>}"

task_json=$("$PYTHON_BIN" - "$REGISTRY" "$TASK_SHORT" <<'PYEOF'
import json, sys
reg = json.load(open(sys.argv[1]))
t = reg["tasks"].get(sys.argv[2])
if not t:
    raise SystemExit(f"unknown task: {sys.argv[2]}")
print(json.dumps(t))
PYEOF
)
INSTANCE_ID=$(echo "$task_json" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['instance_id'])")

RUN_DIR="$PILOT_ROOT/runs/$RUN_ID"
PATCH="$RUN_DIR/$INSTANCE_ID/$INSTANCE_ID.patch"
TRAJ="$RUN_DIR/$INSTANCE_ID/$INSTANCE_ID.traj"
if [[ ! -f "$PATCH" ]]; then
  echo "REFUSED: patch not found: $PATCH" >&2
  exit 2
fi

SANITIZED="$PILOT_ROOT/sanitized-repos/$TASK_SHORT"
if [[ -d "$SANITIZED" ]] && git -C "$SANITIZED" cat-file -e HEAD^ 2>/dev/null; then
  echo "REFUSED: sanitized repo has accessible parent history" >&2
  exit 3
fi

"$PYTHON_BIN" - "$RUN_DIR" "$INSTANCE_ID" "$PATCH" "$TRAJ" "$TASK_SHORT" <<'PYEOF'
import json, re, sys
from pathlib import Path

run_dir, instance_id, patch_path, traj_path, task_short = sys.argv[1:6]
patch = Path(patch_path).read_text()

def is_test_path(p: str) -> bool:
    name = Path(p).name.lower()
    parts = {part.lower() for part in Path(p).parts}
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py") or name.endswith("_test.py") or name.endswith(".spec.js")

changed = re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE)
test_edits = [p for p in changed if is_test_path(p)]

traj = json.load(open(traj_path))
steps = traj["trajectory"] if isinstance(traj, dict) and "trajectory" in traj else traj
parent_exposure = False
hidden_exposure = []
for step in steps:
    act = str(step.get("action") or "") if isinstance(step, dict) else ""
    obs = str(step.get("observation") or "") if isinstance(step, dict) else ""
    if re.search(r"git (log|show|reflog)\b", act):
        # shallow repo must only ever expose the single grafted commit
        grafted = re.findall(r"\(grafted[^)]*\)", obs)
        commits = re.findall(r"^[0-9a-f]{7,40}\b", obs, flags=re.MULTILINE)
        if len(commits) > 1:
            parent_exposure = True
    for pat in ("Bug Patch", "HEAD^", "test_patch", "FAIL_TO_PASS"):
        if pat in obs or pat in act:
            hidden_exposure.append(pat)

result = {
    "instance_id": instance_id,
    "run_id": run_dir.split("/")[-1],
    "task_short": task_short,
    "patch_modifies_test_files": test_edits,
    "parent_history_exposed": parent_exposure,
    "hidden_test_or_gold_exposure": hidden_exposure,
    "verdict": "ok" if not test_edits and not parent_exposure and not hidden_exposure else "invalid",
}
out_dir = Path(run_dir) / instance_id
(out_dir / "integrity.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
PYEOF
