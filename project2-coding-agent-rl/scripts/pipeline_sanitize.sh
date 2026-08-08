#!/usr/bin/env bash
# Create a sanitized single-commit snapshot for one SWE-smith task, refusing
# any repo whose parent history would be accessible. Also writes the
# local-instances/<task_short>-sanitized.json rollout config.
# Usage: bash scripts/pipeline_sanitize.sh <task_short> <instance_id> <repo_dir> [--force]
set -euo pipefail

PILOT_ROOT=/media/imc/data/yzy/agent/project2/swesmith-pilot20
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK_SHORT="${1:?usage: pipeline_sanitize.sh <task_short> <instance_id> <repo_dir>}"
INSTANCE_ID="${2:?usage: pipeline_sanitize.sh <task_short> <instance_id> <repo_dir>}"
REPO_DIR="${3:?usage: pipeline_sanitize.sh <task_short> <instance_id> <repo_dir>}"
FORCE="${4:-}"

REPO_SRC="$PILOT_ROOT/repos/$REPO_DIR"
DEST="$PILOT_ROOT/sanitized-repos/$TASK_SHORT"
INSTANCE_JSON="$PILOT_ROOT/local-instances/$TASK_SHORT-sanitized.json"

if [[ -d "$DEST" && "$FORCE" != "--force" ]]; then
  echo "SKIP: $DEST already exists (use --force to recreate)"
  exit 0
fi

if [[ ! -d "$REPO_SRC" ]]; then
  echo "ERROR: source repo missing: $REPO_SRC (clone it first)" >&2
  exit 2
fi

# The branch must exist locally before file:// shallow cloning (clone only
# exposes refs/heads/*, not refs/remotes/origin/*).
if ! git -C "$REPO_SRC" cat-file -e "origin/$INSTANCE_ID^{commit}" 2>/dev/null; then
  git -C "$REPO_SRC" fetch -q origin "+refs/heads/*:refs/remotes/origin/*"
fi
git -C "$REPO_SRC" branch "$INSTANCE_ID" "origin/$INSTANCE_ID" 2>/dev/null || true

rm -rf "$DEST"
if ! git clone -q --depth 1 --branch "$INSTANCE_ID" "file://$REPO_SRC" "$DEST"; then
  echo "ERROR: shallow clone failed" >&2
  exit 3
fi

if git -C "$DEST" cat-file -e HEAD^ 2>/dev/null; then
  echo "ERROR: snapshot has accessible parent history; refusing" >&2
  exit 4
fi

"$PROJECT_ROOT/.venvs/swe-tools/bin/python" \
  "$PROJECT_ROOT/scripts/materialize_local_swesmith_instance.py" \
  "$PILOT_ROOT/sweagent_instances.json" "$INSTANCE_ID" "$DEST" "$INSTANCE_JSON" >/dev/null

echo "OK: sanitized snapshot at $DEST"
echo "OK: instance config at $INSTANCE_JSON"
