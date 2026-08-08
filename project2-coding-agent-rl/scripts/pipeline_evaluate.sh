#!/usr/bin/env bash
# Independently evaluate one rollout patch against the hidden-test eval checkout.
# Usage: bash scripts/pipeline_evaluate.sh <run_id> <task_short>
#   run_id:    e.g. deepseek-v4-flash-run10 (directory under PILOT_ROOT/runs/)
#   task_short: e.g. bottlepy-func-basic-0mdlomrj (registry key)
# Writes candidate-evals/<task_short>-run<num>/eval_result.json with full-suite
# counts, FAIL_TO_PASS verdict, and evidence paths.
set -euo pipefail

PILOT_ROOT=/media/imc/data/yzy/agent/project2/swesmith-pilot20
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="$PROJECT_ROOT/scripts/tasks-registry.json"
EVAL_VENV_DIR=/media/imc/data/yzy/agent/project2/eval-venvs
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/swe-tools/bin/python}

RUN_ID="${1:?usage: pipeline_evaluate.sh <run_id> <task_short>}"
TASK_SHORT="${2:?usage: pipeline_evaluate.sh <run_id> <task_short>}"

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
REPO_DIR=$(echo "$task_json" | "$PYTHON_BIN" -c "import json,sys; print(json.load(sys.stdin)['repo_dir'])")
EXTRA_PIP=$(echo "$task_json" | "$PYTHON_BIN" -c "import json,sys; print(' '.join(json.load(sys.stdin).get('test_extra_pip',[])))")

RUN_DIR="$PILOT_ROOT/runs/$RUN_ID"
PATCH="$RUN_DIR/$INSTANCE_ID/$INSTANCE_ID.patch"
if [[ ! -f "$PATCH" ]]; then
  echo "REFUSED: patch not found: $PATCH" >&2
  exit 2
fi

EVAL_REPO="$PILOT_ROOT/eval-repos/$TASK_SHORT"
CANDIDATE="$PILOT_ROOT/candidate-evals/$TASK_SHORT-$RUN_ID"

# 1. Create the hidden-test eval checkout if missing (Bug Patch commit = instance branch tip~1).
if [[ ! -d "$EVAL_REPO" ]]; then
  echo "Creating eval-repo $EVAL_REPO ..."
  git clone -q --branch "$INSTANCE_ID" "file://$PILOT_ROOT/repos/$REPO_DIR" "$EVAL_REPO"
  git -C "$EVAL_REPO" checkout -q "$INSTANCE_ID~1" || { echo "ERROR: no parent commit (Bug Patch) in branch" >&2; exit 3; }
  echo "Eval-repo checked out at $(git -C "$EVAL_REPO" rev-parse --short HEAD)"
fi

# 2. Copy into candidate-evals (never mutate the shared eval-repo), strip
#    build artifacts, and drop binary-file hunks from the patch (models
#    sometimes commit __pycache__/*.pyc; git apply refuses binary hunks
#    without full index lines, which would abort the whole evaluation).
rm -rf "$CANDIDATE"
cp -a "$EVAL_REPO/." "$CANDIDATE/"
find "$CANDIDATE" -name __pycache__ -type d -exec rm -rf {} +
"$PYTHON_BIN" - "$PATCH" "$CANDIDATE/.patch_filter_stats.json" <<'PYEOF'
import json, re, sys

raw = open(sys.argv[1]).read()
# split so each "diff --git" line starts its own section, even at file start
sections = re.split(r"(?m)^(?=diff --git )", raw)
header, kept, stripped = sections[0], [], []
for sec in sections[1:]:
    is_binary = ("__pycache__" in sec.split("\n", 1)[0]
                 or "GIT binary patch" in sec
                 or re.search(r"^Binary files .* differ$", sec, flags=re.M))
    if is_binary:
        m = re.search(r"^diff --git a/(\S+)", sec, flags=re.M)
        stripped.append(m.group(1) if m else "(unnamed)")
        continue
    kept.append(sec)
filtered = header + "".join(kept)
out = sys.argv[1] + ".filtered"
open(out, "w").write(filtered)
json.dump({"binary_entries_stripped": stripped}, open(sys.argv[2], "w"))
print(f"filtered patch -> {out}; stripped {len(stripped)} binary entries")
PYEOF
git -C "$CANDIDATE" apply "$PATCH.filtered" || { echo "ERROR: patch did not apply" >&2; exit 4; }
# Commit the applied patch so setuptools_scm-style packages (e.g. typeguard)
# don't bake a "0.post2+dirty" version into the editable install, which breaks
# bytecode-cache naming at pytest collection time. The candidate is a
# disposable copy, so committing here is harmless.
git -C "$CANDIDATE" add -A
git -C "$CANDIDATE" -c user.name=eval -c user.email=eval@local commit -qm "apply model patch (eval copy)" || true

# 3. Install dependencies into a per-repo venv.
VENV="$EVAL_VENV_DIR/$REPO_DIR"
if [[ ! -x "$VENV/bin/pytest" ]]; then
  echo "Creating eval venv $VENV ..."
  "$PYTHON_BIN" -m venv "$VENV" || { echo "ERROR: venv creation failed" >&2; exit 5; }
  "$VENV/bin/pip" install -q --upgrade pip
fi
"$VENV/bin/pip" install -q -e "$CANDIDATE" pytest $EXTRA_PIP

# 4. Run the full suite from a SHORT-path copy and capture JUnit XML.
#    Some repos' tests strip the checkout path from yielded file names with
#    str.lstrip(path) (a character-set strip, not a prefix strip) — e.g.
#    boltons' test_iter_find_files — which over-strips on long paths and
#    produces false failures. A minimal digit-only path (no letters in common
#    with source basenames) sidesteps this.
#    pytest's non-zero exit for failed tests is an expected outcome here, so
#    the pipeline continues and parses the XML (parse step is the truth).
RUN_NUM=$(echo "$RUN_ID" | grep -oE '[0-9]+$' || echo 0)
SHORT_EVAL=/tmp/e${RUN_NUM}
rm -rf "$SHORT_EVAL"
mkdir -p "$SHORT_EVAL"
cp -a "$CANDIDATE/." "$SHORT_EVAL/"
find "$SHORT_EVAL" -name __pycache__ -type d -exec rm -rf {} +
cd "$SHORT_EVAL"
export PATH="$VENV/bin:$PATH"  # some tests shell out to tools (e.g. mypy) via subprocess
export COLUMNS=300  # mypy wraps error messages at terminal width; wide width keeps them
                   # single-line so typeguard's exact-string test matches (its parser
                   # drops wrapped continuation lines)
set +e
"$VENV/bin/python" -m pytest -q --junitxml=result.xml -o junit_family=xunit2 2>&1 | tee pytest.out | tail -3
set -e
# keep the evidence at the long-path candidate dir for traceability
cp -f result.xml pytest.out "$CANDIDATE/" 2>/dev/null || true

# 5. Parse counts into eval_result.json.
"$PYTHON_BIN" - "$CANDIDATE" "$RUN_ID" "$INSTANCE_ID" "$PATCH" "$TASK_SHORT" <<'PYEOF'
import json, sys
from pathlib import Path
import xml.etree.ElementTree as ET

candidate = Path(sys.argv[1])
run_id = sys.argv[2]
instance_id = sys.argv[3]
patch = sys.argv[4]
task_short = sys.argv[5]

xml_path = candidate / "result.xml"
if not xml_path.exists():
    # pytest died before writing JUnit XML (e.g. plugin incompatibility or
    # collection crash) — record the error instead of crashing the pipeline.
    tail = ""
    out = candidate / "pytest.out"
    if out.exists():
        tail = "".join(out.read_text(errors="replace").splitlines()[-5:])
    result = {
        "instance_id": instance_id,
        "run_id": run_id,
        "eval_time": "2026-08-08",
        "eval_checkout": str(candidate),
        "patch_source": patch,
        "error": f"pytest did not produce result.xml; pytest.out tail: {tail}",
        "reward": None,
    }
    (candidate / "eval_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0)

tree = ET.parse(xml_path)
root = tree.getroot()
# counts live on <testsuite> children, not the <testsuites> root
total = failed = errors = skipped = 0
for suite in root.iter("testsuite"):
    total += int(suite.attrib.get("tests", 0))
    failed += int(suite.attrib.get("failures", 0))
    errors += int(suite.attrib.get("errors", 0))
    skipped += int(suite.attrib.get("skipped", 0))
passed = total - failed - errors - skipped

# FAIL_TO_PASS verdict from the task definition.
instances = json.loads((candidate.parent.parent / "local-instances" / f"{task_short}-sanitized.json").read_text())
f2p = instances[0].get("FAIL_TO_PASS", [])
f2p_failed = []
for tc in f2p:
    for case in root.iter("testcase"):
        cn = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        # accept both pytest dotted classname and task-file slash notation
        keys = {f"{cn}::{name}", f"{cn.replace('.', '/')}.py::{name}"}
        if tc in keys and list(case):
            f2p_failed.append(tc)
            break

# binary hunks stripped before apply (models sometimes commit __pycache__/*.pyc)
strip_stats = candidate / ".patch_filter_stats.json"
binary_stripped = json.loads(strip_stats.read_text()).get("binary_entries_stripped", []) if strip_stats.exists() else []

result = {
    "instance_id": instance_id,
    "run_id": run_id,
    "eval_time": "2026-08-08",
    "eval_venv": str((candidate.parent.parent.parent / "eval-venvs").resolve()),
    "eval_checkout": str(candidate),
    "patch_source": patch,
    "binary_entries_stripped": binary_stripped,
    "full_suite": f"{passed} passed, {failed + errors} failed, {skipped} skipped",
    "fail_to_pass": {
        "total": len(f2p),
        "passed": len(f2p) - len(f2p_failed),
        "failed": f2p_failed,
    },
    # skipped/xfailed tests are expected (typeguard has 9 xfails); the model's
    # job is to fix the F2P tests without breaking anything else.
    "reward": 1 if (failed + errors) == 0 and not f2p_failed else 0,
}
(candidate / "eval_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
PYEOF
