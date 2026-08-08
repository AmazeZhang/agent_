#!/usr/bin/env bash
# WP7: evaluate one holdout SWE-agent batch against the hidden-test eval repo.
# Same protocol as pipeline_evaluate.sh (filtered apply -> full pytest ->
# FAIL_TO_PASS verdict) but reads the batch output under
# PILOT_ROOT/holdout-eval/runs/<short>-<variant> and writes evidence under
# PILOT_ROOT/holdout-eval/evaluations/.
#
# A batch whose agent never submitted has model_patch=null in its .pred and
# is recorded as reward 0 with no test run (nothing to apply).
#
# Usage: bash scripts/holdout_evaluate.sh <short>-<variant>
#   e.g. bash scripts/holdout_evaluate.sh funcy-lookuper-3y0j7te5-base
set -euo pipefail

PILOT_ROOT=/media/imc/data/yzy/agent/project2/swesmith-pilot20
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="$PROJECT_ROOT/scripts/tasks-registry.json"
EVAL_VENV_DIR=/media/imc/data/yzy/agent/project2/eval-venvs
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venvs/swe-tools/bin/python}

NAME="${1:?usage: holdout_evaluate.sh <short>-<variant>}"
SHORT="${NAME%-*}"   # strip the last -<variant> suffix (base|sft|grpo)
VARIANT="${NAME##*-}"
TASK_SHORT="$SHORT"  # holdout short names ARE the registry keys / eval-repo names
RUN_DIR="$PILOT_ROOT/holdout-eval/runs/$NAME"
EVAL_OUT="$PILOT_ROOT/holdout-eval/evaluations"
mkdir -p "$EVAL_OUT"

# locate the single instance pred file
PRED=$(find "$RUN_DIR" -name "*.pred" | head -1)
if [[ -z "$PRED" ]]; then
  echo "REFUSED: no .pred found in $RUN_DIR" >&2
  exit 2
fi
INSTANCE_ID=$(basename "$PRED" .pred)

EVAL_REPO="$PILOT_ROOT/eval-repos/$TASK_SHORT"
REPO_DIR=$("$PYTHON_BIN" - "$REGISTRY" "$TASK_SHORT" <<'PYEOF'
import json, sys
reg = json.load(open(sys.argv[1]))
print(reg["tasks"][sys.argv[2]]["repo_dir"])
PYEOF
)

# extract model_patch from the pred file
PATCH="$RUN_DIR/$INSTANCE_ID/$INSTANCE_ID.patch"
"$PYTHON_BIN" - "$PRED" "$PATCH" <<'PYEOF'
import json, sys
pred = json.load(open(sys.argv[1]))
patch = pred.get("model_patch")
if patch:
    open(sys.argv[2], "w").write(patch if patch.endswith("\n") else patch + "\n")
    print(f"model_patch: {len(patch)} chars")
else:
    print("model_patch: null (no submission)")
PYEOF
if [[ ! -s "$PATCH" ]]; then
  result="$EVAL_OUT/$NAME-eval.json"
  cat > "$result" <<JSONEOF
{"instance_id": "$INSTANCE_ID", "variant": "$VARIANT", "task_short": "$TASK_SHORT",
 "eval_time": "$(date +%F)", "submitted": false, "patch_source": null,
 "error": "no submission (agent exhausted calls without submitting)",
 "test_total": 0, "test_failed": 0, "test_errors": 0, "test_skipped": 0,
 "f2p_passed": false, "reward": 0.0, "evidence": "$RUN_DIR/$INSTANCE_ID"}
JSONEOF
  echo "no submission -> $result (reward 0.0)"
  exit 0
fi

# ---- submitted: same protocol as pipeline_evaluate.sh ----
CANDIDATE="$PILOT_ROOT/holdout-eval/candidate-evals/$NAME"
rm -rf "$CANDIDATE"
cp -a "$EVAL_REPO/." "$CANDIDATE/"
find "$CANDIDATE" -name __pycache__ -type d -exec rm -rf {} +
"$PYTHON_BIN" - "$PATCH" "$CANDIDATE/.patch_filter_stats.json" <<'PYEOF'
import json, re, sys
raw = open(sys.argv[1]).read()
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
open(sys.argv[1] + ".filtered", "w").write(filtered)
json.dump({"binary_entries_stripped": stripped}, open(sys.argv[2], "w"))
print(f"filtered; stripped {len(stripped)} binary entries")
PYEOF
if ! git -C "$CANDIDATE" apply "$PATCH.filtered" 2> "$CANDIDATE/.apply.err"; then
  result="$EVAL_OUT/$NAME-eval.json"
  cat > "$result" <<JSONEOF
{"instance_id": "$INSTANCE_ID", "variant": "$VARIANT", "task_short": "$TASK_SHORT",
 "eval_time": "$(date +%F)", "submitted": true, "patch_source": "$PATCH",
 "error": "patch did not apply", "apply_stderr": "$(tail -c 300 "$CANDIDATE/.apply.err")",
 "test_total": 0, "test_failed": 0, "test_errors": 0, "test_skipped": 0,
 "f2p_passed": false, "reward": 0.0, "evidence": "$CANDIDATE"}
JSONEOF
  echo "apply failed -> $result (reward 0.0)"
  exit 0
fi
git -C "$CANDIDATE" add -A
git -C "$CANDIDATE" -c user.name=eval -c user.email=eval@local commit -qm "apply model patch (eval copy)" || true

VENV="$EVAL_VENV_DIR/$REPO_DIR"
"$VENV/bin/pip" install -q -e "$CANDIDATE" >/dev/null 2>&1 || true

SHORT_EVAL=/tmp/e-wp7-$NAME
rm -rf "$SHORT_EVAL"
mkdir -p "$SHORT_EVAL"
cp -a "$CANDIDATE/." "$SHORT_EVAL/"
find "$SHORT_EVAL" -name __pycache__ -type d -exec rm -rf {} +
cd "$SHORT_EVAL"
export PATH="$VENV/bin:$PATH"
export COLUMNS=300
set +e
"$VENV/bin/python" -m pytest -q --junitxml=result.xml -o junit_family=xunit2 2>&1 | tee pytest.out | tail -3
set -e
cp -f result.xml pytest.out "$CANDIDATE/" 2>/dev/null || true

"$PYTHON_BIN" - "$CANDIDATE" "$NAME" "$INSTANCE_ID" "$PATCH" "$TASK_SHORT" "$EVAL_OUT" <<'PYEOF'
import json, sys
from pathlib import Path
import xml.etree.ElementTree as ET

candidate = Path(sys.argv[1])
name = sys.argv[2]
instance_id = sys.argv[3]
patch = sys.argv[4]
task_short = sys.argv[5]
eval_out = Path(sys.argv[6])

xml_path = candidate / "result.xml"
if not xml_path.exists():
    tail = ""
    out = candidate / "pytest.out"
    if out.exists():
        tail = "".join(out.read_text(errors="replace").splitlines()[-5:])
    result = {"instance_id": instance_id, "variant": name.split("-")[-1],
              "task_short": task_short, "eval_time": "2026-08-08",
              "submitted": True, "patch_source": patch,
              "error": f"pytest did not produce result.xml; pytest.out tail: {tail}",
              "reward": None, "evidence": str(candidate)}
else:
    root = ET.parse(xml_path).getroot()
    total = failed = errors = skipped = 0
    for suite in root.iter("testsuite"):
        total += int(suite.attrib.get("tests", 0))
        failed += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))

    # FAIL_TO_PASS from the task registry, dual-format matching
    import json as _json
    reg = _json.load(open("/home/imc/yzy/agent/project2-coding-agent-rl/scripts/tasks-registry.json"))
    f2p = reg["tasks"][task_short].get("fail_to_pass", [])
    f2p_failed = False
    for tc in f2p:
        for case in root.iter("testcase"):
            cn = case.attrib.get("classname", "")
            nm = case.attrib.get("name", "")
            keys = {f"{cn}::{nm}", f"{cn.replace('.', '/')}.py::{nm}"}
            if tc in keys and list(case):
                f2p_failed = True
                break
        if f2p_failed:
            break
    result = {"instance_id": instance_id, "variant": name.split("-")[-1],
              "task_short": task_short, "eval_time": "2026-08-08",
              "submitted": True, "patch_source": patch,
              "test_total": total, "test_failed": failed, "test_errors": errors,
              "test_skipped": skipped, "f2p_failed": f2p_failed,
              "f2p_passed": (failed + errors) == 0 and not f2p_failed,
              "reward": 1.0 if ((failed + errors) == 0 and not f2p_failed)
                        else 0.5 if (failed + errors) == 0 else 0.3,
              "evidence": str(candidate)}

out = eval_out / f"{name}-eval.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False, indent=2))
PYEOF
echo "evaluation -> $EVAL_OUT/$NAME-eval.json"
