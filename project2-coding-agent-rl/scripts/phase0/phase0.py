#!/usr/bin/env python3
"""Phase 0 harness: verify that a model can *submit* on our holdout tasks.

Three subcommands:

  run  <short>   -- run the agent (mini-swe-agent loop) on one holdout task
                   against a fresh work copy of the eval repo, with the
                   task's venv on PATH. On `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
                   we collect `git diff HEAD` as the model patch.
  eval <short>   -- evaluate a patch file (or the work copy diff) with the
                   same protocol as the reward evaluator: filtered apply ->
                   full pytest -> FAIL_TO_PASS verdict, plus audit fields.
  gate           -- gate-2 matrix on the two tasks with verified patches:
                   verified patch -> 1.0, empty patch -> 0.0, garbage patch -> 0.0,
                   tests_modified -> 0.0, zero-collected-tests -> 0.0.

Usage examples:
  python3 scripts/phase0/phase0.py run boltons-7nlifqzn --model swe-master-4b-rl \
      --protocol textbased --step-limit 40
  python3 scripts/phase0/phase0.py eval boltons-7nlifqzn --patch /path/to/x.patch
  python3 scripts/phase0/phase0.py eval boltons-7nlifqzn --from-work
  python3 scripts/phase0/phase0.py gate
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/imc/yzy/agent/project2-coding-agent-rl")
PILOT_ROOT = Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20")
DATA_ROOT = Path("/media/imc/data/yzy/agent/project2")
PHASE0_ROOT = DATA_ROOT / "phase0"

COMPLETE = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# --------------------------------------------------------------------------
# task plumbing
# --------------------------------------------------------------------------


def load_task(short: str) -> dict:
    """Load task definition from the sanitized local-instance JSON."""
    p = PILOT_ROOT / "local-instances" / f"{short}-sanitized.json"
    inst = json.loads(p.read_text())
    if isinstance(inst, list):
        inst = inst[0]
    return inst


def registry() -> dict:
    return json.loads((PROJECT_ROOT / "scripts" / "tasks-registry.json").read_text())


def repo_dir_of(short: str) -> str:
    return registry()["tasks"][short]["repo_dir"]


# eval-venvs directory names diverge from the registry repo_dir for some
# tasks (registry was authored later and never exercised for those); pin the
# actual names for the phase-0 holdout set.
VENV_OVERRIDES = {
    "funcy-lookuper-3y0j7te5": "funcy-py311",
    "funcy-curry-compose-3u9hti2d": "funcy-py311",
}

# funcy tasks' test deps live in test_requirements.txt but the registry
# test_extra_pip is empty for them; pin them here (pytest==7.4.3, whatever).
EXTRA_PIP_OVERRIDES = {
    "funcy-lookuper-3y0j7te5": ["pytest==7.4.3", "whatever==0.7"],
    "funcy-curry-compose-3u9hti2d": ["pytest==7.4.3", "whatever==0.7"],
}


def venv_of(short: str) -> Path:
    name = VENV_OVERRIDES.get(short, repo_dir_of(short))
    return Path("/media/imc/data/yzy/agent/project2/eval-venvs") / name


def extra_pip_of(short: str) -> list[str]:
    return EXTRA_PIP_OVERRIDES.get(
        short, registry()["tasks"][short].get("test_extra_pip", [])
    )


def clean_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    for pyc in dst.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)


def ensure_venv(venv: Path, extra_pip: list[str]) -> None:
    """Create the task venv if missing and install pip/pytest + extras (same
    as pipeline_evaluate.sh). Rebuilds broken shell venvs (python present but
    no pip)."""
    base_py = "/home/imc/yzy/agent/project2-coding-agent-rl/.venvs/rllm-base/bin/python"
    if not (venv / "bin" / "python").exists():
        subprocess.run([base_py, "-m", "venv", str(venv)], check=True)
    if not (venv / "bin" / "pip").exists():
        # broken shell venv (e.g. funcy-py311 has python but no pip/ensurepip);
        # rebuild from scratch
        shutil.rmtree(venv)
        subprocess.run([base_py, "-m", "venv", str(venv)], check=True)
    needs_reinstall = not (venv / "bin" / "pytest").exists()
    if any(p.startswith("pytest") for p in extra_pip):
        needs_reinstall = True  # version constraint in extras (e.g. pytest<8)
    if needs_reinstall:
        subprocess.run([str(venv / "bin" / "pip"), "install", "-q", "--upgrade", "pip"],
                       check=False, stdout=subprocess.DEVNULL)
        subprocess.run([str(venv / "bin" / "pip"), "install", "-q", *extra_pip],
                       check=False, stdout=subprocess.DEVNULL)


def reinstall_editable(venv: Path, repo: Path) -> None:
    """Point the task venv's editable install at `repo` so pytest tests it."""
    subprocess.run(
        [str(venv / "bin" / "pip"), "install", "-q", "-e", str(repo)],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_pytest(venv: Path, repo: Path, *, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run full pytest from a SHORT path copy (long paths break tests that
    lstrip paths, e.g. boltons test_iter_find_files — same fix as
    pipeline_evaluate.sh)."""
    short_eval = Path(f"/tmp/e-p0-{repo.name}")
    if short_eval.exists():
        shutil.rmtree(short_eval)
    shutil.copytree(repo, short_eval)
    for pyc in short_eval.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)
    env = {**os.environ, "PATH": f"{venv / 'bin'}:" + os.environ.get("PATH", ""),
           "VIRTUAL_ENV": str(venv), "COLUMNS": "300"}
    proc = subprocess.run(
        [str(venv / "bin" / "python"), "-m", "pytest", "-q",
         "--junitxml", "result.xml", "-o", "junit_family=xunit2"],
        cwd=short_eval, env=env, capture_output=True, text=True, timeout=timeout,
    )
    # keep the evidence next to the repo for traceability (copy: /tmp and
    # /media may be on different filesystems, rename would fail)
    shutil.copyfile(short_eval / "result.xml", repo / "result.xml")
    return proc


# --------------------------------------------------------------------------
# safe local environment (mini-swe-agent LocalEnvironment subclass)
# --------------------------------------------------------------------------

BLOCKLIST = [
    "rm -rf", "rm -fr",
    "sudo", "mkfs", "dd if=", "shutdown", "reboot", "mount", "umount",
    "fdisk", "passwd", "chown", "systemctl", "killall", "pkill",
    "docker", "podman", "singularity",
    "git push", "git remote",
    "curl", "wget", "ncat", "telnet",
    "pip install", "pip3 install", "python -m venv",
    "make ",
]


def _truncate(text: str, limit: int = 6000) -> str:
    """Keep the head and tail of long command output (errors live at the end)."""
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-(limit // 2):]
    return f"{head}\n... [output truncated: {len(text)} chars] ...\n{tail}"


def blocked(command: str) -> str | None:
    for pat in BLOCKLIST:
        if pat in command:
            return pat
    return None


class Phase0Env:
    """Minimal environment matching the minisweagent Environment protocol.

    Executes commands in a subshell under the task venv, blocks dangerous
    commands, and on the COMPLETE marker collects `git diff HEAD` as the
    submission.
    """

    def __init__(self, *, workdir: Path, venv: Path, timeout: int = 120):
        self.config = {"cwd": str(workdir), "env": {}, "timeout": timeout}
        self.workdir = workdir
        self.venv = venv

    def env_vars(self) -> dict:
        env = {
            "PATH": f"{self.venv / 'bin'}:" + os.environ.get("PATH", ""),
            "VIRTUAL_ENV": str(self.venv),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R",
            "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1",
            "COLUMNS": "300",
        }
        return env

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        # SWE-agent style submission: a bare `exit` submits without shell
        # execution (bash would otherwise fail with `exit: command not found`).
        if command.strip() == "exit":
            from minisweagent.exceptions import Submitted
            submission = self.collect_diff()
            raise Submitted({
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            })
        if (pat := blocked(command)) is not None:
            return {
                "output": f"BLOCKED by safety filter: command contains '{pat}'. "
                          "Re-run the tests/commands without the blocked tool.",
                "returncode": -1, "exception_info": "",
            }
        try:
            proc = subprocess.run(
                command, shell=True, text=True, cwd=self.workdir,
                env={**os.environ, **self.env_vars()},
                encoding="utf-8", errors="replace",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout or self.config["timeout"],
            )
            output = {"output": _truncate(proc.stdout), "returncode": proc.returncode, "exception_info": ""}
        except subprocess.TimeoutExpired as e:
            output = {
                "output": _truncate((e.stdout or "") + f"\n[TIMEOUT after {timeout or self.config['timeout']}s]"),
                "returncode": -1,
                "exception_info": f"command timed out after {timeout or self.config['timeout']}s",
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        from minisweagent.exceptions import Submitted
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        first = lines[0].strip() if lines else ""
        # COMPLETE marker (mini-swe-agent default) or SWE-agent `exit` submit.
        is_submit = (first == COMPLETE or first == "exit") and output["returncode"] == 0
        if is_submit:
            submission = self.collect_diff()
            raise Submitted({
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            })

    def collect_diff(self) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.workdir), "diff", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            return proc.stdout
        except Exception as e:
            return f"# failed to collect git diff: {e}"

    def get_template_vars(self, **kwargs) -> dict:
        return {"cwd": str(self.workdir), **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": "phase0-local", "workdir": str(self.workdir)}}}


# --------------------------------------------------------------------------
# templates (mini-swe-agent style, SWE-bench flavored instance template)
# --------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact with a computer.
You are working inside a software repository on a Linux machine. The repository
already has the project installed in editable mode with its dependencies. You
can execute bash commands to explore, edit files, run tests, and verify your
fix.

## Recommended Workflow

This workflow should be done step-by-step so that you can iterate on your
changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   (or the single command `exit`). Do not combine it with any other command.
   <important>After submitting, you cannot continue working on this task.</important>

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one bash tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- Do not modify or delete test files. The final submission must only change source code.
- STOP right after your command tag. Never write the command's result yourself:
  do not emit `<tool_response>`, `Exit code:`, `Execution Success:`,
  `Execution Output:`, or any output text after `</command>`. The environment
  appends the real result immediately after your command executes. One command
  per response only.
- Submit your changes and finish your work by issuing the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  (or the single command `exit`). Do not combine it with any other command.
  <important>After submitting, you cannot continue working on this task.</important>

## Useful command examples

### Edit files with sed:

```bash
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Run tests:

```bash
python -m pytest tests/ -x -q
```
"""

INSTANCE_TEMPLATE = """\
<uploaded_files>
{{cwd}}
</uploaded_files>
I've uploaded a python code repository in the directory {{cwd}}. Consider the
following PR description:

<pr_description>
{{task}}
</pr_description>

Make the minimal changes to non-test files needed to satisfy the PR
description. Inspect only the directly relevant code, reproduce the bug, and
implement promptly; do not spend many calls surveying existing tests. Rerun
the reproduction and focused tests, then submit. Do not modify tests.
"""

OBSERVATION_TEMPLATE = """\
{%- if output.exception_info %}<exception>{{output.exception_info}}</exception>
{% endif -%}
Exit code: {{output.returncode}}
Execution Success: {{'true' if output.returncode == 0 else 'false'}}
Execution Output: [STDOUT]

{{output.output}}
"""

FORMAT_ERROR_TEMPLATE = """\
Your previous response was not in the expected format. Every response MUST end
with exactly one bash command in the required format. If you are ready to
submit, issue exactly:
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` (or the single command `exit`)
Respond now with one tool call.
"""


# --------------------------------------------------------------------------
# subcommand: run
# --------------------------------------------------------------------------


def make_model(args, short: str):
    """Build the minisweagent model object for the local vLLM endpoint."""
    base_url = getattr(args, "base_url", "http://127.0.0.1:8012/v1")
    model_kwargs = {
        "api_base": base_url,
        "api_key": "EMPTY",
        "max_tokens": getattr(args, "max_tokens", 2048),
        "temperature": getattr(args, "temperature", 0.0),
        "drop_params": True,
    }
    if args.protocol == "toolcall":
        from minisweagent.models.litellm_model import LitellmModel
        model = LitellmModel(
            model_name=f"openai/{args.model}",
            model_kwargs=model_kwargs,
            observation_template=OBSERVATION_TEMPLATE,
            format_error_template=FORMAT_ERROR_TEMPLATE,
            cost_tracking="ignore_errors",
        )
    else:
        from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel
        model = LitellmTextbasedModel(
            model_name=f"openai/{args.model}",
            model_kwargs=model_kwargs,
            observation_template=OBSERVATION_TEMPLATE,
            format_error_template=FORMAT_ERROR_TEMPLATE,
            cost_tracking="ignore_errors",
        )
    return model


def cmd_run(args) -> None:
    from minisweagent.agents.default import DefaultAgent

    short = args.short
    task = load_task(short)
    workdir = PHASE0_ROOT / "work" / short

    print(f"[phase0] task {short}: {task['instance_id']}", flush=True)
    print(f"[phase0] problem: {task['problem_statement'][:120].splitlines()[0]}", flush=True)

    # fresh work copy of the eval repo (bug state, F2P tests present)
    clean_copy(PILOT_ROOT / "eval-repos" / short, workdir)
    venv = venv_of(short)
    reinstall_editable(venv, workdir)
    print(f"[phase0] work copy at {workdir}, editable install pointed at it", flush=True)

    model = make_model(args, short)
    env = Phase0Env(workdir=workdir, venv=venv, timeout=args.timeout)
    agent = DefaultAgent(
        model, env,
        system_template=SYSTEM_TEMPLATE,
        instance_template=INSTANCE_TEMPLATE,
        step_limit=args.step_limit,
        cost_limit=0.0,          # disable cost limit (local endpoint)
        max_consecutive_format_errors=5,
        output_path=PHASE0_ROOT / "trajs" / f"{short}.traj.json",
    )
    info = agent.run(task["problem_statement"])
    exit_status = info.get("exit_status", "Unknown")
    submission = info.get("submission", "")
    n_calls = agent.n_calls

    (PHASE0_ROOT / "preds").mkdir(parents=True, exist_ok=True)
    pred = {
        "instance_id": task["instance_id"],
        "task_short": short,
        "model_name": args.model,
        "protocol": args.protocol,
        "step_limit": args.step_limit,
        "n_calls": n_calls,
        "exit_status": exit_status,
        "model_patch": submission,
        "workdir": str(workdir),
        "traj": str(PHASE0_ROOT / "trajs" / f"{short}.traj.json"),
    }
    out = PHASE0_ROOT / "preds" / f"{short}.json"
    out.write_text(json.dumps(pred, ensure_ascii=False, indent=2))
    print(f"[phase0] exit_status={exit_status} n_calls={n_calls} "
          f"patch_chars={len(submission)} -> {out}", flush=True)


# --------------------------------------------------------------------------
# subcommand: eval
# --------------------------------------------------------------------------


def patch_from_work(short: str) -> str:
    workdir = PHASE0_ROOT / "work" / short
    proc = subprocess.run(["git", "-C", str(workdir), "diff", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout


def eval_patch(short: str, patch_text: str) -> dict:
    """Evaluate a patch with the reward-evaluator protocol + audit fields."""
    task = load_task(short)
    venv = venv_of(short)
    extra_pip = extra_pip_of(short)
    ensure_venv(venv, extra_pip)
    cand = PHASE0_ROOT / "candidate-evals" / short
    clean_copy(PILOT_ROOT / "eval-repos" / short, cand)

    audit = {
        "instance_id": task["instance_id"],
        "task_short": short,
        "valid_action": bool(patch_text.strip()),
        "patch_nonempty": bool(patch_text.strip()),
        "patch_applied": False,
        "tests_modified": False,
        "tests_collected": 0,
        "test_failed": 0,
        "test_errors": 0,
        "test_skipped": 0,
        "fail_to_pass_matched": 0,
        "fail_to_pass_passed": 0,
        "eval_infra_error": False,
        "error": None,
    }

    if not patch_text.strip():
        audit["error"] = "empty patch"
        audit["final_reward"] = 0.0
        return audit

    # filter binary entries (same as holdout_evaluate.sh)
    sections = re.split(r"(?m)^(?=diff --git )", patch_text)
    header, kept, stripped = sections[0], [], []
    for sec in sections[1:]:
        first = sec.split("\n", 1)[0]
        if ("__pycache__" in first or "GIT binary patch" in sec
                or re.search(r"^Binary files .* differ$", sec, flags=re.M)):
            stripped.append(first)
            continue
        kept.append(sec)
    filtered = header + "".join(kept)
    if stripped:
        audit["binary_entries_stripped"] = stripped

    # tests-modified detection: any diff touching test paths
    test_path_re = re.compile(r"^diff --git a/(tests?/|.*_test\.py|test_|.*/test_)", re.M)
    if test_path_re.search(filtered):
        audit["tests_modified"] = True

    patch_file = cand / "model.patch"
    patch_file.write_text(filtered if filtered.endswith("\n") else filtered + "\n")

    if not audit["tests_modified"]:
        apply = subprocess.run(
            ["git", "-C", str(cand), "apply", "--check", str(patch_file)],
            capture_output=True, text=True,
        )
        if apply.returncode != 0:
            audit["error"] = f"patch did not apply: {apply.stderr.strip()[:300]}"
            audit["final_reward"] = 0.0
            return audit
        subprocess.run(["git", "-C", str(cand), "apply", str(patch_file)], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(cand), "add", "-A"], check=True, capture_output=True)
        audit["patch_applied"] = True
    else:
        audit["error"] = "tests_modified (patch touches test files)"
        audit["final_reward"] = 0.0
        return audit

    reinstall_editable(venv, cand)
    # commit applied patch so setuptools_scm-style packages don't bake a
    # "dirty" version into the editable install (same fix as pipeline)
    subprocess.run(["git", "-C", str(cand), "-c", "user.name=eval",
                    "-c", "user.email=eval@local", "commit", "-qm",
                    "apply model patch (eval copy)"],
                   check=False, capture_output=True)

    # full pytest (from a SHORT path copy)
    pytest = run_pytest(venv, cand, timeout=600)
    xml_path = cand / "result.xml"
    if not xml_path.exists():
        audit["error"] = (f"pytest produced no result.xml (rc={pytest.returncode}); "
                          f"tail: {' | '.join(pytest.stdout.splitlines()[-3:])}")
        audit["eval_infra_error"] = True
        audit["final_reward"] = 0.0
        return audit

    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    total = failed = errors = skipped = 0
    for suite in root.iter("testsuite"):
        total += int(suite.attrib.get("tests", 0))
        failed += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))
    audit["tests_collected"] = total
    audit["test_failed"] = failed
    audit["test_errors"] = errors
    audit["test_skipped"] = skipped

    if total == 0:
        audit["error"] = "pytest collected 0 tests (evaluation anomaly)"
        audit["eval_infra_error"] = True
        audit["final_reward"] = 0.0
        return audit

    # baseline exclusion: environment-inherent failures (see cmd_baseline)
    baseline_path = PHASE0_ROOT / "baseline" / f"{short}.json"
    baseline_failed = set()
    if baseline_path.exists():
        baseline_failed = set(json.loads(baseline_path.read_text())["baseline_failed_tests"])
    audit["baseline_failed_tests"] = sorted(baseline_failed)

    failed_tests = []
    for suite in root.iter("testsuite"):
        for tc in suite.iter("testcase"):
            for child in tc:
                if child.tag in ("failure", "error"):
                    cn = tc.attrib.get("classname", "")
                    nm = tc.attrib.get("name", "")
                    failed_tests.append(f"{cn}::{nm}")
    delta_failed = [t for t in failed_tests if t not in baseline_failed]
    audit["delta_failed_tests"] = delta_failed
    audit["delta_failed"] = len(delta_failed)

    # FAIL_TO_PASS matching (dual-format, same as pipeline_evaluate.sh)
    f2p = task.get("FAIL_TO_PASS", [])
    for tc in f2p:
        for case in root.iter("testcase"):
            cn = case.attrib.get("classname", "")
            nm = case.attrib.get("name", "")
            keys = {f"{cn}::{nm}", f"{cn.replace('.', '/')}.py::{nm}"}
            if tc in keys:
                audit["fail_to_pass_matched"] += 1
                if not list(case):  # empty element = passed
                    audit["fail_to_pass_passed"] += 1
                break

    p2p_clean = len(delta_failed) == 0
    f2p_all_passed = audit["fail_to_pass_passed"] == len(f2p)
    if p2p_clean:
        audit["final_reward"] = 1.0 if f2p_all_passed else 0.5
    else:
        audit["final_reward"] = 0.3
    return audit


def cmd_eval(args) -> None:
    short = args.short
    if args.from_work:
        patch_text = patch_from_work(short)
        patch_src = "work-diff"
    else:
        p = Path(args.patch)
        patch_text = p.read_text() if p.exists() else args.patch
        patch_src = str(p) if p.exists() else "inline"
    audit = eval_patch(short, patch_text)
    audit["patch_source"] = patch_src
    (PHASE0_ROOT / "evals").mkdir(parents=True, exist_ok=True)
    out = PHASE0_ROOT / "evals" / f"{short}.json"
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"[phase0] -> {out}")


def cmd_baseline(args) -> None:
    """Run full pytest on the clean eval repo and record baseline-failing tests.

    Baseline failures (environment-inherent, e.g. Python-version drift) are
    excluded from P2P verdicts in eval_patch.
    """
    short = args.short
    venv = venv_of(short)
    extra_pip = extra_pip_of(short)
    ensure_venv(venv, extra_pip)
    cand = PHASE0_ROOT / "baseline-work" / short
    clean_copy(PILOT_ROOT / "eval-repos" / short, cand)
    reinstall_editable(venv, cand)
    pytest = run_pytest(venv, cand, timeout=900)
    xml_path = cand / "result.xml"
    if not xml_path.exists():
        print(f"[baseline] {short}: FAILED - no result.xml; rc={pytest.returncode}", flush=True)
        return
    import xml.etree.ElementTree as ET
    root = ET.parse(xml_path).getroot()
    total = failed = errors = skipped = 0
    failures = []
    for suite in root.iter("testsuite"):
        total += int(suite.attrib.get("tests", 0))
        failed += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))
        for tc in suite.iter("testcase"):
            for child in tc:
                if child.tag in ("failure", "error"):
                    cn = tc.attrib.get("classname", "")
                    nm = tc.attrib.get("name", "")
                    failures.append(f"{cn}::{nm}")
    baseline = {
        "task_short": short,
        "tests_collected": total,
        "baseline_failed": failed + errors,
        "baseline_failed_tests": failures,
    }
    (PHASE0_ROOT / "baseline").mkdir(parents=True, exist_ok=True)
    out = PHASE0_ROOT / "baseline" / f"{short}.json"
    out.write_text(json.dumps(baseline, ensure_ascii=False, indent=2))
    print(f"[baseline] {short}: {total} collected, {len(failures)} baseline failures -> {out}",
          flush=True)
    print(json.dumps(baseline, ensure_ascii=False, indent=2), flush=True)


# --------------------------------------------------------------------------
# subcommand: gate
# --------------------------------------------------------------------------

VERIFIED_PATCHES = {
    "boltons-7nlifqzn": DATA_ROOT / "training-data" / "successes"
        / "deepseek-v4-flash-run15"
        / "mahmoud__boltons.3bfcfdd0.combine_file__7nlifqzn.patch",
    "funcy-lookuper-3y0j7te5": DATA_ROOT / "training-data" / "successes"
        / "deepseek-v4-flash-run8-sanitized"
        / "Suor__funcy.207a7810.combine_file__3y0j7te5.patch",
}


def cmd_gate(args) -> None:
    results = {}
    for short, vp in VERIFIED_PATCHES.items():
        if not vp.exists():
            print(f"[gate] SKIP {short}: verified patch missing at {vp}", flush=True)
            continue
        print(f"[gate] {short}: verified patch -> expect 1.0", flush=True)
        a1 = eval_patch(short, vp.read_text())
        results[f"{short}-verified"] = a1["final_reward"]
        print(f"[gate] {short}: empty patch -> expect 0.0", flush=True)
        a2 = eval_patch(short, "")
        results[f"{short}-empty"] = a2["final_reward"]
        print(f"[gate] {short}: garbage patch -> expect 0.0", flush=True)
        a3 = eval_patch(short, "diff --git a/boltons/jsonutils.py b/boltons/jsonutils.py\n"
                              "index 0000000..1111111 100644\n--- a/boltons/jsonutils.py\n"
                              "+++ b/boltons/jsonutils.py\n@@ -1,1 +1,1 @@\n-whatever\n+broken")
        results[f"{short}-garbage"] = a3["final_reward"]
        print(f"[gate] {short}: tests-modified patch -> expect 0.0", flush=True)
        tm_patch = vp.read_text() + (
            "\ndiff --git a/tests/fake_test.py b/tests/fake_test.py\n"
            "new file mode 100644\n--- /dev/null\n+++ b/tests/fake_test.py\n"
            "@@ -0,0 +1,1 @@\n+def test_fake():\n+    assert True\n"
        )
        a4 = eval_patch(short, tm_patch)
        results[f"{short}-tests_modified"] = a4["final_reward"]
    (PHASE0_ROOT / "evals").mkdir(parents=True, exist_ok=True)
    out = PHASE0_ROOT / "gate2.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    ok = all(v == 1.0 for k, v in results.items() if k.endswith("-verified"))
    ok = ok and all(v == 0.0 for k, v in results.items() if not k.endswith("-verified"))
    print(f"[gate] gate-2 {'PASS' if ok else 'FAIL'}: {results}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run agent on one task")
    p_run.add_argument("short")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--protocol", choices=["toolcall", "textbased"], default="textbased")
    p_run.add_argument("--step-limit", type=int, default=40)
    p_run.add_argument("--timeout", type=int, default=120)
    p_run.add_argument("--max-tokens", type=int, default=4096)
    p_run.add_argument("--temperature", type=float, default=0.0)
    p_run.add_argument("--base-url", default="http://127.0.0.1:8012/v1")
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="evaluate a patch")
    p_eval.add_argument("short")
    p_eval.add_argument("--patch", default=None)
    p_eval.add_argument("--from-work", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_base = sub.add_parser("baseline", help="record baseline-failing tests on clean repo")
    p_base.add_argument("short")
    p_base.set_defaults(func=cmd_baseline)

    p_gate = sub.add_parser("gate", help="gate-2 matrix")
    p_gate.set_defaults(func=cmd_gate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
