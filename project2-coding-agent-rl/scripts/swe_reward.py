"""Hidden-test reward for the WP6 Agentic RL (GRPO) smoke.

Applies a candidate patch to a scratch copy of the eval repo (Bug Patch
checkout containing the hidden tests) and runs the repo's pytest suite with
its WP2-built eval venv.

Train-time reward is *shaped* (pipeline_evaluate.sh keeps the strict 1/0 for
WP1 evaluations): the base 3B model's patches rarely apply, so a hard 0/1
would hand GRPO a constant-zero advantage and a no-op update. Credit the
intermediate steps monotonically:

    git apply fails                      -> 0.0
    applies (suite fail/crash/timeout)   -> 0.3
    applies, suite clean (0 failed)      -> 0.5
    applies, suite clean, all f2p pass   -> 1.0

Results are memoized per (instance_id, patch sha) so repeated rollouts of
identical patches are free.

Usage (self-test): python scripts/swe_reward.py <instance-id> <patch-file>
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


class SweHiddenTestReward:
    """Rule-based reward: candidate patch -> hidden test suite verdict."""

    def __init__(self, work_root: str | Path, timeout: int = 900, cache: bool = True):
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._cache: dict[tuple, float] = {} if cache else None

    # -- patch handling -----------------------------------------------------

    @staticmethod
    def extract_patch(text: str) -> str:
        """Strip markdown fences and surrounding chatter; keep the diff verbatim."""
        if not text:
            return ""
        m = re.search(r"```(?:diff|patch)?\s*(.*?)```", text, re.S)
        if m:
            text = m.group(1)
        else:
            # raw completions may carry thinking blocks (e.g. a <think> tag from
            # a non-thinking model); they are not part of the patch.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        lines = [ln for ln in text.splitlines() if not ln.startswith("```")]
        out = "\n".join(lines)
        if out and not out.endswith("\n"):
            out += "\n"  # git apply rejects a patch whose final line is incomplete
        return out

    # -- evaluation ---------------------------------------------------------

    def __call__(self, task: dict, action) -> float:
        """task: dict with instance_id/eval_repo/eval_venv/f2p; action: Action."""
        raw = getattr(action, "action", action) if not isinstance(action, str) else action
        patch = self.extract_patch(raw)
        if not patch:
            return 0.0
        key = (task["instance_id"], hashlib.sha256(patch.encode()).hexdigest()[:16])
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        reward = self._run_eval(task, patch)
        if self._cache is not None:
            self._cache[key] = reward
        return reward

    def _run_eval(self, task: dict, patch: str) -> float:
        eval_repo = Path(task["eval_repo"])
        venv = Path(task["eval_venv"])
        f2p: list[str] = task.get("f2p", [])

        short = task["instance_id"].split(".")[-1]
        scratch = self.work_root / f"{short}-{hashlib.sha256(patch.encode()).hexdigest()[:8]}"
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.copytree(eval_repo, scratch, symlinks=True)

        # Apply the candidate patch (same policy as pipeline_evaluate.sh:
        # binary entries stripped, --whitespace=nowarn).
        p = subprocess.run(
            ["git", "-C", str(scratch), "apply", "--whitespace=nowarn", "-"],
            input=patch,
            text=True,
            capture_output=True,
            timeout=self.timeout,
        )
        if p.returncode != 0:
            return 0.0  # unapplicable patch
        reward = 0.3  # applied cleanly

        # Run the repo's own eval venv pytest (WP2 installed per-repo venvs).
        try:
            r = subprocess.run(
                [str(venv / "bin/python"), "-m", "pytest", "-q", "--junitxml=result.xml", "-o", "junit_family=xunit2"],
                cwd=scratch,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return reward

        xml_path = scratch / "result.xml"
        if not xml_path.exists():
            return reward  # collection crash / pytest died

        root = ET.parse(xml_path).getroot()
        failed = errors = 0
        for suite in root.iter("testsuite"):
            failed += int(suite.attrib.get("failures", 0))
            errors += int(suite.attrib.get("errors", 0))

        # FAIL_TO_PASS verdict — dual-format matching like the pipeline.
        f2p_failed = False
        for tc in f2p:
            for case in root.iter("testcase"):
                cn = case.attrib.get("classname", "")
                name = case.attrib.get("name", "")
                keys = {f"{cn}::{name}", f"{cn.replace('.', '/')}.py::{name}"}
                if tc in keys and list(case):
                    f2p_failed = True
                    break
            if f2p_failed:
                break

        if (failed + errors) == 0:
            reward = 0.5  # suite clean
        if (failed + errors) == 0 and not f2p_failed:
            reward = 1.0  # suite clean + all FAIL_TO_PASS pass
        return reward


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    instance_id, patch_file = sys.argv[1], sys.argv[2]

    root = Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20")
    inst_file = next(
        (f for f in (root / "local-instances").glob("*-sanitized.json")
         if json.loads(f.read_text())[0].get("instance_id") == instance_id),
        None,
    )
    if inst_file is None:
        raise SystemExit(f"no sanitized instance found for {instance_id}")
    data = json.loads(inst_file.read_text())[0]
    short = inst_file.name.removesuffix("-sanitized.json")
    registry = json.loads(Path("/home/imc/yzy/agent/project2-coding-agent-rl/scripts/tasks-registry.json").read_text())
    venv_name = registry["tasks"][short]["repo_dir"]
    task = {
        "instance_id": data["instance_id"],
        "eval_repo": str(root / "eval-repos" / short),
        "eval_venv": str(Path("/media/imc/data/yzy/agent/project2/eval-venvs") / venv_name),
        "f2p": data["FAIL_TO_PASS"],
    }
    reward = SweHiddenTestReward("/tmp/swe-reward-selfcheck")(
        task, type("A", (), {"action": Path(patch_file).read_text()})()
    )
    print(f"reward={reward}")


if __name__ == "__main__":
    main()
