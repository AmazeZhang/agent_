"""Rebuild evaluations/summary.json from raw rollout and eval artifacts.

Deterministic reconstruction: every number in the summary traces back to files
under PILOT_ROOT/runs/ and PILOT_ROOT/candidate-evals/. Historical invalid runs
(run4-6, parent-history leakage) are treated as immutable facts carried in
INVALID_RUNS; everything else is recomputed from artifacts.

Usage: python scripts/pipeline_summary.py [--pilot-root PATH] [--write]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

INVALID_RUNS = {
    "deepseek-v4-flash-run4": "parent history accessible in local upload",
    "deepseek-v4-flash-run5": "model read Bug Patch / ground-truth implementation",
    "deepseek-v4-flash-run6": "answer accessible via parent history",
}

RUN_RE = re.compile(r"^deepseek-v4-flash-run\d+.*$")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path,
                        default=Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20"))
    parser.add_argument("--write", action="store_true", help="write summary.json (default: print only)")
    args = parser.parse_args()

    root = args.pilot_root
    runs_dir = root / "runs"
    evals_dir = root / "candidate-evals"
    evaluations_dir = root / "evaluations"

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and RUN_RE.match(d.name))
    full_test_results: dict[str, str] = {}
    task_quality_flags: dict[str, str] = {}
    per_run: dict[str, dict] = {}
    rewards: list[int] = []
    trusted = 0
    executed = 0

    for run_dir in run_dirs:
        run_id = run_dir.name
        executed += 1
        # instance subdirs hold patch/traj; candidate-evals hold eval_result.json
        instance_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        entry: dict = {"run_id": run_id}

        if run_id in INVALID_RUNS:
            entry["status"] = "invalid"
            entry["reason"] = INVALID_RUNS[run_id]
            for idir in instance_dirs:
                full_test_results.setdefault(idir.name, f"invalid: {INVALID_RUNS[run_id]}")
            per_run[run_id] = entry
            continue

        entry["status"] = "trusted"
        for idir in instance_dirs:
            instance_id = idir.name
            patch = idir / f"{instance_id}.patch"
            traj = idir / f"{instance_id}.traj"
            entry["instance_id"] = instance_id
            entry["has_patch"] = patch.exists()
            entry["has_traj"] = traj.exists()
            if patch.exists():
                entry["patch"] = str(patch.relative_to(root))
            if traj.exists():
                entry["traj"] = str(traj.relative_to(root))

            # integrity: prefer integrity.json, then eval_result.json's field
            integrity = idir / "integrity.json"
            if integrity.exists():
                iverdict = load_json(integrity)
                entry["integrity"] = iverdict.get("verdict", "unknown")
                if iverdict.get("verdict") != "ok":
                    entry["status"] = "invalid"
                    entry["reason"] = iverdict
            else:
                entry["integrity"] = "no integrity.json (pre-pipeline run)"

            # eval result: candidate-evals/<task>-<run_id>/eval_result.json
            # legacy naming: candidate-evals/<task>-run<N>[-suffix]
            run_number = re.search(r"run(\d+)", run_id).group(1) if re.search(r"run(\d+)", run_id) else None
            candidate_matches = list(evals_dir.glob(f"*-{run_id}"))
            if not candidate_matches and run_number:
                candidate_matches = list(evals_dir.glob(f"*-run{run_number}*"))
            if candidate_matches:
                er = candidate_matches[0] / "eval_result.json"
                if er.exists():
                    erd = load_json(er)
                    full_test_results[instance_id] = erd.get("full_suite", "unknown")
                    ftp = erd.get("fail_to_pass", {})
                    entry["fail_to_pass"] = ftp
                    reward = erd.get("reward")
                    entry["reward"] = reward
                    if entry["status"] == "trusted":
                        trusted += 1
                        rewards.append(reward if reward is not None else 0)
                    else:
                        rewards.append(-1)
                else:
                    entry["status"] = "pending-eval"
            else:
                entry["status"] = "pending-eval"
            per_run[run_id] = entry

    trusted_attempted = len(rewards) - rewards.count(-1)
    reward_1 = rewards.count(1) if trusted_attempted else 0
    reward_0 = sum(1 for r in rewards if r == 0)
    rate = round(reward_1 / trusted_attempted, 3) if trusted_attempted else None

    summary = {
        "schema_version": 2,
        "selected_tasks": 20,
        "executed_runs": executed,
        "trusted_attempted": trusted_attempted,
        "invalidated_runs": len(INVALID_RUNS),
        "pending_eval": sum(1 for e in per_run.values() if e.get("status") == "pending-eval"),
        "reward_1": reward_1,
        "reward_0": reward_0,
        "strict_success_rate_so_far": rate,
        "full_test_results": full_test_results,
        "task_quality_flags": task_quality_flags,
        "integrity_correction": "Runs 4-6 used local clones with accessible parent history. "
            "They are excluded from metrics and training data. All local uploads must pass "
            "the single-commit/parent-inaccessible guard.",
        "per_run": per_run,
        "scope_warning": "Sample remains too small for a stable performance estimate.",
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.write:
        evaluations_dir.mkdir(parents=True, exist_ok=True)
        (evaluations_dir / "summary.json").write_text(text + "\n", encoding="utf-8")
        print(f"wrote {evaluations_dir / 'summary.json'}")
    else:
        print(text)


if __name__ == "__main__":
    main()
