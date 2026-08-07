"""Validate independently tested SWE-agent trajectories and export JSONL."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def changed_paths(patch: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE)


def is_test_path(path: str) -> bool:
    name = Path(path).name.lower()
    parts = {part.lower() for part in Path(path).parts}
    return (
        "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".spec.js")
        or name.endswith(".test.js")
    )


def inspect_container(name: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "inspect", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)[0]


def validate_task(task: dict[str, Any], verify_containers: bool) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    paths = {
        key: PROJECT_ROOT / task[key]
        for key in ("fixture", "issue", "trajectory", "patch")
    }
    for key, path in paths.items():
        if not path.exists():
            errors.append(f"missing {key}: {path}")
    if errors:
        return errors, {}

    trajectory = json.loads(paths["trajectory"].read_text(encoding="utf-8"))
    patch = paths["patch"].read_text(encoding="utf-8")
    issue = paths["issue"].read_text(encoding="utf-8").strip()
    info = trajectory.get("info", {})
    files = changed_paths(patch)

    if info.get("exit_status") != "submitted":
        errors.append(f"exit_status is {info.get('exit_status')!r}, expected 'submitted'")
    if not patch.strip() or not files:
        errors.append("patch is empty or has no modified b/ paths")
    if info.get("submission", "").strip() != patch.strip():
        errors.append("persisted patch differs from trajectory submission")
    modified_tests = [path for path in files if is_test_path(path)]
    if modified_tests:
        errors.append(f"patch modifies test files: {modified_tests}")
    if not trajectory.get("history") or not trajectory.get("trajectory"):
        errors.append("trajectory history or action trace is empty")

    verification: dict[str, Any] = {
        "method": "independent_cpu_only_offline_docker",
        "container": task["verification_container"],
        "expected_tests": task["expected_tests"],
        "checked_live": verify_containers,
    }
    if verify_containers:
        try:
            container = inspect_container(task["verification_container"])
        except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as exc:
            errors.append(f"cannot inspect verification container: {exc}")
        else:
            state = container.get("State", {})
            host = container.get("HostConfig", {})
            exit_code = state.get("ExitCode")
            network_mode = host.get("NetworkMode")
            verification.update(
                {
                    "status": state.get("Status"),
                    "exit_code": exit_code,
                    "network_mode": network_mode,
                    "command": container.get("Config", {}).get("Cmd"),
                }
            )
            if state.get("Status") != "exited" or exit_code != 0:
                errors.append(f"verification container did not exit cleanly: {state}")
            if network_mode != "none":
                errors.append(f"verification was not offline: network_mode={network_mode!r}")

    record = {
        "id": task["task_id"],
        "problem_statement": issue,
        "messages": trajectory.get("history", []),
        "trajectory": trajectory.get("trajectory", []),
        "patch": patch,
        "reward": 1 if not errors else 0,
        "metadata": {
            "source": "controlled_pilot",
            "fixture": task["fixture"],
            "trajectory_path": task["trajectory"],
            "patch_path": task["patch"],
            "changed_paths": files,
            "model_stats": info.get("model_stats", {}),
            "verification": verification,
        },
    }
    return errors, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pilot5-rollouts.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-containers", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for task in config["tasks"]:
        errors, record = validate_task(task, args.verify_containers)
        if errors:
            rejected.append({"task_id": task["task_id"], "errors": errors})
        else:
            accepted.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "verified_rollouts.jsonl"
    output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "summary": {
            "input": len(config["tasks"]),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "container_verification_required": args.verify_containers,
            "reward_sum": sum(row["reward"] for row in accepted),
            "expected_tests": sum(
                row["metadata"]["verification"]["expected_tests"] for row in accepted
            ),
        },
        "accepted_ids": [row["id"] for row in accepted],
        "rejected": rejected,
        "output": str(output_jsonl),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
