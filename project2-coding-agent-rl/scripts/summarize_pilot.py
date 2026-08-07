"""Summarize the fixed five-task SWE-agent pilot from persisted trajectories."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "pilot5-deepseek-20260806"
TASKS = {
    "buggy-calculator": PROJECT_ROOT
    / "runs/sweagent-feasibility-deepseek-v4-flash-20260806-retry2/0ed001/0ed001.traj",
    "buggy-slug": None,
    "buggy-inventory": None,
    "buggy-pagination": None,
    "buggy-dedupe": None,
}


def trajectory_for(task: str, configured: Path | None) -> Path:
    if configured is not None:
        return configured
    matches = sorted((RUN_ROOT / task).glob("*/*.traj"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one trajectory for {task}, found {len(matches)}")
    return matches[0]


def main() -> None:
    rows = []
    for task, configured in TASKS.items():
        path = trajectory_for(task, configured)
        data = json.loads(path.read_text(encoding="utf-8"))
        stats = data["info"]["model_stats"]
        rows.append(
            {
                "task": task,
                "exit_status": data["info"]["exit_status"],
                "trajectory_steps": len(data.get("trajectory", [])),
                "api_calls": stats["api_calls"],
                "tokens_sent": stats["tokens_sent"],
                "tokens_received": stats["tokens_received"],
                "instance_cost_usd": stats["instance_cost"],
                "trajectory": str(path.relative_to(PROJECT_ROOT)),
            }
        )

    totals = {
        "tasks": len(rows),
        "submitted": sum(row["exit_status"] == "submitted" for row in rows),
        "api_calls": sum(row["api_calls"] for row in rows),
        "tokens_sent": sum(row["tokens_sent"] for row in rows),
        "tokens_received": sum(row["tokens_received"] for row in rows),
        "instance_cost_usd": sum(row["instance_cost_usd"] for row in rows),
    }
    print(json.dumps({"tasks": rows, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
