"""Build a reviewable diagnosis benchmark manifest from tau2 result batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def failed_actions(reward_info: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for check in reward_info.get("action_checks", []):
        if check.get("action_match") is False:
            action = check.get("action") or {}
            failures.append(
                {
                    "action_id": action.get("action_id"),
                    "name": action.get("name"),
                    "arguments": action.get("arguments"),
                    "tool_type": check.get("tool_type"),
                }
            )
    return failures


def manifest_entry(simulation: dict[str, Any], source: Path) -> dict[str, Any]:
    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    db_check = reward_info.get("db_check") or {}
    outcome = "success" if reward == 1 else "failure"
    return {
        "trajectory_id": str(simulation.get("id")),
        "task_id": str(simulation.get("task_id")),
        "source_results": str(source),
        "outcome": outcome,
        "reward": reward,
        "termination_reason": simulation.get("termination_reason"),
        "duration_seconds": simulation.get("duration"),
        "agent_cost_usd": simulation.get("agent_cost"),
        "user_cost_usd": simulation.get("user_cost"),
        "db_match": db_check.get("db_match"),
        "failed_actions": failed_actions(reward_info),
        "message_count": len(simulation.get("messages", [])),
        "diagnosis_annotation": {
            "review_status": "not_required" if outcome == "success" else "pending_human_review",
            "failure_label": None,
            "critical_step": None,
            "evidence": [],
            "notes": "",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    batch = json.loads(args.input.read_text(encoding="utf-8"))
    entries = [manifest_entry(sim, args.input) for sim in batch.get("simulations", [])]
    summary = {
        "total": len(entries),
        "success": sum(entry["outcome"] == "success" for entry in entries),
        "failure": sum(entry["outcome"] == "failure" for entry in entries),
        "pending_human_review": sum(
            entry["diagnosis_annotation"]["review_status"] == "pending_human_review"
            for entry in entries
        ),
        "agent_cost_usd": sum(float(entry["agent_cost_usd"] or 0) for entry in entries),
        "user_cost_usd": sum(float(entry["user_cost_usd"] or 0) for entry in entries),
    }
    payload = {"schema_version": 1, "summary": summary, "entries": entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
