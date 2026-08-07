"""Convert τ²-bench 1.x result batches into AgentRx trajectory wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_message(message: dict[str, Any]) -> dict[str, Any]:
    event: dict[str, Any] = {
        "role": message.get("role") or "unknown",
        "content": message.get("content"),
    }
    tool_calls = message.get("tool_calls")
    if tool_calls:
        event["tool_calls"] = tool_calls
    return event


def instruction_from_task(task: dict[str, Any]) -> str:
    scenario = task.get("user_scenario")
    if isinstance(scenario, dict):
        return str(scenario.get("instructions") or scenario)
    return str(scenario or task.get("description") or "")


def convert_batch(batch: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = {str(task.get("id")): task for task in batch.get("tasks", [])}
    converted: list[dict[str, Any]] = []

    for index, simulation in enumerate(batch.get("simulations", []), start=1):
        task_id = str(simulation.get("task_id"))
        task = tasks.get(task_id, {})
        events: list[dict[str, Any]] = []

        policy = simulation.get("policy")
        if policy:
            events.append({"role": "system", "content": str(policy)})
        events.extend(convert_message(message) for message in simulation.get("messages", []))

        converted.append(
            {
                "trajectory_id": str(simulation.get("id") or f"{task_id}-trial-{simulation.get('trial', index)}"),
                "task_id": task_id,
                "instruction": instruction_from_task(task),
                "reward": (simulation.get("reward_info") or {}).get("reward"),
                "events": events,
            }
        )

    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    batch = json.loads(args.input.read_text(encoding="utf-8"))
    converted = convert_batch(batch)
    if not converted:
        raise SystemExit("No simulations found in input batch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(converted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"converted={len(converted)} output={args.output}")


if __name__ == "__main__":
    main()

