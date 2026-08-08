"""Build a small SFT parquet dataset from trusted rollout trajectories.

Each trajectory step becomes one row: the conversation messages up to that step
(system prompt + task + previous assistant actions) and the assistant message =
the step's action (with thought when present). Uses only trusted runs from
training-data/ (successes by default).

NOTE (2026-08-08, deviation documented in reports/): the first SFT smoke run
predates the HOLDOUT filter below and trained on oauthlib-signature-1bsv3m8l
— a frozen WP7 holdout. The filter is added here so re-running never repeats
that contamination.

Usage: python scripts/build_smoke_sft_data.py [--source-dir DIR] [--out PATH] [--max-steps N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SOURCE = Path("/media/imc/data/yzy/agent/project2/training-data/successes")

# WP7 frozen holdouts (spec T7.1) — never train on these.
HOLDOUT = {
    "funcy-curry-compose-3u9hti2d",
    "pygments-groff-0jqqr58z",
    "stackprinter-1i9gep13",
    "oauthlib-signature-1bsv3m8l",
    "boltons-7nlifqzn",
}


def traj_steps_to_rows(traj_path: Path, max_steps: int) -> list[dict]:
    """One row per trajectory step: prior messages + assistant action."""
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    traj = data.get("trajectory", [])
    rows = []
    for step in traj[:max_steps]:
        query = step.get("query") or []
        messages = []
        for m in query:
            if m.get("role") in ("system", "user", "assistant") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
        action = (step.get("thought") or "") + "\n" + (step.get("action") or "")
        action = action.strip()
        if not action:
            continue
        messages.append({"role": "assistant", "content": action})
        rows.append({"messages": messages})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path,
                        default=Path("/media/imc/data/yzy/agent/project2/smoke-data/sft-train.parquet"))
    parser.add_argument("--max-steps", type=int, default=20,
                        help="max steps per trajectory (head of trajectory only)")
    parser.add_argument("--max-tasks", type=int, default=3, help="max tasks to use")
    args = parser.parse_args()

    rows: list[dict] = []
    used: list[str] = []
    for run_dir in sorted(args.source_dir.iterdir()):
        if not run_dir.is_dir() or len(used) >= args.max_tasks:
            continue
        if any(short in run_dir.name for short in HOLDOUT):
            continue  # WP7 holdout — never enters SFT
        traj = next(run_dir.glob("*.traj"), None)
        if traj is None:
            continue
        step_rows = traj_steps_to_rows(traj, args.max_steps)
        rows.extend(step_rows)
        used.append(run_dir.name)
        print(f"{run_dir.name}: {len(step_rows)} rows")

    if not rows:
        raise SystemExit("no rows produced — nothing to write")

    table = pa.table({"messages": [json.dumps(r["messages"]) for r in rows]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote {args.out} ({len(rows)} rows) from {used}")


if __name__ == "__main__":
    main()
