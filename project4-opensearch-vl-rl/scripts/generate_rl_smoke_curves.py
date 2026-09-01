#!/usr/bin/env python3
"""Generate auditable Project 4 RL smoke-training artifacts from managed Runs.

The generator is CPU-only. It keeps replay and fresh-online optimizer updates
separate, and preserves rollout-only failure steps without inventing loss values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def replay_metric_row(replay_dir: Path) -> dict[str, Any]:
    state = load_json(replay_dir / "output" / "trainer_state.json")
    return {
        "step": int(state["global_step"]),
        "stage": "replay",
        "optimizer_update": 1,
        "optimizer_gate_passed": 1,
        "weighted_loss": float(state["weighted_loss"]),
        "grad_norm": float(state["grad_norm"]),
        "learning_rate": float(state["learning_rate"]),
        "rollout_count": int(state.get("active_trajectory_count", 0)),
        "fatal_fraction": "",
        "full_success_fraction": "",
        "reward_mean": "",
        "reward_variance": "",
        "tool_calls_mean": "",
        "note": "replayed trajectories; not a fresh-online point",
    }


def _rollout_event(record: dict[str, Any]) -> dict[str, Any] | None:
    required = {"global_step", "rollout", "reward", "tools"}
    if not required.issubset(record):
        return None
    tools = record["tools"]
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise ValueError("rollout tools must be a list of strings")
    return {
        "global_step": int(record["global_step"]),
        "rollout": int(record["rollout"]),
        "reward": float(record["reward"]),
        "full_success": bool(record.get("full_success", False)),
        "fatal": record.get("fatal"),
        "tools": tools,
        "tool_call_count": len(tools),
    }


def online_metric_rows(online_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = parse_json_lines(online_dir / "stdout.log")
    events = [event for record in raw if (event := _rollout_event(record)) is not None]
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_step[event["global_step"]].append(event)

    states: dict[int, dict[str, Any]] = {}
    output_dir = online_dir / "output"
    for path in sorted(output_dir.glob("checkpoint-*/trainer_state.json")):
        state = load_json(path)
        states[int(state["global_step"])] = state

    rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        step_events = by_step[step]
        rewards = [event["reward"] for event in step_events]
        state = states.get(step)
        gate = (state or {}).get("gate") or {}
        summary = (state or {}).get("group_summary") or {}
        fatal_count = sum(event["fatal"] is not None for event in step_events)
        full_success_count = sum(event["full_success"] for event in step_events)
        row: dict[str, Any] = {
            "step": step,
            "stage": "fresh-online",
            "optimizer_update": int(state is not None),
            "optimizer_gate_passed": int(bool(gate.get("passed"))) if state else 0,
            "weighted_loss": float(state["weighted_loss"]) if state else "",
            "grad_norm": float(state["grad_norm"]) if state else "",
            "learning_rate": float(state["learning_rate"]) if state else "",
            "rollout_count": len(step_events),
            "fatal_fraction": fatal_count / len(step_events),
            "full_success_fraction": full_success_count / len(step_events),
            "reward_mean": float(summary.get("reward_mean", fmean(rewards))),
            "reward_variance": float(summary.get("reward_population_variance", pvariance(rewards))),
            "tool_calls_mean": fmean(event["tool_call_count"] for event in step_events),
            "note": "optimizer update completed" if state else "rollouts recorded; optimizer gate failed",
        }
        rows.append(row)
    return rows, events


FIELDS = [
    "step",
    "stage",
    "optimizer_update",
    "optimizer_gate_passed",
    "weighted_loss",
    "grad_norm",
    "learning_rate",
    "rollout_count",
    "fatal_fraction",
    "full_success_fraction",
    "reward_mean",
    "reward_variance",
    "tool_calls_mean",
    "note",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_events(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]


def _points(rows: list[dict[str, Any]], key: str, stage: str | None = None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if stage is not None and row["stage"] != stage:
            continue
        value = row.get(key, "")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            points.append((float(row["step"]), float(value)))
    return points


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1200, 820
    panels = [
        ("Weighted objective (stage-separated)", [("weighted_loss", "fresh-online", "online"), ("weighted_loss", "replay", "replay")]),
        ("Gradient norm", [("grad_norm", "fresh-online", "online"), ("grad_norm", "replay", "replay")]),
        ("Fresh-online rollout reward", [("reward_mean", "fresh-online", "mean reward")]),
        ("Rollout behavior", [("tool_calls_mean", "fresh-online", "mean tool calls"), ("fatal_fraction", "fresh-online", "fatal fraction")]),
    ]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.sub{font-size:13px;fill:#526071}.pt{font-size:16px;font-weight:650}.axis{font-size:11px;fill:#687386}.legend{font-size:11px}.frame{stroke:#d7dee8;fill:#fff}.grid{stroke:#e7ebf0}.line{fill:none;stroke-width:2.5}.dot{stroke:#fff;stroke-width:1.5}</style>',
        '<text x="34" y="40" class="title">Project 4 RL smoke-training status</text>',
        '<text x="34" y="66" class="sub">Observed logs only · replay step 1 is not connected to fresh-online steps · step 4 has rollouts but no optimizer update</text>',
    ]

    for index, (title, series_specs) in enumerate(panels):
        col, row_index = index % 2, index // 2
        ox, oy = 20 + col * 590, 95 + row_index * 345
        panel_w, panel_h = 570, 325
        left, right, top, bottom = ox + 70, ox + 548, oy + 50, oy + 270
        available = []
        for key, stage, label in series_specs:
            values = _points(rows, key, stage)
            if values:
                available.append((label, values))
        all_points = [point for _, values in available for point in values]
        out.append(f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{panel_h}" rx="10" class="frame"/>')
        out.append(f'<text x="{ox + 20}" y="{oy + 31}" class="pt">{html.escape(title)}</text>')
        xs = [1.0, 4.0]
        ys = [point[1] for point in all_points]
        y_min, y_max = min(ys), max(ys)
        if y_min == y_max:
            pad = max(abs(y_min) * 0.15, 0.01)
        else:
            pad = (y_max - y_min) * 0.15
        y_min, y_max = y_min - pad, y_max + pad
        if y_min < 0 < y_max:
            pass
        elif y_min >= 0:
            y_min = 0.0

        def sx(x: float) -> float:
            return left + (x - xs[0]) / (xs[1] - xs[0]) * (right - left)

        def sy(y: float) -> float:
            return bottom - (y - y_min) / (y_max - y_min) * (bottom - top)

        for tick in range(5):
            y = top + tick * (bottom - top) / 4
            value = y_max - tick * (y_max - y_min) / 4
            out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
            out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis">{html.escape(_fmt(value))}</text>')
        for step in range(1, 5):
            x = sx(float(step))
            out.append(f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" class="axis">{step}</text>')
        out.append(f'<text x="{(left + right) / 2:.1f}" y="{bottom + 42}" text-anchor="middle" class="axis">global step</text>')

        for series_index, (label, values) in enumerate(available):
            color = COLORS[series_index]
            coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
            if len(values) > 1:
                out.append(f'<polyline points="{coords}" class="line" stroke="{color}"/>')
            for x, y in values:
                out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.5" fill="{color}" class="dot"/>')
                out.append(f'<text x="{sx(x):.1f}" y="{sy(y) - 9:.1f}" text-anchor="middle" class="axis">{html.escape(_fmt(y))}</text>')
            legend_x = ox + 300 + series_index * 120
            out.append(f'<line x1="{legend_x}" y1="{oy + 29}" x2="{legend_x + 18}" y2="{oy + 29}" stroke="{color}" stroke-width="3"/>')
            out.append(f'<text x="{legend_x + 24}" y="{oy + 33}" class="legend">{html.escape(label)}</text>')

        if index in (0, 1):
            boundary = (sx(1) + sx(2)) / 2
            out.append(f'<line x1="{boundary:.1f}" y1="{top}" x2="{boundary:.1f}" y2="{bottom}" stroke="#94a3b8" stroke-dasharray="5 5"/>')
        if index in (2, 3):
            failure_x = sx(4)
            out.append(f'<line x1="{failure_x:.1f}" y1="{top}" x2="{failure_x:.1f}" y2="{bottom}" stroke="#dc2626" stroke-dasharray="5 4"/>')
            out.append(f'<text x="{failure_x - 5:.1f}" y="{top + 14}" text-anchor="end" class="axis" fill="#dc2626">gate failed</text>')

    out.extend([
        '<rect x="20" y="790" width="1160" height="1" fill="#d7dee8"/>',
        '<text x="30" y="812" class="sub">Scope: pipeline smoke evidence, not convergence or policy-improvement evidence. KL and entropy were not logged and are intentionally omitted.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def generate(replay_dir: Path, online_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_state = replay_dir / "output" / "trainer_state.json"
    replay_stdout = replay_dir / "stdout.log"
    online_stdout = online_dir / "stdout.log"
    online_states = sorted((online_dir / "output").glob("checkpoint-*/trainer_state.json"))
    sources = [
        ("replay/trainer_state.json", replay_state),
        ("replay/stdout.log", replay_stdout),
        ("fresh-online/stdout.log", online_stdout),
        *[
            (f"fresh-online/{path.parent.name}/trainer_state.json", path)
            for path in online_states
        ],
    ]
    for _, path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    online_rows, events = online_metric_rows(online_dir)
    rows = [replay_metric_row(replay_dir), *online_rows]
    write_csv(output_dir / "p4_rl_smoke_metrics.csv", rows)
    write_events(output_dir / "p4_rl_smoke_events.jsonl", events)
    write_svg(output_dir / "p4_rl_smoke_training_status.svg", rows)
    manifest = {
        "schema_version": 1,
        "claim_scope": "pipeline smoke evidence; not convergence or policy improvement",
        "sources": [
            {"label": label, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for label, path in sources
        ],
        "rows": len(rows),
        "normalized_rollout_events": len(events),
        "omitted_metrics": ["kl", "entropy"],
    }
    (output_dir / "p4_rl_smoke_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--online-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = generate(args.replay_run.resolve(), args.online_run.resolve(), args.output_dir.resolve())
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "metric_rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
