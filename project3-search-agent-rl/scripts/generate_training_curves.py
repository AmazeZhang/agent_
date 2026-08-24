#!/usr/bin/env python3
"""Generate dependency-free, local training curves for a managed P3 run.

The verl console logger writes one flattened metric line per optimizer step to
``stdout.log``.  This script extracts those lines, optionally summarizes v2
``*.audit.jsonl`` rollouts, and creates an immutable derived-artifact directory:

  training_curves/metrics.csv
  training_curves/search_behavior.csv       (when audit rollouts exist)
  training_curves/summary.json
  training_curves/training_overview.svg
  training_curves/training_system.svg
  training_curves/search_behavior.svg        (when audit rollouts exist)
  training_curves/index.html

Only the Python standard library is used.  SVG was chosen so curve generation
does not require matplotlib, a network install, a display server, or a GPU.
The output directory is assembled beside the final path and atomically renamed;
an existing output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)(?:\s|$)")
NUMBER_RE = re.compile(
    r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$|^[-+]?(?:nan|inf)$",
    re.IGNORECASE,
)
COMPONENTS = (
    "answer_reward_c",
    "format_reward_c",
    "evidence_hit_reward_c",
    "searched_correct_bonus_c",
    "invalid_penalty_c",
    "redundant_penalty_c",
    "answer_leak_penalty_c",
)
PALETTE = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2")


def _finite_number(text: str) -> float | None:
    if not NUMBER_RE.match(text.strip()):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def parse_console_metrics(path: Path) -> tuple[list[dict[str, float]], dict[str, object]]:
    """Parse verl's ``step:N - key:value`` console lines.

    When a step is logged more than once, the last complete line wins and the
    duplicate step is recorded in the returned diagnostics.
    """

    by_step: dict[int, dict[str, float]] = {}
    duplicates: list[int] = []
    nonfinite: list[dict[str, object]] = []
    if not path.exists():
        return [], {"missing_stdout": True, "duplicate_steps": [], "nonfinite_metrics": []}

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = ANSI_RE.sub("", raw).strip()
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            blob = line[match.start() :]
            row: dict[str, float] = {"step": float(step)}
            for field in blob.split(" - "):
                if ":" not in field:
                    continue
                key, raw_value = field.rsplit(":", 1)
                key = key.strip()
                if key == "step":
                    continue
                value = _finite_number(raw_value.strip())
                if value is None:
                    if NUMBER_RE.match(raw_value.strip()):
                        nonfinite.append({"line": line_no, "step": step, "metric": key, "value": raw_value.strip()})
                    continue
                row[key] = value
            if len(row) == 1:
                continue
            if step in by_step:
                duplicates.append(step)
            by_step[step] = row

    rows = [by_step[step] for step in sorted(by_step)]
    return rows, {
        "missing_stdout": False,
        "duplicate_steps": sorted(set(duplicates)),
        "nonfinite_metrics": nonfinite,
    }


def _step_from_audit_name(path: Path) -> int | None:
    match = re.match(r"^(\d+)\.audit\.jsonl$", path.name)
    return int(match.group(1)) if match else None


def summarize_audit(path: Path) -> dict[str, float]:
    """Summarize one v2 audit rollout without loading token arrays globally."""

    trajectories: dict[str, dict[str, object]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    duplicate_count = padding = search_records = invalid = evidence_hit = evidence_credit = redundant = 0
    record_count = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record_count += 1
            md = record.get("metadata") or {}
            traj_uid = str(md.get("traj_uid", ""))
            env_step = int(md.get("env_step", -1))
            pair = (traj_uid, env_step)
            if pair in seen_pairs:
                duplicate_count += 1
            seen_pairs.add(pair)
            if md.get("is_padding"):
                padding += 1

            traj = trajectories.setdefault(
                traj_uid,
                {
                    "searched": False,
                    "search_rounds": 0,
                    "terminal": False,
                    "max_env_step": -1,
                    "advantages": [],
                    "components": {},
                },
            )
            traj["max_env_step"] = max(int(traj["max_env_step"]), env_step)
            advantage = record.get("trajectory_advantage")
            if isinstance(advantage, (int, float)) and math.isfinite(float(advantage)):
                traj["advantages"].append(float(advantage))

            sv = md.get("search_v1") or {}
            if sv.get("terminal") is True:
                traj["terminal"] = True
            if sv.get("status") is not None:
                traj["searched"] = True
                traj["search_rounds"] = int(traj["search_rounds"]) + 1
                search_records += 1
                invalid += int(bool(sv.get("invalid_or_error")))
                evidence_hit += int(bool(sv.get("evidence_hit")))
                evidence_credit += int(bool(sv.get("evidence_credit")))
                redundant += int(bool(sv.get("redundant_search")))

            episode = md.get("search_v1_episode") or {}
            if episode:
                traj["components"] = {key: float(episode.get(key, 0)) for key in COMPONENTS}

    n_traj = len(trajectories)
    searched = sum(bool(t["searched"]) for t in trajectories.values())
    terminal = sum(bool(t["terminal"]) for t in trajectories.values())
    reached_step4 = sum(int(t["max_env_step"]) >= 3 for t in trajectories.values())
    positive_adv = sum(bool(t["advantages"]) and max(t["advantages"]) > 0 for t in trajectories.values())
    rounds = Counter(int(t["search_rounds"]) for t in trajectories.values())

    row: dict[str, float] = {
        "step": float(_step_from_audit_name(path) or 0),
        "n_records": float(record_count),
        "n_trajectories": float(n_traj),
        "padding_records": float(padding),
        "duplicate_identity_count": float(duplicate_count),
        "search_trajectory_rate": searched / max(1, n_traj),
        "search_records": float(search_records),
        "useful_search_rate": evidence_hit / max(1, search_records),
        "evidence_hit_records": float(evidence_hit),
        "evidence_credit_records": float(evidence_credit),
        "invalid_search_rate": invalid / max(1, search_records),
        "true_redundant_rate": redundant / max(1, search_records),
        "terminal_trajectory_rate": terminal / max(1, n_traj),
        "reached_step4_rate": reached_step4 / max(1, n_traj),
        "positive_advantage_rate": positive_adv / max(1, n_traj),
    }
    for round_count in range(0, 5):
        row[f"search_round_{round_count}_trajectories"] = float(rounds.get(round_count, 0))
    for component in COMPONENTS:
        total = sum(float(t["components"].get(component, 0)) for t in trajectories.values())
        row[f"mean_{component}"] = total / max(1, n_traj)
    return row


def parse_audit_metrics(rollout_dir: Path) -> tuple[list[dict[str, float]], list[dict[str, object]]]:
    rows: list[dict[str, float]] = []
    failures: list[dict[str, object]] = []
    if not rollout_dir.exists():
        return rows, failures
    paths = sorted(
        (p for p in rollout_dir.glob("*.audit.jsonl") if _step_from_audit_name(p) is not None),
        key=lambda p: _step_from_audit_name(p) or 0,
    )
    for path in paths:
        try:
            rows.append(summarize_audit(path))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    return rows, failures


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fields = ["step"] + sorted({key for row in rows for key in row if key != "step"})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if key not in row else f"{row[key]:.12g}") for key in fields})
        handle.flush()
        os.fsync(handle.fileno())


def _series(rows: list[dict[str, float]], key: str) -> list[tuple[float, float]]:
    return [(row["step"], row[key]) for row in rows if key in row and math.isfinite(row[key])]


def _svg_dashboard(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[dict[str, float]],
    panels: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    columns = 2
    panel_w, panel_h = 560, 300
    top = 100
    width = columns * panel_w
    height = top + math.ceil(len(panels) / columns) * panel_h
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;fill:#172033}.title{font-size:24px;font-weight:700}.sub{font-size:13px;fill:#526071}.pt{font-size:15px;font-weight:650}.axis{font-size:11px;fill:#687386}.legend{font-size:11px}.grid{stroke:#e5e9f0;stroke-width:1}.frame{stroke:#cfd6df;fill:#fbfcfe}.line{fill:none;stroke-width:2}.dot{stroke:#fff;stroke-width:1}</style>',
        f'<text x="30" y="38" class="title">{html.escape(title)}</text>',
        f'<text x="30" y="65" class="sub">{html.escape(subtitle)}</text>',
    ]

    for index, (panel_title, specs) in enumerate(panels):
        col, row_index = index % columns, index // columns
        ox, oy = col * panel_w, top + row_index * panel_h
        left, right, panel_top, bottom = ox + 68, ox + panel_w - 24, oy + 42, oy + panel_h - 48
        plot_w, plot_h = right - left, bottom - panel_top
        available = [(key, label, _series(rows, key)) for key, label in specs]
        available = [(key, label, values) for key, label, values in available if values]
        out.append(f'<rect x="{ox + 12}" y="{oy + 8}" width="{panel_w - 24}" height="{panel_h - 18}" rx="8" class="frame"/>')
        out.append(f'<text x="{ox + 28}" y="{oy + 31}" class="pt">{html.escape(panel_title)}</text>')
        if not available:
            out.append(f'<text x="{left}" y="{panel_top + 30}" class="sub">No metric in this run</text>')
            continue
        all_points = [point for _, _, values in available for point in values]
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_min == x_max:
            x_min, x_max = x_min - 0.5, x_max + 0.5
        if y_min == y_max:
            pad = max(abs(y_min) * 0.1, 1.0)
            y_min, y_max = y_min - pad, y_max + pad
        else:
            pad = (y_max - y_min) * 0.08
            y_min, y_max = y_min - pad, y_max + pad

        def sx(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * plot_w

        def sy(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * plot_h

        for tick in range(5):
            frac = tick / 4
            y = panel_top + frac * plot_h
            value = y_max - frac * (y_max - y_min)
            out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid"/>')
            out.append(f'<text x="{left - 7}" y="{y + 4:.2f}" text-anchor="end" class="axis">{value:.3g}</text>')
        for tick in range(5):
            frac = tick / 4
            x = left + frac * plot_w
            value = x_min + frac * (x_max - x_min)
            out.append(f'<text x="{x:.2f}" y="{bottom + 20}" text-anchor="middle" class="axis">{value:.3g}</text>')
        out.append(f'<text x="{(left + right) / 2:.2f}" y="{bottom + 38}" text-anchor="middle" class="axis">optimizer step</text>')

        legend_x = left
        for series_index, (_, label, values) in enumerate(available):
            color = PALETTE[series_index % len(PALETTE)]
            points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
            out.append(f'<polyline points="{points}" class="line" stroke="{color}"/>')
            for x, y in values:
                out.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="{color}" class="dot"/>')
            out.append(f'<line x1="{legend_x}" y1="{oy + panel_h - 19}" x2="{legend_x + 18}" y2="{oy + panel_h - 19}" stroke="{color}" stroke-width="2"/>')
            out.append(f'<text x="{legend_x + 23}" y="{oy + panel_h - 15}" class="legend">{html.escape(label)}</text>')
            legend_x += 34 + max(70, len(label) * 6)
    out.append("</svg>")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def _write_index(path: Path, run_id: str, files: list[str], summary: dict[str, object]) -> None:
    links = "\n".join(f'<li><a href="{html.escape(name)}">{html.escape(name)}</a></li>' for name in files)
    notes = [
        "Curves are derived from the immutable managed-run console and audit logs.",
        "The verl max_memory_reserved_gb metric is an allocator/worker view, not a physical per-GPU peak.",
        "reached_step4_rate means a trajectory touched env step 4; it does not by itself distinguish an answer at the cap from forced exhaustion.",
        "Missing panels mean that metric was not emitted by that run; values are never fabricated or interpolated.",
    ]
    note_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    content = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>{html.escape(run_id)} training curves</title>
<style>body{{max-width:1160px;margin:32px auto;font:15px/1.5 system-ui;color:#172033}}img{{width:100%;border:1px solid #d8dee8;margin:14px 0}}code{{background:#f3f5f8;padding:2px 5px}}a{{color:#1559c5}}</style>
<h1>{html.escape(run_id)} training curves</h1>
<p>Optimizer steps: <strong>{summary['n_metric_steps']}</strong>; audit steps: <strong>{summary['n_audit_steps']}</strong>.</p>
<ul>{links}</ul>
<h2>Training overview</h2><img src="training_overview.svg" alt="training overview">
<h2>System performance</h2><img src="training_system.svg" alt="training system">
{"<h2>Search behavior</h2><img src='search_behavior.svg' alt='search behavior'>" if 'search_behavior.svg' in files else ''}
<h2>Interpretation notes</h2><ul>{note_html}</ul>
</html>\n"""
    path.write_text(content, encoding="utf-8")


def generate(run_dir: Path, output_name: str = "training_curves") -> Path | None:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    metric_rows, parse_diagnostics = parse_console_metrics(run_dir / "stdout.log")
    if not metric_rows:
        print(f"[CURVES] skipped: no verl optimizer-step metrics in {run_dir / 'stdout.log'}")
        return None

    output_dir = run_dir / output_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing curve directory: {output_dir}")
    partial = run_dir / f".{output_name}.partial-{os.getpid()}"
    partial.mkdir(mode=0o755)
    try:
        audit_rows, audit_failures = parse_audit_metrics(run_dir / "rollouts")
        _write_csv(partial / "metrics.csv", metric_rows)
        if audit_rows:
            _write_csv(partial / "search_behavior.csv", audit_rows)

        run_id = _metadata(run_dir / "metadata.env").get("run_id", run_dir.name)
        subtitle = f"run={run_id}; source=stdout.log; points={len(metric_rows)}; no interpolation"
        overview_panels = [
            ("Outcome and reward", [("critic/score/mean", "score mean"), ("episode/reward/mean", "episode reward"), ("episode/success_rate", "success rate")]),
            ("Policy objective", [("actor/pg_loss", "policy loss"), ("actor/kl_loss", "KL loss"), ("actor/ppo_kl", "PPO KL")]),
            ("Exploration", [("actor/entropy_loss", "entropy"), ("actor/pg_clipfrac", "clip fraction")]),
            ("Gradient stability", [("actor/grad_norm", "gradient norm"), ("training/rollout_probs_diff_mean", "rollout prob diff")]),
            ("Agent interaction", [("episode/tool_call_count/mean", "tool calls"), ("episode/length/mean", "episode length")]),
            ("Sequence lengths", [("response_length/mean", "response length"), ("prompt_length/mean", "prompt length")]),
        ]
        _svg_dashboard(partial / "training_overview.svg", "Training overview", subtitle, metric_rows, overview_panels)
        system_panels = [
            ("Step wall time", [("timing_s/step", "step")]),
            ("Phase time", [("timing_s/gen", "rollout"), ("timing_s/old_log_prob", "old log-prob"), ("timing_s/ref", "reference"), ("timing_s/update_actor", "actor update")]),
            ("Throughput", [("perf/throughput", "tokens/s")]),
            ("Memory views (not additive)", [("perf/max_memory_allocated_gb", "torch allocated"), ("perf/max_memory_reserved_gb", "torch reserved"), ("perf/cpu_memory_used_gb", "CPU used")]),
            ("Response clipping", [("response_length/clip_ratio", "response clipped"), ("prompt_length/clip_ratio", "prompt clipped")]),
            ("Batch balance", [("global_seqlen/minmax_diff", "raw diff"), ("global_seqlen/balanced_max", "balanced max")]),
        ]
        _svg_dashboard(partial / "training_system.svg", "Training system performance", subtitle, metric_rows, system_panels)

        files = ["metrics.csv", "summary.json", "training_overview.svg", "training_system.svg", "index.html"]
        if audit_rows:
            audit_subtitle = f"run={run_id}; source=rollouts/*.audit.jsonl; points={len(audit_rows)}; exact per-step aggregation"
            search_panels = [
                ("Search adoption", [("search_trajectory_rate", "searched trajectories"), ("terminal_trajectory_rate", "terminal records")]),
                ("Evidence quality", [("useful_search_rate", "evidence hit/search"), ("positive_advantage_rate", "positive advantage")]),
                ("Search failure modes", [("invalid_search_rate", "invalid/error"), ("true_redundant_rate", "true redundant")]),
                ("Turn-cap pressure", [("reached_step4_rate", "reached step 4")]),
                ("Search rounds", [("search_round_1_trajectories", "1 search"), ("search_round_2_trajectories", "2 searches"), ("search_round_3_trajectories", "3 searches")]),
                ("Reward components (cents/traj)", [("mean_answer_reward_c", "answer"), ("mean_evidence_hit_reward_c", "evidence"), ("mean_searched_correct_bonus_c", "searched+correct"), ("mean_redundant_penalty_c", "redundant")]),
            ]
            _svg_dashboard(partial / "search_behavior.svg", "Search behavior", audit_subtitle, audit_rows, search_panels)
            files.extend(["search_behavior.csv", "search_behavior.svg"])

        summary: dict[str, object] = {
            "schema_version": 1,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "n_metric_steps": len(metric_rows),
            "metric_steps": [int(row["step"]) for row in metric_rows],
            "n_audit_steps": len(audit_rows),
            "audit_steps": [int(row["step"]) for row in audit_rows],
            "console_parse": parse_diagnostics,
            "audit_failures": audit_failures,
            "stdout_sha256": _sha256(run_dir / "stdout.log"),
            "generation": {
                "dependency": "Python standard library only",
                "gpu_required": False,
                "interpolation": False,
                "overwrite": False,
            },
        }
        (partial / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _write_index(partial / "index.html", run_id, sorted(files), summary)
        for artifact in partial.iterdir():
            if artifact.is_file():
                with artifact.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.rename(partial, output_dir)
        print(f"[CURVES] generated: {output_dir}")
        return output_dir
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="training_curves")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.output_name):
        parser.error("--output-name must be a simple 1-64 character directory name")
    try:
        generate(args.run_dir, args.output_name)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[CURVES] error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
