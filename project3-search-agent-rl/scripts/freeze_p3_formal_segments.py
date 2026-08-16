#!/usr/bin/env python3
"""Freeze the three formal-segment resolved configs (patch 0006 segment stops).

Each segment's override list is obtained from run_p3_grpo_official_exp.sh's own
--dump-overrides path (single source of truth), so this freeze cannot drift from
what the wrapper will actually launch:

  segment   stop   resume source                       run dir (per-run, launch-time)
  0-50      50     none                                <data>/runs/p3-formal-segment-0-50-<stamp>/...
  50-100    100    <0-50 run>/checkpoints/global_step_50
  100-300   300    <50-100 run>/checkpoints/global_step_100

Two fingerprints per segment:
  - full SHA256 over the sorted override list (== the wrapper's config_fp for
    that exact env combination; run dir / resume path are the canonical freeze
    placeholders -- run_managed regenerates the run dir at launch and the
    per-run config_fp is recorded in each run's log)
  - training-invariant SHA256 over the sorted list minus the run-specific keys
    (trainer.segment_stop_step, trainer.resume_mode, trainer.resume_from_path,
     trainer.default_local_dir, trainer.rollout_data_dir, hydra.run.dir,
     trainer.experiment_name)

The three invariant SHAs MUST be identical (same training run on every segment;
only stop/resume/run-dir fields differ) -- asserted here.

Usage:  <env-python> scripts/freeze_p3_formal_segments.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_p3_grpo_official_exp.sh"
DATA = Path("/media/imc/data/project3-search-agent-rl")

# freeze-time canonical stamp; run_managed substitutes the actual run stamp at
# launch (the invariant SHA is unaffected by these fields)
STAMP = "20260816x"

EXCLUDED_KEYS = (
    "trainer.segment_stop_step",
    "trainer.resume_mode",
    "trainer.resume_from_path",
    "trainer.default_local_dir",
    "trainer.rollout_data_dir",
    "hydra.run.dir",
    "trainer.experiment_name",
)

SEGMENTS = [
    {
        "name": "0-50",
        "stop": "50",
        "resume": None,
        "run_dir": str(DATA / f"runs/p3-formal-segment-0-50-fsdp6-b66-n5-s0-{STAMP}"),
    },
    {
        "name": "50-100",
        "stop": "100",
        "resume": str(DATA / f"runs/p3-formal-segment-0-50-fsdp6-b66-n5-s0-{STAMP}/checkpoints/global_step_50"),
        "run_dir": str(DATA / f"runs/p3-formal-segment-50-100-fsdp6-b66-n5-s0-{STAMP}"),
    },
    {
        "name": "100-300",
        "stop": "300",
        "resume": str(DATA / f"runs/p3-formal-segment-50-100-fsdp6-b66-n5-s0-{STAMP}/checkpoints/global_step_100"),
        "run_dir": str(DATA / f"runs/p3-formal-segment-100-300-fsdp6-b66-n5-s0-{STAMP}"),
    },
]


def sha256_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def dump_overrides(segment: dict) -> tuple[list[str], str]:
    env = {
        "PROJECT3_TRAIN_PROFILE": "formal",
        "PROJECT3_SEGMENT_STOP_STEP": segment["stop"],
        "PROJECT3_RUN_DIR": segment["run_dir"],
    }
    if segment["resume"] is not None:
        env["PROJECT3_RESUME_FROM"] = segment["resume"]
    result = subprocess.run(
        ["bash", str(WRAPPER), "--dump-overrides"],
        env={**__import__("os").environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    fp = None
    overrides = []
    for line in lines:
        if line.startswith("__config_fp__="):
            fp = line.split("=", 1)[1]
        elif line.startswith("[OFFICIAL_EXP]"):
            continue
        else:
            overrides.append(line)
    assert fp is not None, f"no config_fp in dump for {segment['name']}"
    return overrides, fp


def main() -> None:
    frozen = {
        "metadata": {
            "created_at": date.today().isoformat(),
            "purpose": "formal three-segment resolved configs (patch 0006)",
            "upstream_commit": "20bd331bdbc9026a5668e11362178e10ab7400c8",
            "patches": ["0001", "0002", "0003", "0004", "0005", "0006"],
            "invariant_definition": "sorted override list minus "
            + ", ".join(EXCLUDED_KEYS),
            "run_dir_stamp_placeholder": STAMP
            + " (run_managed substitutes the launch-time stamp; "
            "per-run config_fp is recorded in each run log; invariant SHA is "
            "unaffected by run dir / resume path / stop fields)",
            "assertion": "invariant SHA identical across all three segments",
        },
        "segments": [],
    }
    invariant_shas = set()
    for segment in SEGMENTS:
        overrides, fp = dump_overrides(segment)
        recomputed = sha256_lines(overrides)
        assert recomputed == fp, (
            f"config_fp mismatch for {segment['name']}: wrapper={fp} recomputed={recomputed}"
        )
        invariant = [o for o in overrides if not any(o.startswith(k + "=") for k in EXCLUDED_KEYS)]
        invariant_sha = sha256_lines(invariant)
        invariant_shas.add(invariant_sha)
        frozen["segments"].append(
            {
                "segment": segment["name"],
                "stop": segment["stop"],
                "resume_from": segment["resume"],
                "run_dir": segment["run_dir"],
                "config_fp_full": fp,
                "training_invariant_sha256": invariant_sha,
                "overrides": overrides,
            }
        )
        print(f"segment {segment['name']}: full={fp} invariant={invariant_sha}")
    assert len(invariant_shas) == 1, f"invariant SHAs differ: {invariant_shas}"
    frozen["training_invariant_sha256_common"] = invariant_shas.pop()
    out_path = ROOT / "configs" / f"p3_formal_segments_{date.today().isoformat()}.json"
    out_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"invariant SHA identical across segments: {frozen['training_invariant_sha256_common']}")
    print(f"frozen to {out_path}")


if __name__ == "__main__":
    main()
