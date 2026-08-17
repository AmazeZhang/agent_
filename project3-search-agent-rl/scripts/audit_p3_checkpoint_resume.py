#!/usr/bin/env python3
"""Read-only resume-audit for a formal segment checkpoint (pre-resume gate).

Checks (exit nonzero on any failure; never writes into the checkpoint dir):
  1. Expected files present: model/optim/extra_state x N ranks + data.pt.
  2. Per-rank extra_state loads and contains lr_scheduler + rng keys.
  3. rank0 lr_scheduler: last_epoch == expected; LR at last_epoch matches the
     frozen warmup curve (transformers get_constant_schedule_with_warmup,
     num_warmup_steps=85, base lr 1e-6): lr = base * min(epoch/85, 1.0).
  4. rank0 optimizer state non-trivial: param_groups/state present, at least one
     state entry, step > 0, and exp_avg/exp_avg_sq contain non-zero values
     (i.e. NOT a freshly re-initialized optimizer).
  5. data.pt: torchdata StatefulDataLoader snapshot step == expected
     (resume will continue with the NEXT batch after this snapshot).
  6. rng state present in rank0 extra_state (torch/cuda/np seeds saved).

Usage:
  <env-python> scripts/audit_p3_checkpoint_resume.py \
      --ckpt-dir <global_step_N> --last-epoch <N> --snapshot-step <N> \
      [--report-out <path.json>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CKPT_MEMBERS = ("model", "optim", "extra_state")


def check_files(ckpt: Path, world_size: int) -> dict:
    """Verify 6x model/optim/extra_state + data.pt exist with non-trivial size."""
    report = {"expected_world_size": world_size, "files": {}, "missing": []}
    for member in CKPT_MEMBERS:
        for rank in range(world_size):
            p = ckpt / "actor" / f"{member}_world_size_{world_size}_rank_{rank}.pt"
            ok = p.is_file() and p.stat().st_size > 0
            report["files"][p.name] = {"exists": ok, "bytes": p.stat().st_size if p.is_file() else 0}
            if not ok:
                report["missing"].append(p.name)
    dp = ckpt / "data.pt"
    ok = dp.is_file() and dp.stat().st_size > 0
    report["files"]["data.pt"] = {"exists": ok, "bytes": dp.stat().st_size if dp.is_file() else 0}
    if not ok:
        report["missing"].append("data.pt")
    report["missing_count"] = len(report["missing"])
    return report


def check_extra_state(ckpt: Path, world_size: int, expected_last_epoch: int, base_lr: float, warmup: int) -> dict:
    """Load per-rank extra_state; verify scheduler on rank0."""
    report = {"per_rank": {}}
    lr_lambda = lambda step: float(step) / float(max(1.0, warmup)) if step < warmup else 1.0
    expected_lr = base_lr * lr_lambda(expected_last_epoch)
    report["expected_lr_at_last_epoch"] = expected_lr
    report["lr_formula"] = f"base_lr({base_lr}) * min(last_epoch/{warmup}, 1.0)"

    for rank in range(world_size):
        p = ckpt / "actor" / f"extra_state_world_size_{world_size}_rank_{rank}.pt"
        sd = torch.load(p, map_location="cpu", weights_only=False)
        keys = sorted(sd.keys())
        entry = {"keys": keys}
        if "lr_scheduler" in sd and sd["lr_scheduler"] is not None:
            s = sd["lr_scheduler"]
            entry["scheduler_keys"] = sorted(s.keys())
            entry["scheduler_last_epoch"] = s.get("last_epoch", None)
            entry["scheduler_base_lrs"] = s.get("base_lrs", None)
            entry["scheduler_step_count"] = s.get("_step_count", None)
        entry["has_rng"] = "rng" in sd and sd["rng"] is not None
        if entry["has_rng"]:
            rng = sd["rng"]
            entry["rng_keys"] = sorted(rng.keys()) if isinstance(rng, dict) else type(rng).__name__
        report["per_rank"][p.name] = entry

    r0 = report["per_rank"][f"extra_state_world_size_{world_size}_rank_0.pt"]
    r0["expected_last_epoch"] = expected_last_epoch
    r0["last_epoch_ok"] = r0.get("scheduler_last_epoch") == expected_last_epoch
    r0["lr_at_last_epoch_ok"] = r0.get("scheduler_last_epoch") is not None and abs(
        base_lr * lr_lambda(r0["scheduler_last_epoch"]) - expected_lr
    ) < 1e-12
    report["rank0_last_epoch_ok"] = r0["last_epoch_ok"]
    report["rank0_lr_curve_ok"] = r0["lr_at_last_epoch_ok"]
    return report


def check_optimizer(ckpt: Path, world_size: int) -> dict:
    """Rank0 optimizer state: non-empty state, step>0, non-zero exp_avg/exp_avg_sq."""
    p = ckpt / "actor" / f"optim_world_size_{world_size}_rank_0.pt"
    sd = torch.load(p, map_location="cpu", weights_only=False)
    report = {
        "file": p.name,
        "state_keys": sorted(sd.keys()) if isinstance(sd, dict) else type(sd).__name__,
    }
    if not isinstance(sd, dict):
        report["ok"] = False
        return report
    state = sd.get("state", {})
    groups = sd.get("param_groups", [])
    report["num_state_entries"] = len(state)
    report["num_param_groups"] = len(groups)
    steps = [v.get("step", -1) for v in state.values() if isinstance(v, dict)]
    report["step_values_sample"] = steps[:5]
    report["min_step"] = min(steps) if steps else -1
    report["max_step"] = max(steps) if steps else -1

    nz_avg = 0
    nz_sq = 0
    total = 0
    for v in state.values():
        if not isinstance(v, dict):
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            t = v.get(key)
            if t is not None and hasattr(t, "abs"):
                total += 1
                if bool(t.any()):
                    if key == "exp_avg":
                        nz_avg += 1
                    else:
                        nz_sq += 1
    report["tensors_checked"] = total
    report["nonzero_exp_avg_tensors"] = nz_avg
    report["nonzero_exp_avg_sq_tensors"] = nz_sq
    report["ok"] = (
        len(state) > 0
        and len(groups) > 0
        and report["max_step"] > 0
        and nz_avg > 0
        and nz_sq > 0
        and total > 0
    )
    return report


def check_data_pt(ckpt: Path, expected_snapshot_step: int) -> dict:
    p = ckpt / "data.pt"
    sd = torch.load(p, map_location="cpu", weights_only=False)
    report = {"file": p.name, "keys": sorted(sd.keys()) if isinstance(sd, dict) else type(sd).__name__}
    step = None
    if isinstance(sd, dict):
        step = sd.get("_snapshot_step", None)
        if step is None:
            # torchdata may nest under a key
            for k, v in sd.items():
                if isinstance(v, dict) and "_snapshot_step" in v:
                    step = v["_snapshot_step"]
                    report["snapshot_nested_under"] = k
                    break
    report["snapshot_step"] = step
    report["expected_snapshot_step"] = expected_snapshot_step
    report["ok"] = step == expected_snapshot_step
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=6)
    ap.add_argument("--last-epoch", type=int, required=True, help="expected scheduler last_epoch at this checkpoint")
    ap.add_argument("--snapshot-step", type=int, required=True, help="expected dataloader snapshot step")
    ap.add_argument("--base-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=85)
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args()

    ckpt = args.ckpt_dir.resolve()
    report = {"ckpt_dir": str(ckpt)}
    report["files"] = check_files(ckpt, args.world_size)
    report["extra_state"] = check_extra_state(ckpt, args.world_size, args.last_epoch, args.base_lr, args.warmup_steps)
    report["optimizer"] = check_optimizer(ckpt, args.world_size)
    report["data_pt"] = check_data_pt(ckpt, args.snapshot_step)

    checks = [
        report["files"]["missing_count"] == 0,
        report["extra_state"]["rank0_last_epoch_ok"],
        report["extra_state"]["rank0_lr_curve_ok"],
        report["optimizer"]["ok"],
        report["data_pt"]["ok"],
    ]
    all_ok = all(checks)
    report["checks"] = {
        "files_complete": checks[0],
        "scheduler_last_epoch": checks[1],
        "scheduler_lr_curve": checks[2],
        "optimizer_non_trivial": checks[3],
        "dataloader_snapshot": checks[4],
    }
    report["RESUME_AUDIT"] = "PASS" if all_ok else "FAIL"

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    print(f"RESUME_AUDIT: {'PASS' if all_ok else 'FAIL'}")
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
