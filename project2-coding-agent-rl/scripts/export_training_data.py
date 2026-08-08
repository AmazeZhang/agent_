"""Export trusted trajectories (patch + trace + integrity + eval verdict) into
training-data/, partitioned by reward, with an isolation manifest for invalid
runs and a manifest.json tracing every file to its PILOT_ROOT artifact.

Deterministic: reads only evaluations/summary.json (itself rebuilt from raw
artifacts by pipeline_summary.py) and copies files under PILOT_ROOT.

Usage: python scripts/export_training_data.py [--pilot-root PATH] [--dest PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

INVALID_REASONS = {
    "deepseek-v4-flash-run4": "parent history accessible in local upload",
    "deepseek-v4-flash-run5": "model read Bug Patch / ground-truth implementation",
    "deepseek-v4-flash-run6": "answer accessible via parent history",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path,
                        default=Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20"))
    parser.add_argument("--dest", type=Path,
                        default=Path("/media/imc/data/yzy/agent/project2/training-data"))
    args = parser.parse_args()

    root = args.pilot_root
    summary = json.loads((root / "evaluations" / "summary.json").read_text(encoding="utf-8"))

    dest = args.dest
    (dest / "successes").mkdir(parents=True, exist_ok=True)
    (dest / "failures").mkdir(parents=True, exist_ok=True)
    (dest / "invalid").mkdir(parents=True, exist_ok=True)

    manifest: dict = {"schema_version": 1, "entries": []}
    copied = 0

    for run_id, entry in sorted(summary["per_run"].items()):
        instance_id = entry.get("instance_id", "")
        if entry.get("status") == "invalid":
            # summary.py records invalid runs without instance_id; they are
            # isolated with a reason file regardless.
            (dest / "invalid" / run_id).mkdir(parents=True, exist_ok=True)
            reason = INVALID_REASONS.get(run_id, entry.get("reason", "unknown"))
            (dest / "invalid" / run_id / "reason.txt").write_text(str(reason), encoding="utf-8")
            manifest["entries"].append({
                "run_id": run_id,
                "status": "invalid",
                "files": {"reason": f"invalid/{run_id}/reason.txt"},
            })
            continue
        if not instance_id:
            continue

        item = {
            "run_id": run_id,
            "instance_id": instance_id,
            "status": entry.get("status"),
            "reward": entry.get("reward"),
            "files": {},
        }

        if entry.get("reward") not in (0, 1):
            continue  # pending / unknown — not exported yet

        bucket = "successes" if entry["reward"] == 1 else "failures"
        target = dest / bucket / run_id
        target.mkdir(parents=True, exist_ok=True)

        for key, rel in (("patch", entry.get("patch")), ("traj", entry.get("traj"))):
            if rel:
                src = root / rel
                if src.exists():
                    out = target / src.name
                    shutil.copy2(src, out)
                    item["files"][key] = str(out.relative_to(dest))

        # integrity.json lives inside the run dir next to patch/traj
        integrity = root / "runs" / run_id / instance_id / "integrity.json"
        if integrity.exists():
            shutil.copy2(integrity, target / "integrity.json")
            item["files"]["integrity"] = str((target / "integrity.json").relative_to(dest))

        # eval verdict: candidate-evals/<task>-<run_id>/eval_result.json
        er = next((root / "candidate-evals").glob(f"*-{run_id}/eval_result.json"), None)
        if er and er.exists():
            shutil.copy2(er, target / "eval_result.json")
            item["files"]["eval_result"] = str((target / "eval_result.json").relative_to(dest))

        if item["files"]:
            copied += 1
            manifest["entries"].append(item)

    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported {copied} trusted runs -> {dest}")
    print(f"successes: {len(list((dest/'successes').iterdir()))}, "
          f"failures: {len(list((dest/'failures').iterdir()))}, "
          f"invalid: {len(list((dest/'invalid').iterdir()))}")


if __name__ == "__main__":
    main()
