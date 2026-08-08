"""Generate SWE-agent run-batch configs for sanitized SWE-smith instances.

Turns one sanitized local instance (local-instances/<name>-sanitized.json) into a
run-batch config for a numbered rollout run, using the proven run8 template
(DeepSeek-v4-flash, 40 calls, per-instance cost limit $0.05, single worker).
"""

from __future__ import annotations

import argparse
import json
import yaml
from pathlib import Path

PILOT_ROOT = Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20")
TEMPLATE = PILOT_ROOT / "runs/deepseek-v4-flash-run8-sanitized/run_batch.config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance_json", type=Path, help="local-instances/<name>-sanitized.json")
    parser.add_argument("run_id", help="e.g. deepseek-v4-flash-run10")
    parser.add_argument("--out-dir", type=Path, default=PILOT_ROOT / "local-instances")
    args = parser.parse_args()

    instances = json.loads(args.instance_json.read_text(encoding="utf-8"))
    if len(instances) != 1:
        raise SystemExit(f"expected one instance, found {len(instances)}")
    instance_id = instances[0]["instance_id"]

    raw = TEMPLATE.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        cfg = json.loads(cfg)

    cfg["instances"]["path"] = str(args.instance_json.resolve())
    cfg["output_dir"] = str(PILOT_ROOT / "runs" / args.run_id)

    output = args.out_dir / f"{args.run_id}.config.yaml"
    output.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "instance_id": instance_id, "config": str(output)}))


if __name__ == "__main__":
    main()
