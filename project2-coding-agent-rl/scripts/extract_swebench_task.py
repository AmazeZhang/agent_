"""Extract gold source/test patches for one local SWE-bench instance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("instance_id")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    instances = json.loads(args.dataset.read_text(encoding="utf-8"))
    matches = [row for row in instances if row.get("instance_id") == args.instance_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one {args.instance_id!r}, found {len(matches)}")
    instance = matches[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gold.patch").write_text(instance["patch"], encoding="utf-8")
    (args.output_dir / "test.patch").write_text(instance["test_patch"], encoding="utf-8")
    metadata = {
        key: instance.get(key)
        for key in ("instance_id", "repo", "base_commit", "FAIL_TO_PASS", "PASS_TO_PASS")
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
