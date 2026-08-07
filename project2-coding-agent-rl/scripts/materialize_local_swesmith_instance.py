"""Create a one-row SWE-agent config using an already cloned local repository."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", type=Path)
    parser.add_argument("instance_id")
    parser.add_argument("local_repo", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = json.loads(args.instances.read_text(encoding="utf-8"))
    matches = [row for row in rows if row["instance_id"] == args.instance_id]
    if len(matches) != 1:
        raise SystemExit(f"expected one matching instance, found {len(matches)}")
    if not (args.local_repo / ".git").is_dir():
        raise SystemExit(f"not a git checkout: {args.local_repo}")
    parent_check = subprocess.run(
        ["git", "-C", str(args.local_repo), "cat-file", "-e", "HEAD^"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if parent_check.returncode == 0:
        raise SystemExit(
            "refusing repository with accessible parent history; use a depth-1 sanitized snapshot"
        )

    row = dict(matches[0])
    row["repo_name"] = str(args.local_repo.resolve())
    row["base_commit"] = "HEAD"
    row["source_repo_name"] = matches[0]["repo_name"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([row], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"instance_id": args.instance_id, "repo_name": row["repo_name"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
