"""Curate a small, reproducible SWE-smith pilot without downloading images.

The output keeps the official bug patch and test IDs for later strict evaluation,
and also emits SWE-agent's simple-file format for GitHub branch rollouts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


REPOSITORIES = (
    "swesmith/oauthlib__oauthlib.1fd52536",
    "swesmith/pygments__pygments.27649ebb",
    "swesmith/Suor__funcy.207a7810",
    "swesmith/bottlepy__bottle.a8dfef30",
    "swesmith/cknd__stackprinter.219fcc52",
    "swesmith/mahmoud__boltons.3bfcfdd0",
    "swesmith/agronholm__typeguard.b6a7e438",
    "swesmith/PyCQA__flake8.cf1542ce",
    "swesmith/jd__tenacity.0d40e76f",
    "swesmith/pytest-dev__iniconfig.16793ead",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--per-repo", type=int, default=2)
    parser.add_argument("--scan-limit", type=int, default=10_000)
    args = parser.parse_args()

    selected: dict[str, list[dict]] = defaultdict(list)
    dataset = load_dataset("SWE-bench/SWE-smith", split="train", streaming=True)
    for index, row in enumerate(dataset):
        if index >= args.scan_limit:
            break
        repo = row["repo"]
        if repo not in REPOSITORIES or len(selected[repo]) >= args.per_repo:
            continue
        if not row["problem_statement"].strip():
            continue
        if not (1 <= len(row["FAIL_TO_PASS"]) <= 5):
            continue
        if len(row["PASS_TO_PASS"]) < 1 or len(row["patch"]) > 8_000:
            continue
        selected[repo].append(dict(row))
        if all(len(selected[repo]) >= args.per_repo for repo in REPOSITORIES):
            break

    missing = {repo: args.per_repo - len(selected[repo]) for repo in REPOSITORIES if len(selected[repo]) < args.per_repo}
    if missing:
        raise SystemExit(f"insufficient qualifying tasks: {missing}")

    official = [row for repo in REPOSITORIES for row in selected[repo]]
    simple = [
        {
            "instance_id": row["instance_id"],
            "image_name": "agent/swe-rex-py311:20260806",
            "official_image_name": row["image_name"],
            "repo_name": f"https://github.com/{row['repo']}.git",
            "base_commit": row["instance_id"],
            "problem_statement": row["problem_statement"],
            "FAIL_TO_PASS": row["FAIL_TO_PASS"],
            "PASS_TO_PASS": row["PASS_TO_PASS"],
        }
        for row in official
    ]
    manifest = {
        "schema_version": 1,
        "source": "SWE-bench/SWE-smith train split",
        "streaming": True,
        "scan_limit": args.scan_limit,
        "selection": {
            "repositories": list(REPOSITORIES),
            "per_repo": args.per_repo,
            "nonempty_problem_statement": True,
            "fail_to_pass_range": [1, 5],
            "minimum_pass_to_pass": 1,
            "maximum_bug_patch_bytes": 8_000,
        },
        "count": len(official),
        "note": "The generic local image is for rollout smoke tests. Final reward must use task test IDs and a dependency-compatible evaluator; official image names are preserved.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "official_tasks.json").write_text(
        json.dumps(official, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "sweagent_instances.json").write_text(
        json.dumps(simple, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
