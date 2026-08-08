"""Build a diff-format SFT parquet from the trusted gold patches.

The first SFT smoke (build_smoke_sft_data.py) distilled the SWE-agent
trajectory steps — tool calls (bash / str_replace_editor / submit). A policy
trained on those emits tool calls, which the GRPO single-turn reward
(git apply of a standalone diff) can never accept: runs 9-11 all produced
zero rewards for exactly this reason.

This builder distills the trustworthy gold PATCHES instead, in the exact
input format the GRPO workflow expects (system prompt = swe_workflow's
SYSTEM_PROMPT, user = problem statement, assistant = the unified diff).
WP7 holdouts are excluded; oauthlib-1bsv3m8l is a special case — already
removed from the holdout eval set because SFT#1 contaminated it.

Usage: python scripts/build_patch_sft_data.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PILOT_ROOT = Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20")
SUCCESSES = Path("/media/imc/data/yzy/agent/project2/training-data/successes")

SYSTEM_PROMPT = (
    "You are an expert software engineer. You are given a GitHub issue "
    "description. Produce a minimal patch (unified diff) that fixes the "
    "issue. Output ONLY the diff, inside a ```diff code block, with no "
    "surrounding commentary."
)

HOLDOUT = {
    "funcy-curry-compose-3u9hti2d",
    "pygments-groff-0jqqr58z",
    "stackprinter-1i9gep13",
    "boltons-7nlifqzn",
    # oauthlib-signature-1bsv3m8l intentionally NOT excluded: SFT#1 already
    # trained on it (documented deviation), it is out of the WP7 eval set.
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("/media/imc/data/yzy/agent/project2/smoke-data/sft-patch-train.parquet"))
    args = parser.parse_args()

    master = {i["instance_id"]: i for i in json.loads((PILOT_ROOT / "sweagent_instances.json").read_text())}
    rows = []
    used = []
    for run_dir in sorted(SUCCESSES.iterdir()):
        if not run_dir.is_dir():
            continue
        patch = next(run_dir.glob("*.patch"), None)
        if patch is None:
            continue
        iid = patch.stem
        iid_sha = iid.split("combine_file__")[-1]
        if any(short.split("-")[-1] == iid_sha for short in HOLDOUT):
            print(f"skip (holdout): {iid}")
            continue
        inst = master.get(iid)
        if inst is None:
            print(f"skip (no master entry): {run_dir.name}")
            continue
        diff = patch.read_text(encoding="utf-8")
        if not diff.strip():
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": inst["problem_statement"]},
                {"role": "assistant", "content": diff},
            ]
        })
        used.append(iid)
        print(f"  {iid}: {len(diff.splitlines())} diff lines")

    if not rows:
        raise SystemExit("no rows produced")
    table = pa.table({"messages": [json.dumps(r["messages"], ensure_ascii=False) for r in rows]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
