"""WP6: build the GRPO smoke training dataset from WP2 artifacts.

Selects the N tasks with the smallest hidden-test suites (fast reward
evaluations during rollout), excluding the WP7 holdout set. Each row
carries everything the workflow + reward function need: problem
statement, F2P list, eval repo path, eval venv path.

Registers the dataset via rllm's DatasetRegistry so the verl parquet
companion (_verl.parquet with extra_info) is generated automatically.

Usage: python scripts/build_grpo_smoke_data.py [--n 4] [--name p2_swe_smoke]
       python scripts/build_grpo_smoke_data.py --tasks short1,short2,short3,short4 --name p2_swe_smoke_v2
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from rllm.data.dataset import DatasetRegistry

PILOT_ROOT = Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20")
EVAL_VENVS = Path("/media/imc/data/yzy/agent/project2/eval-venvs")

# Never train on the WP7 holdout tasks.
HOLDOUT = {
    "funcy-curry-compose-3u9hti2d",
    "pygments-groff-0jqqr58z",
    "stackprinter-1i9gep13",
    "oauthlib-signature-1bsv3m8l",
    "boltons-7nlifqzn",
}


def junit_test_count(result_xml: Path) -> int:
    try:
        return len(ET.parse(result_xml).getroot().findall(".//testcase"))
    except Exception:
        return 10**9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=4, help="number of tasks to select")
    parser.add_argument("--name", type=str, default="p2_swe_smoke")
    parser.add_argument("--tasks", type=str, default="",
                        help="explicit comma-separated task list (overrides auto-ranking; "
                             "used for the SFT-warm-start GRPO run whose tasks must be "
                             "solvable by the SFT'd policy)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the dataset exists")
    args = parser.parse_args()

    if args.tasks:
        selected = [s.strip() for s in args.tasks.split(",") if s.strip()]
        print(f"selected tasks (explicit): {selected}")
    else:
        # Rank tasks by hidden-test suite size (from WP2 eval evidence).
        ranked = []
        for result_xml in sorted((PILOT_ROOT / "candidate-evals").glob("*/result.xml")):
            task_short = result_xml.parent.name.split("-deepseek")[0]
            if task_short in HOLDOUT:
                continue
            ranked.append((task_short, junit_test_count(result_xml)))
        ranked.sort(key=lambda x: x[1])
        selected = [short for short, _ in ranked[: args.n]]
        print(f"selected tasks (smallest suites, holdout excluded): {selected}")

    # Build rows from the sanitized instance metadata. The eval venv is named
    # after the registry's repo_dir (owner__repo), NOT the instance's
    # repo_name (which is a local filesystem path).
    registry = json.loads((Path(__file__).resolve().parent / "tasks-registry.json").read_text())
    # oauthlib-1bsv3m8l (run1 era) was never materialized as a sanitized
    # instance file — fall back to the master 20-task list.
    master = {i["instance_id"]: i for i in json.loads((PILOT_ROOT / "sweagent_instances.json").read_text())}
    rows = []
    for short in selected:
        inst_file = PILOT_ROOT / "local-instances" / f"{short}-sanitized.json"
        if inst_file.exists():
            inst = json.loads(inst_file.read_text())[0]
        else:
            match = [i for i in master.values() if i["instance_id"].split("combine_file__")[-1] == short.split("-")[-1]]
            if not match:
                raise SystemExit(f"no instance data for {short}")
            inst = match[0]
        venv_name = registry["tasks"][short]["repo_dir"]
        row = {
            "instance_id": inst["instance_id"],
            "repo_dir": short,
            "problem_statement": inst["problem_statement"],
            "f2p": inst.get("FAIL_TO_PASS", []),
            "p2p": inst.get("PASS_TO_PASS", []),
            "eval_repo": str(PILOT_ROOT / "eval-repos" / short),
            "eval_venv": str(EVAL_VENVS / venv_name),
            "data_source": args.name,
        }
        rows.append(row)
        print(f"  {short}: f2p={len(row['f2p'])} p2p={len(row['p2p'])} venv={venv_name}")

    if DatasetRegistry.dataset_exists(args.name, "train") and not args.force:
        print(f"dataset {args.name} already registered — use --force to rebuild")
        return

    ds = DatasetRegistry.register_dataset(args.name, rows, "train")
    print(f"registered {args.name} -> {ds.get_data_path()}")
    print(f"verl parquet: {ds.get_verl_data_path()}")


if __name__ == "__main__":
    main()
