#!/usr/bin/env python3
"""Phase 1a: build the training task pool from SWE-smith datasets.

Filter chain (mirrors SWE-Master's recipe, using our parquet's resolved label
as the teacher reward proxy):
  1. per-instance resolve rate (bon pass-rate), keep 0 < rate < 1 (mid difficulty)
  2. match against the 59k task-instance pool (problem statement + gold patch)
  3. trajectory length: max messages <= MAX_CHARS (conservative char budget
     for the 32k-token training limit; ~3.5 chars/token for Qwen tokenizers)
  4. drop known non-Python/heavy repos (Go, etc.)

Outputs:
  phase1/task_pool/candidates.jsonl     — filtered candidates with metadata
  phase1/stats/candidate_stats.json    — per-repo counts + length stats
"""
import argparse
import collections
import json
import os

import pyarrow.parquet as pq

DEFAULT_TRAJ_DIR = "/media/imc/data/yzy/agent/project2/datasets/swe-smith-trajectories"
DEFAULT_TASK_DIR = "/media/imc/data/yzy/agent/project2/datasets/swe-smith-tasks"
DEFAULT_OUT = "/media/imc/data/yzy/agent/project2/phase1"

# Heavy-dependency / slow-to-build repos: installing + running tests in our
# light pip environment is not worth it for these (fast iteration matters more).
BLOCKLIST_REPOS = {
    # Go
    "blevesearch__bleve", "doug-martin__goqu", "c-bata__go-prompt",
    "incu6us__goimports-reviser",
    # heavy Python: big dep trees / C or Rust extensions / slow test suites
    "dask__dask.5f61e423", "getmoto__moto.694ce1f4", "conan-io__conan.86f29e13",
    "pylint-dev__astroid.b114f6b5", "iterative__dvc.1d6ea681",
    "pandas-dev__pandas.95280573", "encode__starlette.db5063c2",
    "scanny__python-pptx.278b47b1", "pydicom__pydicom.7d361b3d",
    "django-money__django-money.835c1ab8", "pydantic__pydantic.acb0f10f",
    "sqlfluff__sqlfluff.50a1c4b6", "modin-project__modin.8c7799fd",
    "django__daphne.32ac73e1", "django__channels.a144b4b8",
    "pyca__pyopenssl.04766a49", "paramiko__paramiko.23f92003",
    "amueller__word_cloud.ec24191c", "Mimino666__langdetect.a1598f1a",
    "gawel__pyquery.811cd048", "pdfminer__pdfminer.six.1a8bd2f7",
    "tornadoweb__tornado.d5ac65c1", "benoitc__gunicorn.bacbf8aa",
    "pallets__markupsafe.620c06c9", "pudo__dataset.5c2dc8d3",
}


def iter_parquet(directory: str):
    files = sorted(f for f in os.listdir(directory) if f.endswith(".parquet"))
    for f in files:
        yield pq.read_table(os.path.join(directory, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--task-dir", default=DEFAULT_TASK_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-chars", type=int, default=115_000,
                    help="max messages chars per instance (~32k tokens)")
    args = ap.parse_args()

    # ---- 1. aggregate resolve rates per instance ----
    per_inst = collections.defaultdict(list)  # instance_id -> [(resolved, msg_len)]
    for t in iter_parquet(args.traj_dir):
        for inst, res, msgs in zip(t["instance_id"], t["resolved"], t["messages"]):
            per_inst[str(inst)].append((bool(res), len(str(msgs))))
    mid = {i: rows for i, rows in per_inst.items()
           if 0 < sum(r for r, _ in rows) / len(rows) < 1}
    print(f"instances: {len(per_inst)} -> mid-difficulty: {len(mid)}")

    # ---- 2. match task pool ----
    tasks = {}
    for t in iter_parquet(args.task_dir):
        for inst, repo, prob, patch in zip(t["instance_id"], t["repo"], t["problem_statement"], t["patch"]):
            tasks.setdefault(str(inst), {"repo": str(repo), "problem": str(prob), "patch": str(patch)})
    print(f"task pool: {len(tasks)}")

    # ---- 3. filter: matched + length + repo blocklist ----
    candidates, dropped_len, dropped_repo = [], 0, collections.Counter()
    len_vals = []
    for inst, rows in mid.items():
        t = tasks.get(inst)
        if t is None:
            continue
        repo_short = t["repo"].split("/")[-1]
        if repo_short in BLOCKLIST_REPOS:
            dropped_repo[repo_short] += 1
            continue
        max_len = max(l for _, l in rows)
        if max_len > args.max_chars:
            dropped_len += 1
            continue
        len_vals.append(max_len)
        candidates.append({
            "instance_id": inst,
            "repo": t["repo"],
            "problem_statement": t["problem"],
            "gold_patch": t["patch"],
            "n_traj": len(rows),
            "resolve_rate": sum(r for r, _ in rows) / len(rows),
            "max_msg_chars": max_len,
            "traj_meta": [{"resolved": r, "msg_chars": l} for r, l in rows],
        })

    # ---- output ----
    os.makedirs(os.path.join(args.out, "task_pool"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "stats"), exist_ok=True)
    cand_path = os.path.join(args.out, "task_pool", "candidates.jsonl")
    with open(cand_path, "w") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    repo_cnt = collections.Counter(c["repo"].split("/")[-1] for c in candidates)
    stats = {
        "mid_difficulty_instances": len(mid),
        "candidates_after_all_filters": len(candidates),
        "dropped_by_length": dropped_len,
        "dropped_by_repo_blocklist": dict(dropped_repo),
        "max_msg_chars": {
            "p50": sorted(len_vals)[len(len_vals)//2] if len_vals else 0,
            "max": max(len_vals) if len_vals else 0,
        },
        "per_repo_counts": dict(repo_cnt.most_common(40)),
    }
    with open(os.path.join(args.out, "stats", "candidate_stats.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"wrote {cand_path}")


if __name__ == "__main__":
    main()
