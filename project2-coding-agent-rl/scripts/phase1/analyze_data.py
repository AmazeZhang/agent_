#!/usr/bin/env python3
"""Phase 1a: analyze SWE-smith datasets (trajectories + task instances).

Reads all parquet files, aggregates per-instance resolve rates (SWE-Master
"bon pass-rate" proxy — resolved label is the teacher's 0/1 reward), message
lengths, repo distribution. Writes stats JSON to phase1/stats/.
"""
import argparse
import collections
import json
import os

import pyarrow.parquet as pq

DEFAULT_TRAJ_DIR = "/media/imc/data/yzy/agent/project2/datasets/swe-smith-trajectories"
DEFAULT_TASK_DIR = "/media/imc/data/yzy/agent/project2/datasets/swe-smith-tasks"


def read_all_parquet(directory: str):
    """Yield rows from all parquet files in a directory (sorted, stable order)."""
    files = sorted(f for f in os.listdir(directory) if f.endswith(".parquet"))
    for f in files:
        table = pq.read_table(os.path.join(directory, f))
        yield f, table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default=DEFAULT_TRAJ_DIR)
    ap.add_argument("--task-dir", default=DEFAULT_TASK_DIR)
    ap.add_argument("--out", default="/media/imc/data/yzy/agent/project2/phase1/stats/data_overview.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---------- trajectories ----------
    n_traj = 0
    resolved_true = 0
    model_cnt = collections.Counter()
    per_instance = collections.defaultdict(list)  # instance_id -> [resolved, msg_len]
    for f, t in read_all_parquet(args.traj_dir):
        for inst, res, msgs in zip(t["instance_id"], t["resolved"], t["messages"]):
            n_traj += 1
            if bool(res):
                resolved_true += 1
            model_cnt[str(t["model"].to_pylist()[0])]  # placeholder, replaced below
            per_instance[str(inst)].append((bool(res), len(str(msgs))))
        # model column read properly once per file
        for m in t["model"]:
            model_cnt[str(m)] += 1

    inst_stats = {}
    rate_hist = collections.Counter()
    len_hist = collections.Counter()
    for inst, rows in per_instance.items():
        n = len(rows)
        rate = sum(r for r, _ in rows) / n
        max_len = max(l for _, l in rows)
        bucket = round(rate * 10) / 10
        rate_hist[f"{bucket:.1f}"] += 1
        len_bucket = min(max_len // 10000 * 10000, 200000)
        len_hist[str(len_bucket)] += 1
        inst_stats[inst] = {
            "n_traj": n,
            "resolve_rate": rate,
            "n_resolved": sum(r for r, _ in rows),
            "max_msg_chars": max_len,
        }

    lens = sorted(max(l for _, l in rows) for rows in per_instance.values())
    n_inst = len(lens)

    # ---------- task instances ----------
    task_instances = {}  # instance_id -> {repo, problem_len, patch_len}
    repo_cnt = collections.Counter()
    for f, t in read_all_parquet(args.task_dir):
        for inst, repo, prob, patch in zip(t["instance_id"], t["repo"], t["problem_statement"], t["patch"]):
            task_instances[str(inst)] = {
                "repo": str(repo),
                "problem_chars": len(str(prob)),
                "patch_chars": len(str(patch)),
            }
            repo_cnt[str(repo).split("/")[-1]] += 1

    # ---------- overlap ----------
    overlap = [i for i in inst_stats if i in task_instances]

    def pct(k):
        return lens[int(k * (n_inst - 1))]

    overview = {
        "trajectories": {
            "total": n_traj,
            "resolved_true": resolved_true,
            "resolved_rate": round(resolved_true / n_traj, 4),
            "models": dict(model_cnt),
            "instances": n_inst,
        },
        "per_instance": {
            "resolve_rate_hist": dict(sorted(rate_hist.items())),
            "max_msg_chars_percentiles": {"p50": pct(0.5), "p75": pct(0.75), "p90": pct(0.9), "p95": pct(0.95)},
            "max_msg_len_hist_10k_buckets": dict(sorted(len_hist.items(), key=lambda kv: int(kv[0]))),
        },
        "tasks": {
            "total_instances": len(task_instances),
            "repos": len(repo_cnt),
            "repo_counts": dict(repo_cnt.most_common(30)),
        },
        "overlap": {
            "instances_with_both": len(overlap),
            "overlap_rate": round(len(overlap) / max(n_inst, 1), 4),
        },
    }
    with open(args.out, "w") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    print(json.dumps(overview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
