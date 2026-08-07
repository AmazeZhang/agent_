#!/usr/bin/env python
"""M1: tau2 任务 → Agent Lightning Dataset(jsonl) 适配器。

输入: baseline 运行目录的 results.json（含任务定义与执行结果）
输出: data/datasets/tau2_retail.jsonl + data/datasets/task_manifest.json

字段:
  {"id": "<task_id>", "task_input": {...任务上下文...}, "expected": null}
tau2 是对话式任务，reward 由 DB 校验在 rollout 内计算，"expected" 无单一答案，置 null。
"""

import argparse
import json
import sys
from pathlib import Path

PROJ1 = Path(__file__).resolve().parent.parent
DATASETS = PROJ1 / "data" / "datasets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="baseline 运行目录的 results.json 路径")
    ap.add_argument("--out-name", default="tau2_retail", help="输出 jsonl 名")
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    tasks = {str(t["id"]): t for t in results["tasks"]}

    # 与 summary 的 per_task 对齐：只收录有执行结果的仿真
    sim_by_task = {}
    for sim in results["simulations"]:
        tid = str(sim["task_id"])
        sim_by_task.setdefault(tid, []).append(sim)

    DATASETS.mkdir(parents=True, exist_ok=True)
    out_path = DATASETS / f"{args.out_name}.jsonl"
    manifest = {"schema_version": 1, "source": args.results, "tasks": {}}

    with out_path.open("w", encoding="utf-8") as f:
        for tid, task in sorted(tasks.items(), key=lambda kv: int(kv[0])):
            entry = {
                "id": tid,
                "task_input": {
                    "purpose": (task.get("description") or {}).get("purpose"),
                    "user_scenario": task.get("user_scenario"),
                    "task_set": results.get("info", {}).get("task_set_name"),
                },
                "expected": None,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            manifest["tasks"][tid] = {
                "simulations": [s["id"] for s in sim_by_task.get(tid, [])],
                "max_reward": max(
                    ((s.get("reward_info") or {}).get("reward") for s in sim_by_task.get(tid, [])),
                    default=None,
                ),
            }

    manifest["num_tasks"] = len(tasks)
    manifest_path = DATASETS / "task_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"==> dataset: {out_path} ({len(tasks)} 任务)")
    print(f"==> manifest: {manifest_path}")


if __name__ == "__main__":
    main()
