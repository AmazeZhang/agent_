#!/usr/bin/env python
"""基线 v0 在 val 集多次重跑（r3 协议修正：gate 对照基准同尺度）。

背景（2026-08-08 方法论修正）:
  r2 门控用 40 任务基线的 0.900 作参照，但候选在 val 8 上评测——不同尺度。
  基线在 val 8 子集实测为 0.875（7/8，失败任务 27 恰好在 val 中）。
  LLM API 层存在非确定性（temperature=0 下任务 27 仍时对时错），
  单次重跑噪声大 → 本脚本对基线 val 8 重跑 N 次，按任务多数票计算通过率，
  作为 r3 及以后各臂 gate 的对照基准。

用法（agent-lightning venv + DeepSeek env）:
  PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
  .venvs/agent-lightning/bin/python scripts/run_baseline_val_rerun.py \
      --repeats 3 --out runs/baseline_val_rerun.json

产物:
  runs/baseline_val_rerun.json  {task_votes, majority_rate, mean_rate, repeats, prompt_hash}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ1 = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJ1))

import tau2_deepseek_cli  # noqa: F401,E402

from optimizers.tau2_rollout import set_task_pool, tau2_rollout  # noqa: E402
from evaluation.metrics import load_results  # noqa: E402
from resources.loader import load_resources  # noqa: E402

RESULTS_JSON = Path("/media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json")
DATASETS = PROJ1 / "data" / "datasets"


class _PromptText:
    def __init__(self, template: str):
        self.template = template


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(PROJ1 / "runs" / "baseline_val_rerun.json"))
    args = ap.parse_args()

    # 与各臂 runner 相同的 rollout 记录文件
    os.environ.setdefault("P1_ROLLOUT_LOG", str(PROJ1 / "runs" / "baseline_val_rerun_rollout.jsonl"))

    results = load_results(RESULTS_JSON)
    set_task_pool(results["tasks"])
    val = [json.loads(l) for l in (DATASETS / "val.jsonl").read_text().splitlines()]
    v0 = load_resources(0)
    prompt = _PromptText(v0["system_prompt"])

    votes: dict[str, list[float]] = {str(t["id"]): [] for t in val}
    t0 = time.time()
    for rep in range(args.repeats):
        for task in val:
            r = tau2_rollout(task, prompt)
            votes[str(task["id"])].append(r)
        done = sum(len(v) for v in votes.values())
        print(f"==> 第 {rep + 1}/{args.repeats} 轮完成（{done}/{args.repeats * len(val)} 次仿真）")

    majority_rate = sum(
        1.0 for v in votes.values() if sum(v) >= (args.repeats + 1) // 2
    ) / len(val)
    mean_rate = sum(sum(v) for v in votes.values()) / (len(val) * args.repeats)

    record = {
        "schema_version": 1,
        "prompt_hash": v0["system_prompt"][:40],
        "repeats": args.repeats,
        "task_votes": {tid: {"rewards": vs, "majority_pass": sum(vs) >= (args.repeats + 1) // 2}
                       for tid, vs in votes.items()},
        "majority_rate": majority_rate,
        "mean_rate": mean_rate,
        "duration_s": round(time.time() - t0, 1),
        "note": "基线 v0 在 val 8 上多次重跑（LLM 非确定性降噪）；r3 起 gate 对照基准",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"==> 基线 val8 重跑 {args.repeats} 次: 多数票 {majority_rate:.3f} | 均值 {mean_rate:.3f}")
    print(f"==> 记录: {out}")


if __name__ == "__main__":
    main()
