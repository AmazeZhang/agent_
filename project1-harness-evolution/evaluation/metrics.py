#!/usr/bin/env python
"""M2: 指标计算与基线快照。

输入: 运行目录的 results.json（tau2 输出格式，见 scripts/run_tau2_baseline.py）
输出: 指标 dict（写入 baseline_summary.json 或 eval 报告）

指标口径（SPEC 02）:
- success_rate: reward == 1.0 的仿真占比（严格口径；任务级成功取其任一仿真成功）。
- cost: agent 成本 / user 成本 / 合计（美元，由 DeepSeek 定价在 tau2 侧记账）。
- failure categories: 失败仿真按 termination_reason 归组。
- 全部指标在运行时统计；不修改任何输入文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def compute_metrics(results: dict, name: str | None = None) -> dict:
    """从 tau2 results.json 计算指标。

    任务级成功 = 该任务任一仿真 reward == 1.0（多 trials 时）
    仿真级成功率 = reward == 1.0 的仿真 / 全部仿真
    """
    sims = results.get("simulations", [])
    tasks = {str(t["id"]): t for t in results.get("tasks", [])}

    per_sim = []
    for s in sims:
        rw = s.get("reward_info") or {}
        per_sim.append({
            "id": s["id"],
            "task_id": str(s["task_id"]),
            "reward": rw.get("reward"),
            "termination_reason": s.get("termination_reason"),
            "agent_cost": s.get("agent_cost"),
            "user_cost": s.get("user_cost"),
            "duration_s": s.get("duration"),
        })

    rewards = [p["reward"] for p in per_sim if p["reward"] is not None]
    n_sim = len(per_sim)
    sim_success = sum(1 for r in rewards if r == 1.0)

    # 任务级成功
    task_reward = {}
    for p in per_sim:
        task_reward.setdefault(p["task_id"], []).append(p["reward"])
    task_success = sum(1 for rw in task_reward.values() if 1.0 in rw)
    n_task = len(task_reward)

    # 失败归组
    failures = [p for p in per_sim if p["reward"] != 1.0]
    by_reason: dict[str, int] = {}
    for p in failures:
        reason = p["termination_reason"] or "unknown"
        by_reason[reason] = by_reason.get(reason, 0) + 1

    total_cost = sum((p["agent_cost"] or 0) + (p["user_cost"] or 0) for p in per_sim)
    total_sim_time = sum(p["duration_s"] or 0 for p in per_sim)

    return {
        "name": name or results.get("info", {}).get("run_name"),
        "num_tasks": n_task,
        "num_simulations": n_sim,
        "task_success_rate": (task_success / n_task) if n_task else None,
        "sim_success_rate": (sim_success / n_sim) if n_sim else None,
        "num_task_success": task_success,
        "num_sim_success": sim_success,
        "num_failures": len(failures),
        "failure_by_reason": by_reason,
        "total_cost_usd": round(total_cost, 6),
        "total_sim_time_s": round(total_sim_time, 1),
        "per_task": task_reward,  # task_id -> [reward...]
    }


def summarize_baseline(results_path: Path | str, out_path: Path | str | None = None) -> dict:
    """生成/更新基线快照（含时间戳），写入 data/baseline_summary.json。"""
    results = load_results(results_path)
    metrics = compute_metrics(results)
    snapshot = {
        "schema_version": 2,
        "timestamp": results.get("timestamp"),
        "results_path": str(results_path),
        "metrics": metrics,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return snapshot


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="tau2 运行 results.json")
    ap.add_argument("--out", help="可选：指标快照输出路径")
    args = ap.parse_args()

    snap = summarize_baseline(args.results, args.out)
    m = snap["metrics"]
    print(f"任务 {m['num_tasks']} | 仿真 {m['num_simulations']} | "
          f"任务成功率 {m['task_success_rate']} | 仿真成功率 {m['sim_success_rate']}")
    print(f"失败 {m['num_failures']} (按原因: {m['failure_by_reason']}) | "
          f"总成本 ${m['total_cost_usd']} | 总模拟时间 {m['total_sim_time_s']}s")
    if args.out:
        print(f"==> 快照: {args.out}")
