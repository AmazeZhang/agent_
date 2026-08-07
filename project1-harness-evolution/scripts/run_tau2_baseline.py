#!/usr/bin/env python
"""M1 基线采集：程序化调用 tau2 执行 retail 任务并汇总 baseline_summary.json。

用法（tau2 venv 内）:
  .venvs/tau2/bin/python scripts/run_tau2_baseline.py --task-ids 0-19 --name retail20-v1
  .venvs/tau2/bin/python scripts/run_tau2_baseline.py --task-ids 3,4,5 --name smoke

约定:
- 结果落盘到数据盘 /media/imc/data/yzy/agent/project1/baseline/<name>/（大文件不进 Git）。
- 汇总文件 data/baseline_summary.json 进 Git（小文件，可追溯）。
- DeepSeek 配置复用 scripts/tau2_deepseek_cli.py 的注册与 evaluator 补丁。
"""

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ1 = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import tau2_deepseek_cli  # noqa: F401,E402  # 触发 DeepSeek 模型注册 + evaluator 补丁

from tau2.run import run_domain  # noqa: E402
from tau2.data_model.simulation import TextRunConfig  # noqa: E402


DATA_ROOT = Path(os.environ.get(
    "P1_BASELINE_ROOT", "/media/imc/data/yzy/agent/project1/baseline"
))
SUMMARY_PATH = PROJ1 / "data" / "baseline_summary.json"


def parse_task_ids(spec: str) -> list[str]:
    ids: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.extend(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            ids.append(part)
    return ids


def llm_args() -> dict:
    return {
        "api_base": os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        "max_tokens": 8192,
        "temperature": 0.0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids", required=True, help="如 0-19 或 3,4,5")
    ap.add_argument("--name", required=True, help="运行名，如 retail40-v1")
    ap.add_argument("--seed", type=int, default=301)
    ap.add_argument("--max-concurrency", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--num-trials", type=int, default=1)
    args = ap.parse_args()

    task_ids = parse_task_ids(args.task_ids)
    model = f"openai/{os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"
    run_dir = DATA_ROOT / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> 任务数: {len(task_ids)}  {task_ids[:10]}{'...' if len(task_ids) > 10 else ''}")
    print(f"==> 模型: {model}")
    print(f"==> 输出: {run_dir}")

    config = TextRunConfig(
        domain="retail",
        task_ids=task_ids,
        num_trials=args.num_trials,
        agent="llm_agent",
        llm_agent=model,
        llm_args_agent=llm_args(),
        user="user_simulator",
        llm_user=model,
        llm_args_user=llm_args(),
        max_steps=args.max_steps,
        timeout=300,
        max_retries=1,
        seed=args.seed,
        max_concurrency=args.max_concurrency,
        save_to=str(run_dir),
        log_level="INFO",
    )
    run_domain(config)

    # ---- 汇总 ----
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"!! 未找到 {results_path}", file=sys.stderr)
        sys.exit(1)

    results = json.loads(results_path.read_text())
    per_task = {}
    for sim in results["simulations"]:
        tid = str(sim["task_id"])
        rw = sim.get("reward_info") or {}
        per_task[tid] = {
            "task_id": tid,
            "simulation_id": sim["id"],
            "reward": rw.get("reward"),
            "db_match": (rw.get("db_check") or {}).get("db_match"),
            "agent_cost": sim.get("agent_cost"),
            "user_cost": sim.get("user_cost"),
            "duration_s": sim.get("duration"),
            "termination_reason": sim.get("termination_reason"),
            "trajectory": f"{run_dir}/results.json#simulations/{sim['id']}",
        }

    rewards = [v["reward"] for v in per_task.values() if v["reward"] is not None]
    summary = {
        "schema_version": 1,
        "name": args.name,
        "timestamp": results.get("timestamp"),
        "seed": args.seed,
        "model": model,
        "num_tasks_planned": len(task_ids),
        "num_tasks_result": len(per_task),
        "success_rate": (sum(1 for r in rewards if r == 1.0) / len(rewards)) if rewards else None,
        "total_agent_cost": round(sum(v["agent_cost"] or 0 for v in per_task.values()), 6),
        "total_user_cost": round(sum(v["user_cost"] or 0 for v in per_task.values()), 6),
        "failures": sorted(
            [tid for tid, v in per_task.items() if v["reward"] != 1.0]
        ),
        "per_task": per_task,
        "results_path": str(results_path),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"==> 汇总写入 {SUMMARY_PATH}")
    print(f"==> 成功率: {summary['success_rate']}  ({sum(1 for r in rewards if r == 1.0)}/{len(rewards)})")
    print(f"==> 失败任务: {summary['failures']}")


if __name__ == "__main__":
    main()
