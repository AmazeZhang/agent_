#!/usr/bin/env python
"""M1: 基线失败轨迹的 AgentRx 自动诊断。

流程:
  results.json → 过滤失败仿真(reward != 1.0) → 转换为 AgentRx wrapper 单文件
  → 调用 run_agentrx_deepseek.sh（DeepSeek 六阶段诊断）→ 汇总预测结果

用法（agentrx venv 内）:
  .venvs/agentrx/bin/python scripts/diagnose_baseline_failures.py \
      --results /media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json \
      --run-dir /media/imc/data/yzy/agent/project1/diagnostics/retail40-v1

说明:
- 自然任务失败轨迹无人工标注，输出为"预测"结果（类别/步骤/证据/token），
  供闭环反馈使用；如需评测诊断准确率需另行人工标注。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJ1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ1 / "tracing"))
from tau2_to_agentrx import convert_message, instruction_from_task  # noqa: E402

DIAG_SUMMARY = PROJ1 / "data" / "diagnostics" / "summary.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="baseline 运行目录的 results.json")
    ap.add_argument("--run-dir", required=True, help="AgentRx 输出目录（数据盘）")
    args = ap.parse_args()

    batch = json.loads(Path(args.results).read_text())
    tasks = {str(t.get("id")): t for t in batch.get("tasks", [])}
    failures = [
        s for s in batch.get("simulations", [])
        if (s.get("reward_info") or {}).get("reward") != 1.0
    ]
    if not failures:
        print("!! 没有失败轨迹，无需诊断")
        return

    staging = Path(args.run_dir) / "trajectories"
    staging.mkdir(parents=True, exist_ok=True)
    for sim in failures:
        task = tasks.get(str(sim.get("task_id")), {})
        events = []
        policy = sim.get("policy")
        if policy:
            events.append({"role": "system", "content": str(policy)})
        events.extend(convert_message(m) for m in sim.get("messages", []))
        wrapper = {
            "trajectory_id": sim["id"],
            "task_id": str(sim.get("task_id")),
            "instruction": instruction_from_task(task),
            "reward": (sim.get("reward_info") or {}).get("reward"),
            "events": events,
        }
        (staging / f"{sim['id']}.json").write_text(
            json.dumps(wrapper, ensure_ascii=False, indent=2)
        )

    print(f"==> 失败轨迹 {len(failures)} 条 → {staging}")

    # 注意: AgentRx 目录输入时所有轨迹共享 run_dir，judge 输出会互相覆盖
    # （judge_output/runs/run1.json 只保留最后一条）。因此每条轨迹用独立
    # run_dir（run_dir/traj_<id>/），保证每条诊断结果可回溯。
    per_traj_dirs = []
    for sim in failures:
        traj_dir = Path(args.run_dir) / f"traj_{sim['id']}"
        per_traj_dirs.append(traj_dir)
    cmd = [
        str(PROJ1 / "scripts" / "run_agentrx_deepseek.sh"),
        str(staging),
        "--domain", "tau",
        "--endpoint", "azure",
        "--dynamic-mode", "oneshot",
        "--run-dir", str(per_traj_dirs[0]),
    ]
    print(f"==> 启动 AgentRx 诊断（逐轨迹独立 run_dir）: {' '.join(cmd[:4])} ...")
    failed_trajs = []  # 单轨迹诊断失败不中断整体（AgentRx 生成检查代码有已知 bug）
    for sim, traj_dir in zip(failures, per_traj_dirs):
        # 已诊断过的轨迹跳过（重跑容错）
        if (traj_dir / "judge_output" / "runs" / "run1.json").exists():
            print(f"==> 跳过已诊断轨迹 {sim['id']} (task={sim.get('task_id')})")
            continue
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_cmd = [
            str(PROJ1 / "scripts" / "run_agentrx_deepseek.sh"),
            str(staging / f"{sim['id']}.json"),
            "--domain", "tau",
            "--endpoint", "azure",
            "--dynamic-mode", "oneshot",
            "--run-dir", str(traj_dir),
        ]
        print(f"==> 诊断轨迹 {sim['id']} (task={sim.get('task_id')}) ...")
        rc = subprocess.run(traj_cmd).returncode
        if rc != 0:
            failed_trajs.append({"trajectory_id": sim["id"], "task_id": str(sim.get("task_id")),
                                 "error": "AgentRx pipeline failed (see console)"})
            print(f"!! 轨迹 {sim['id']} 诊断失败（rc={rc}），继续下一条")

    # ---- 汇总预测结果 ----
    # judge 输出里的 task_id 实际是 wrapper 的 trajectory_id（uuid），
    # 用 traj_<sim_id> 目录名反查 tau2 任务 id（wrapper 文件字段）。
    sim_id_to_task = {s["id"]: str(s.get("task_id")) for s in failures}
    entries = []
    for run1 in sorted(Path(args.run_dir).glob("traj_*/judge_output/runs/run1.json")):
        payload = json.loads(run1.read_text())
        result = payload["detailed_results"][0]
        prediction = result["failures"][0]
        sim_id = run1.parents[2].name.replace("traj_", "")
        entries.append({
            "case": sim_id,
            "trajectory_id": sim_id,
            "task_id": sim_id_to_task.get(sim_id),
            "predicted_category": int(prediction.get("failure_case")),
            "predicted_step": int(prediction.get("step_number")),
            "evidence": str(prediction.get("description"))[:300],
            "prompt_tokens": payload["summary"].get("total_prompt_tokens"),
            "output_tokens": payload["summary"].get("total_output_tokens"),
            "total_tokens": payload["summary"].get("total_tokens"),
        })

    summary = {
        "schema_version": 1,
        "source": str(Path(args.results)),
        "run_dir": args.run_dir,
        "num_failures_planned": len(failures),
        "num_diagnosed": len(entries),
        "entries": entries,
        "failed_trajectories": failed_trajs,
        "total_tokens": sum(e["total_tokens"] or 0 for e in entries),
        "note": "无人工标注：仅预测结果，用于闭环反馈；诊断准确率评测需另标。"
                " 部分轨迹诊断失败原因：AgentRx 静态检查生成代码引用未定义辅助函数。",
    }
    DIAG_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    DIAG_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"==> 诊断汇总: {DIAG_SUMMARY}  ({len(entries)} 条)")


if __name__ == "__main__":
    main()
