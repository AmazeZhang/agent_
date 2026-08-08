#!/usr/bin/env python
"""M4: GEPA 闭环 runner（SPEC 04）——进化式候选生成 + 验证集门控。

一轮完整闭环:
  GEPA 进化（dev minibatch 反思 + val 全量评测跟踪 pareto）→ best candidate
  → 候选过滤 → best 在 val 独立重跑 → 指标计算 → 回归门控 → 版本更新 / 拒绝记录

用法（agent-lightning venv + DeepSeek env，tmux 中运行）:
  PYTHONPATH=vendor/tau2-bench/src \
  .venvs/agent-lightning/bin/python optimizers/run_gepa.py \
      --arm gepa-diagnosis --round 1
  .venvs/agent-lightning/bin/python optimizers/run_gepa.py \
      --arm gepa-plain --round 1

产物:
  runs/loop-gepa-<arm>/round<N>.json      本轮完整记录（指标/gate 决策/版本）
  runs/loop-gepa-<arm>/rollout_log.jsonl  GEPA evaluate 每次仿真的诚实记录
  runs/loop-gepa-<arm>/state/              GEPA 断点续跑状态（run_dir）
  resources/versions/v{N+1}/ + CHANGELOG  门控接受时
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import tau2_deepseek_cli  # noqa: F401,E402  # DeepSeek 模型注册

from gepa import optimize  # noqa: E402

from evaluation.gate import (  # noqa: E402
    GateDecision,
    evaluate_decision,
    record_rejection,
    update_version,
)
from evaluation.metrics import load_results  # noqa: E402
from optimizers.candidate_filter import CandidateFilter  # noqa: E402
from optimizers.gepa_adapter import COMPONENT, DeepSeekLM, Tau2GEPAAdapter  # noqa: E402
from optimizers.tau2_rollout import set_task_pool, tau2_rollout  # noqa: E402
from resources.loader import load_resources  # noqa: E402

RESULTS_JSON = Path("/media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json")
DATASETS = SCRIPT_DIR.parent / "data" / "datasets"
DIAGNOSIS_SUMMARY = SCRIPT_DIR.parent / "data" / "diagnostics" / "summary.json"


def load_split(name: str) -> list[dict]:
    path = DATASETS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


class _PromptText:
    """轻量 PromptTemplate 鸭子类型（tau2_rollout 只访问 .template）。"""

    def __init__(self, template: str):
        self.template = template


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["gepa-diagnosis", "gepa-plain"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--max-metric-calls", type=int, default=40,
                    help="GEPA 总评测预算（一次 metric call = 一次仿真）")
    ap.add_argument("--max-workers", type=int, default=2, help="evaluate 并行仿真线程数")
    ap.add_argument("--seed", type=int, default=301)
    args = ap.parse_args()

    inject_diagnosis = args.arm == "gepa-diagnosis"
    run_dir = SCRIPT_DIR.parent / "runs" / f"loop-gepa-{args.arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rollout_log = run_dir / "rollout_log.jsonl"

    dev = load_split("dev")
    val = load_split("val")
    results = load_results(RESULTS_JSON)
    set_task_pool(results["tasks"])

    v0 = load_resources(0)
    seed_candidate = {COMPONENT: v0["system_prompt"]}

    adapter = Tau2GEPAAdapter(
        diagnosis_path=DIAGNOSIS_SUMMARY if inject_diagnosis else None,
        inject_diagnosis=inject_diagnosis,
        rollout_log=rollout_log,
        max_workers=args.max_workers,
    )

    print(f"==> 臂: {args.arm} | 轮次: {args.round} | dev={len(dev)} val={len(val)}"
          f" | metric 预算={args.max_metric_calls} | workers={args.max_workers}")

    # ---- GEPA 进化（同步阻塞，直到预算耗尽或停止条件）----
    # reflection_lm 传裸 LanguageModel（DeepSeekLM）——GEPA 内部会用
    # StatelessReflectionLM 包装；若传包装后的实例会被二次包装导致
    # "'StatelessReflectionLM' object is not callable"（2026-08-08 修复）。
    # skip_perfect_score=False: minibatch 全对也继续反思（不跳过迭代）。
    t0 = time.time()
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=dev,
        valset=val,
        adapter=adapter,
        reflection_lm=DeepSeekLM(),
        skip_perfect_score=False,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(run_dir / "state"),
        seed=args.seed,
        track_best_outputs=False,
        display_progress_bar=False,
    )
    train_s = time.time() - t0

    best = result.best_candidate[COMPONENT]
    gepa_val_score = result.val_aggregate_scores[result.best_idx]
    print(f"==> GEPA best（内部 val 均分 {gepa_val_score:.3f}，"
          f"候选 {result.num_candidates} 个，metric calls {result.total_metric_calls}，"
          f"full val evals {result.num_full_val_evals}，耗时 {train_s:.0f}s）")

    # ---- 候选过滤（写入前）----
    cf = CandidateFilter()
    ok, reasons = cf.check(best)
    if not ok:
        # 过滤失败: 记录拒绝 + 完整 round 记录（含 best 文本），不产生版本
        decision = GateDecision(
            accept=False,
            reason=f"候选未通过过滤: {'; '.join(reasons)}",
            candidate_success_rate=None,
            baseline_success_rate=0.9,
            candidate_cost=None,
            baseline_cost=None,
        )
        record_rejection(decision, round_id=f"{args.arm}-r{args.round}")
        record = {
            "schema_version": 1,
            "arm": args.arm,
            "round": args.round,
            "train_duration_s": round(train_s, 1),
            "max_metric_calls": args.max_metric_calls,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
            "num_full_val_evals": result.num_full_val_evals,
            "best_prompt": best,
            "best_internal_val_score": gepa_val_score,
            "val_rerun_success_rate": None,
            "filter_reasons": reasons,
            "gate": decision.as_dict(),
            "new_version": None,
            "rollout_log": str(rollout_log),
            "run_dir_state": str(run_dir / "state"),
        }
        out_path = run_dir / f"round{args.round}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"==> 记录: {out_path}（候选未通过过滤，本轮无有效候选）")
        sys.exit(1)

    # ---- best 在 val 独立重跑（与 APO 臂一致，串行 tau2_rollout）----
    print(f"==> best 在 val（{len(val)} 任务）独立重跑 ...")
    rewards = []
    for task in val:
        rewards.append(tau2_rollout(task, _PromptText(best)))
    val_success = sum(1 for r in rewards if r == 1.0) / len(rewards)
    print(f"==> val 重跑: 成功率 {val_success:.3f} ({sum(1 for r in rewards if r == 1.0)}/{len(rewards)})")

    # ---- 门控 ----
    baseline = {"task_success_rate": 0.9, "total_cost_usd": 0.058}  # M1 基线（retail40-v1）
    cand = {"task_success_rate": val_success, "total_cost_usd": None}
    decision = evaluate_decision(cand, baseline)
    new_v = None
    if decision.accept:
        new_v = update_version(
            decision, {"system_prompt": best},
            round_id=f"{args.arm}-r{args.round}",
        )
        print(f"==> 接受 → 版本 v{new_v}")
    else:
        record_rejection(decision, round_id=f"{args.arm}-r{args.round}")
        print(f"==> 拒绝: {decision.reason}")

    # ---- 本轮记录 ----
    record = {
        "schema_version": 1,
        "arm": args.arm,
        "round": args.round,
        "train_duration_s": round(train_s, 1),
        "max_metric_calls": args.max_metric_calls,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "best_prompt": best,
        "best_internal_val_score": gepa_val_score,
        "val_rerun_success_rate": val_success,
        "gate": decision.as_dict(),
        "new_version": new_v,
        "rollout_log": str(rollout_log),
        "run_dir_state": str(run_dir / "state"),
    }
    out_path = run_dir / f"round{args.round}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"==> 记录: {out_path}")


if __name__ == "__main__":
    main()
