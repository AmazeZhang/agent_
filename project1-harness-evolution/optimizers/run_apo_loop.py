#!/usr/bin/env python
"""M3: APO 最小闭环 runner（SPEC 03）。

一轮完整闭环:
  dev 集 rollout（Trainer 内部）→ APO 生成候选 → 候选过滤 → best 在 val 独立重跑
  → 指标计算 → 回归门控 → 版本更新 / 拒绝记录

用法（agent-lightning venv + DeepSeek env，tmux 中运行）:
  PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
  .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
      --arm apo-diagnosis --round 1 --feedback diagnosis
  PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
  .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
      --arm apo-plain --round 1 --feedback plain

产物:
  runs/loop-apo-<arm>/round<N>.json        本轮完整记录（指标/gate 决策/版本）
  rollout_log.jsonl                        每次 rollout 的诚实记录
  resources/versions/v{N+1}/ + CHANGELOG  门控接受时
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
sys.path.insert(0, str(PROJ1))
sys.path.insert(0, str(PROJ1 / "scripts"))

import tau2_deepseek_cli  # noqa: F401,E402

from agentlightning import Trainer  # noqa: E402
from agentlightning.algorithm.apo import APO  # noqa: E402
from agentlightning.types import PromptTemplate  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from evaluation.gate import (  # noqa: E402
    GateDecision,
    evaluate_decision,
    record_rejection,
    update_version,
)
from evaluation.metrics import load_results  # noqa: E402
from optimizers.candidate_filter import CandidateFilter  # noqa: E402
from optimizers.diagnosis_to_feedback import DiagnosisAwareAdapter, load_diagnosis_summary  # noqa: E402
from resources.loader import load_resources  # noqa: E402
# 注意: optimizers.tau2_rollout 在模块加载时读取 P1_ROLLOUT_LOG 环境变量，
# 必须在设置环境变量之后再 import（见 main() 中延迟 import）。

RESULTS_JSON = Path("/media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json")
DATASETS = PROJ1 / "data" / "datasets"
PARTITION_MANIFEST = DATASETS / "partition_manifest.json"
BASELINE_VAL_RERUN = PROJ1 / "runs" / "baseline_val_rerun.json"


def load_baseline_val_rate(override: str | None = None) -> float:
    """gate 对照基准：基线在 val 8 上的实测通过率（r3 协议修正）。

    r2 用 40 任务基线的 0.900 作参照，与候选的 val 8 评测尺度不一致；
    基线 val 子集实测为 0.875（含失败任务 27）。r3 起优先读取
    scripts/run_baseline_val_rerun.py 的多数票结果，可用
    --baseline-val-rate 覆盖（不传时回退 0.9 并告警）。
    """
    if override is not None:
        return float(override)
    if BASELINE_VAL_RERUN.exists():
        rec = json.loads(BASELINE_VAL_RERUN.read_text())
        return float(rec["majority_rate"])
    print("!! 警告: runs/baseline_val_rerun.json 不存在，回退基线参照 0.9"
          "（应先运行 scripts/run_baseline_val_rerun.py）")
    return 0.9


def load_tasks() -> list:
    results = load_results(RESULTS_JSON)
    return results["tasks"]


def load_split(name: str) -> list[dict]:
    path = DATASETS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def deepseek_env() -> tuple[str, dict]:
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    return model, {
        "api_base": os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["apo-diagnosis", "apo-plain"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--feedback", default="diagnosis",
                    help="诊断注入模式（arm=apo-plain 时自动用 plain）")
    ap.add_argument("--gradient-batch-size", type=int, default=4)
    ap.add_argument("--val-batch-size", type=int, default=8)
    ap.add_argument("--n-runners", type=int, default=2)
    ap.add_argument("--beam-rounds", type=int, default=2)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--val-repeats", type=int, default=3,
                    help="val 独立重跑次数（LLM 非确定性降噪，按任务多数票计成功率）")
    ap.add_argument("--baseline-val-rate", type=float, default=None,
                    help="gate 基线参照（默认读 runs/baseline_val_rerun.json 实测值）")
    args = ap.parse_args()

    feedback = "plain" if args.arm == "apo-plain" else args.feedback
    run_dir = PROJ1 / "runs" / f"loop-apo-{args.arm}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 先设置 P1_ROLLOUT_LOG 再 import tau2_rollout（模块加载时读取该 env）
    os.environ["P1_ROLLOUT_LOG"] = str(run_dir / "rollout_log.jsonl")
    from optimizers.tau2_rollout import set_task_pool, tau2_rollout  # noqa: E402

    model, llm_kw = deepseek_env()
    dev = load_split("dev")
    val = load_split("val")
    tasks = load_tasks()
    set_task_pool(tasks)

    print(f"==> 臂: {args.arm} | 反馈: {feedback} | 轮次: {args.round}")
    print(f"==> dev={len(dev)} val={len(val)} 模型={model}")

    # ---- 适配器 ----
    if feedback == "diagnosis":
        diag_map = load_diagnosis_summary(PROJ1 / "data" / "diagnostics" / "summary.json")
        print(f"==> 加载诊断 {len(diag_map)} 条（诊断反馈注入）")
        adapter = DiagnosisAwareAdapter(diagnosis_map=diag_map, mode="diagnosis")
    else:
        print("==> plain 模式：无诊断注入（对照臂）")
        from agentlightning.adapter import TraceToMessages
        adapter = TraceToMessages()

    # ---- 资源 ----
    v0 = load_resources(0)
    seed_prompt = PromptTemplate(
        template=v0["system_prompt"],
        engine="f-string",
    )

    client = AsyncOpenAI(base_url=llm_kw["api_base"])
    algo = APO(
        client,
        gradient_model=model,
        apply_edit_model=model,
        val_batch_size=args.val_batch_size,
        gradient_batch_size=args.gradient_batch_size,
        beam_width=args.beam_width,
        branch_factor=2,
        beam_rounds=args.beam_rounds,
    )
    trainer = Trainer(
        algorithm=algo,
        n_runners=args.n_runners,
        initial_resources={"prompt_template": seed_prompt},
        adapter=adapter,
    )

    # ---- 训练（Trainer 内部在 dev 集 rollout、在 val 集 beam 评测）----
    t0 = time.time()
    trainer.fit(agent=tau2_rollout, train_dataset=dev, val_dataset=val)
    train_s = time.time() - t0

    best = algo.get_best_prompt()
    best_text = best.template
    print(f"==> APO best（内部 val 分 {algo._history_best_score:.3f}）")

    # ---- 候选过滤（写入前）----
    cf = CandidateFilter()
    ok, reasons = cf.check(best_text)
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
            "feedback": feedback,
            "train_duration_s": round(train_s, 1),
            "best_prompt": best_text,
            "best_internal_val_score": algo._history_best_score,
            "val_rerun_success_rate": None,
            "filter_reasons": reasons,
            "gate": decision.as_dict(),
            "new_version": None,
            "rollout_log": str(os.environ["P1_ROLLOUT_LOG"]),
        }
        out_path = run_dir / f"round{args.round}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        print(f"==> 记录: {out_path}（候选未通过过滤，本轮无有效候选）")
        sys.exit(1)

    # ---- best 在 val 独立重跑 ×N 多数票（SPEC 03 §4；r3 起降噪）----
    print(f"==> best 在 val（{len(val)} 任务）独立重跑 ×{args.val_repeats} ...")
    votes: dict[str, list[float]] = {str(t["id"]): [] for t in val}
    for rep in range(args.val_repeats):
        for task in val:
            votes[str(task["id"])].append(tau2_rollout(task, best))
    majority = [
        1.0 if sum(votes[str(t["id"])]) >= (args.val_repeats + 1) // 2 else 0.0
        for t in val
    ]
    val_success = sum(majority) / len(majority)
    n_win = sum(1 for v in votes.values() if sum(v) >= (args.val_repeats + 1) // 2)
    print(f"==> val 重跑 ×{args.val_repeats}（多数票）: 成功率 {val_success:.3f} ({n_win}/{len(val)})")

    # ---- 门控（基线参照 = 基线在 val8 实测值，同尺度对比）----
    baseline_rate = load_baseline_val_rate(args.baseline_val_rate)
    baseline = {"task_success_rate": baseline_rate, "total_cost_usd": 0.058}  # M1 基线（retail40-v1）
    cand = {"task_success_rate": val_success, "total_cost_usd": None}
    decision = evaluate_decision(cand, baseline)
    new_v = None
    if decision.accept:
        new_v = update_version(
            decision, {"system_prompt": best_text},
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
        "feedback": feedback,
        "train_duration_s": round(train_s, 1),
        "best_prompt": best_text,
        "best_internal_val_score": algo._history_best_score,
        "val_repeats": args.val_repeats,
        "val_rerun_success_rate": val_success,
        "val_task_majority": {tid: vs for tid, vs in votes.items()},
        "gate": decision.as_dict(),
        "new_version": new_v,
        "rollout_log": str(os.environ["P1_ROLLOUT_LOG"]),
    }
    out_path = run_dir / f"round{args.round}.json"
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"==> 记录: {out_path}")


if __name__ == "__main__":
    main()
