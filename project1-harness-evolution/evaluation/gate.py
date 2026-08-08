#!/usr/bin/env python
"""M3: 回归门控与资源版本更新（SPEC 03 §4）。

每轮 APO 输出历史最优候选 → 在 val 集重跑得到指标 → 与基线比较 → 接受/拒绝。

判定（SPEC 02 §2.3）:
- 接受条件: val 成功率 >= 基线 - 2pp 且 成本 <= 基线成本 × 1.5
- 拒绝: 其余情况（记录拒绝原因与证据）

接受 → resources/versions/v{N+1}/ 写入资源 + CHANGELOG.md 追加记录。
拒绝 → 记录拒绝理由，不产生新版本。

诚实性: gate 判定与指标全部由 evaluation/metrics.py 计算，禁止手工改数。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJ1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ1))

from evaluation.metrics import compute_metrics, load_results  # noqa: E402
from resources.loader import save_resources  # noqa: E402


@dataclass
class GateConfig:
    success_margin_pp: float = 2.0     # val 成功率可低于基线的容忍（百分点）
    cost_multiplier: float = 1.5       # 成本可超过基线的倍数
    changelog_path: Path = PROJ1 / "resources" / "versions" / "CHANGELOG.md"


@dataclass
class GateDecision:
    accept: bool
    reason: str
    candidate_success_rate: float | None
    baseline_success_rate: float | None
    candidate_cost: float | None
    baseline_cost: float | None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accept": self.accept,
            "reason": self.reason,
            "candidate_success_rate": self.candidate_success_rate,
            "baseline_success_rate": self.baseline_success_rate,
            "candidate_cost": self.candidate_cost,
            "baseline_cost": self.baseline_cost,
            "details": self.details,
        }


def evaluate_decision(
    candidate_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    cfg: GateConfig | None = None,
) -> GateDecision:
    """根据 val 集指标做接受/拒绝决策（纯函数，可单测）。"""
    cfg = cfg or GateConfig()
    cand = candidate_metrics
    base = baseline_metrics

    c_rate = cand.get("task_success_rate")
    b_rate = base.get("task_success_rate")
    c_cost = cand.get("total_cost_usd")
    b_cost = base.get("total_cost_usd")

    if c_rate is None or b_rate is None:
        return GateDecision(False, "缺少成功率指标，拒绝", c_rate, b_rate, c_cost, b_cost)

    reasons = []
    if c_rate < b_rate - cfg.success_margin_pp / 100:
        reasons.append(f"成功率回退（{c_rate:.3f} < {b_rate - cfg.success_margin_pp/100:.3f}）")
    elif c_rate <= b_rate:
        # 语义修正（2026-08-08 r3）: 持平不得产生新版本。
        # 原实现只拒绝"明显回退"，持平（c_rate == b_rate）会判 accept，
        # 导致 best 回退到 seed 的候选也能写版本（GEPA r3 实测发生）。
        # 多次重跑多数票协议下，真实提升表现为 c_rate > b_rate；
        # 持平即"无收益"，不产生版本。
        reasons.append(
            f"未产生收益（{c_rate:.3f} ≤ 基线 {b_rate:.3f}，持平/无提升）"
        )
    if c_cost is not None and b_cost and c_cost > b_cost * cfg.cost_multiplier:
        reasons.append(
            f"成本超限（${c_cost:.4f} > 基线 ${b_cost:.4f} × ${cfg.cost_multiplier}）"
        )

    if reasons:
        return GateDecision(
            False, "; ".join(reasons), c_rate, b_rate, c_cost, b_cost,
            details={"candidate": cand, "baseline": base},
        )
    gain = (c_rate - b_rate) * 100
    return GateDecision(
        True, f"通过（成功率 +{gain:.2f}pp，成本 ${c_cost or 0:.4f}）",
        c_rate, b_rate, c_cost, b_cost,
        details={"candidate": cand, "baseline": base},
    )


def update_version(
    decision: GateDecision,
    candidate_resources: dict[str, str],
    *,
    round_id: str,
    cfg: GateConfig | None = None,
) -> int | None:
    """接受 → 写 v{N+1} 资源 + CHANGELOG 记录；拒绝 → 只记录拒绝条目。

    返回新版本号；拒绝时返回 None。
    """
    cfg = cfg or GateConfig()
    if not decision.accept:
        return None

    from resources.loader import latest_version

    new_v = latest_version() + 1
    save_resources(new_v, candidate_resources)

    entry = {
        "version": new_v,
        "round": round_id,
        "decision": "accept",
        "reason": decision.reason,
        "metrics": {
            "candidate_success_rate": decision.candidate_success_rate,
            "baseline_success_rate": decision.baseline_success_rate,
            "candidate_cost": decision.candidate_cost,
            "baseline_cost": decision.baseline_cost,
        },
    }
    _append_changelog(entry, cfg.changelog_path)
    return new_v


def record_rejection(
    decision: GateDecision,
    *,
    round_id: str,
    cfg: GateConfig | None = None,
) -> None:
    """拒绝记录（保持证据链完整）。"""
    cfg = cfg or GateConfig()
    entry = {
        "version": None,
        "round": round_id,
        "decision": "reject",
        "reason": decision.reason,
        "metrics": {
            "candidate_success_rate": decision.candidate_success_rate,
            "baseline_success_rate": decision.baseline_success_rate,
            "candidate_cost": decision.candidate_cost,
            "baseline_cost": decision.baseline_cost,
        },
    }
    _append_changelog(entry, cfg.changelog_path)


def _append_changelog(entry: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    if path.exists():
        path.write_text(path.read_text(encoding="utf-8").rstrip() + "\n" + line + "\n",
                        encoding="utf-8")
    else:
        path.write_text("# Harness 资源版本 CHANGELOG（M3 gate 自动维护）\n"
                        "格式: 每行一条 JSON 记录；版本递增 v1, v2, ...\n\n"
                        + line + "\n", encoding="utf-8")


if __name__ == "__main__":
    # 自检：决策逻辑（r3 语义: 持平拒绝）
    base = {"task_success_rate": 0.85, "total_cost_usd": 0.50}
    good = {"task_success_rate": 0.90, "total_cost_usd": 0.55}
    tie = {"task_success_rate": 0.85, "total_cost_usd": 0.55}
    bad = {"task_success_rate": 0.80, "total_cost_usd": 0.51}
    expensive = {"task_success_rate": 0.95, "total_cost_usd": 1.00}
    for name, cand in [("good", good), ("tie", tie), ("bad", bad), ("expensive", expensive)]:
        d = evaluate_decision(cand, base)
        print(f"{name}: accept={d.accept}  reason={d.reason}")
    assert evaluate_decision(good, base).accept
    assert not evaluate_decision(tie, base).accept, "持平必须拒绝（无收益）"
    assert not evaluate_decision(bad, base).accept
    assert not evaluate_decision(expensive, base).accept
    print("门控自检通过")
