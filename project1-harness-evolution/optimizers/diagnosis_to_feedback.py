#!/usr/bin/env python
"""M3: AgentRx 诊断 → APO 文本反馈适配器（SPEC 03 §2）。

把 AgentRx judge 的诊断结果转换为 APO gradient 提示中的注入段落。

两种模式（供 M5 消融）:
- feedback=diagnosis: 在轨迹消息后追加 [DIAGNOSIS] 系统段（本项目方案）
- feedback=plain:    原样返回（纯 APO 对照臂）

用法（集成层）:
  diag_map = load_diagnosis_summary("data/diagnostics/summary.json")  # task_id -> DiagnosisFeedback
  adapter = DiagnosisAwareAdapter(diagnosis_map=diag_map, mode="diagnosis",
                                  task_id_extractor=my_extractor)
  trainer = Trainer(..., adapter=adapter)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from agentlightning.adapter.messages import TraceToMessages
from agentlightning.adapter.messages import OpenAIMessages
from agentlightning.types import Span


@dataclass
class DiagnosisFeedback:
    """单个失败轨迹的诊断（预测结果，无人工标注）。

    字段对应 AgentRx judge 输出与 scripts/diagnose_baseline_failures.py 的汇总。
    """

    task_id: str
    category: int          # AgentRx 10 类失败类别（1-10）
    step: int              # 失败发生步骤号
    evidence: str          # 证据摘要（violation 日志）
    suggestion: Optional[str] = None  # 修复建议（judge 生成，若有）
    trajectory_id: Optional[str] = None


def load_diagnosis_summary(path: str | Path) -> dict[str, DiagnosisFeedback]:
    """解析 data/diagnostics/summary.json → {task_id: DiagnosisFeedback}。

    同一任务多条诊断时保留第一条（诊断按轨迹给出，任务级去重保持确定性）。
    """
    payload = json.loads(Path(path).read_text())
    out: dict[str, DiagnosisFeedback] = {}
    for e in payload.get("entries", []):
        tid = str(e["task_id"])
        if tid in out:
            continue
        out[tid] = DiagnosisFeedback(
            task_id=tid,
            category=int(e["predicted_category"]),
            step=int(e["predicted_step"]),
            evidence=str(e.get("evidence") or "")[:600],
            suggestion=e.get("suggestion"),
            trajectory_id=e.get("trajectory_id"),
        )
    return out


def format_diagnosis_block(diag: DiagnosisFeedback, max_evidence_chars: int = 500) -> str:
    """把诊断格式化为注入段（SPEC 03 §2.2 的 [DIAGNOSIS] 块）。"""
    evidence = (diag.evidence or "").strip()[:max_evidence_chars]
    lines = [
        "[DIAGNOSIS] 该轨迹失败的自动诊断结果（供改进系统提示使用）：",
        f"类别: {diag.category}",
        f"失败步骤: {diag.step}",
        f"证据: {evidence}",
    ]
    if diag.suggestion:
        lines.append(f"建议: {diag.suggestion[:300]}")
    return "\n".join(lines)


def inject_diagnosis(
    messages: List[OpenAIMessages],
    diag: Optional[DiagnosisFeedback],
    mode: str = "diagnosis",
    max_evidence_chars: int = 500,
) -> List[OpenAIMessages]:
    """注入诊断段；mode=plain 或 diag 为空时原样返回。

    OpenAIMessages 是 pydantic 模型，追加新实例不修改原对象。
    """
    if mode != "diagnosis" or diag is None:
        return messages
    block = format_diagnosis_block(diag, max_evidence_chars)
    diag_msg: OpenAIMessages = {
        "messages": [{"role": "system", "content": block}],
    }
    return [*messages, diag_msg]


def extract_rollout_id_from_spans(spans: Sequence[Span]) -> Optional[str]:
    """从 span 集合提取 rollout_id（agent-lightning 在 span attribute 中记录）。"""
    for span in spans:
        rid = (span.attributes or {}).get("lightning.rollout_id")
        if rid:
            return str(rid)
    return None


class DiagnosisAwareAdapter(TraceToMessages):
    """包装官方 TraceToMessages：消息拼装后按模式注入诊断段。

    diagnosis_map 键与 key_fn 的输出对齐（默认 rollout_id）。
    M3 集成时若以任务为单位关联诊断，传入相应的 key_fn 即可。
    """

    def __init__(
        self,
        diagnosis_map: dict[str, DiagnosisFeedback] | None = None,
        mode: str = "diagnosis",
        max_evidence_chars: int = 500,
        key_fn: Callable[[Sequence[Span]], Optional[str]] = extract_rollout_id_from_spans,
    ) -> None:
        super().__init__()
        if mode not in ("diagnosis", "plain"):
            raise ValueError(f"mode 必须是 diagnosis/plain，得到 {mode}")
        self.diagnosis_map = diagnosis_map or {}
        self.mode = mode
        self.max_evidence_chars = max_evidence_chars
        self.key_fn = key_fn

    def adapt(self, source: Sequence[Span]) -> List[OpenAIMessages]:
        messages = super().adapt(source)
        if self.mode == "plain":
            return messages
        key = self.key_fn(source)
        diag = self.diagnosis_map.get(key) if key else None
        return inject_diagnosis(messages, diag, "diagnosis", self.max_evidence_chars)


if __name__ == "__main__":
    # 自检：直接运行验证两种模式
    import sys

    demo = DiagnosisFeedback(
        task_id="7", category=6, step=12,
        evidence="agent 在查询订单详情时使用了错误的 order_id，DB 校验不匹配。",
        suggestion="要求先列出可见订单再选择。",
    )
    base = [{"messages": [{"role": "user", "content": "任务"}]}]
    plain = inject_diagnosis(base, demo, "plain")
    diag = inject_diagnosis(base, demo, "diagnosis")
    print(f"plain: {len(plain)} 条消息, 无 DIAGNOSIS: {'[DIAGNOSIS]' not in str(plain)}")
    print(f"diagnosis: {len(diag)} 条消息, 含 DIAGNOSIS: {'[DIAGNOSIS]' in str(diag)}")
    print("---")
    print(diag[-1]["messages"][0]["content"])
    sys.exit(0 if ('[DIAGNOSIS]' in str(diag) and '[DIAGNOSIS]' not in str(plain)) else 1)
