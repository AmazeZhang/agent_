#!/usr/bin/env python
"""M4: GEPA 适配器（SPEC 04）——tau2 retail 域接入 GEPA 进化优化。

职责:
  evaluate():              候选 system_prompt 在 batch 任务上跑 tau2 仿真 → reward 分数
  make_reflective_dataset(): 失败轨迹的反馈记录（diagnosis 模式注入 AgentRx 诊断
                            → Actionable Side Information；plain 模式仅失败事实）
  DeepSeekLM:              反思 LLM 的轻量实现（openai 兼容协议，与 APO 臂同客户端）

用法（供 run_gepa.py 导入）:
  from optimizers.gepa_adapter import Tau2GEPAAdapter, DeepSeekLM, load_tau2_batch
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import tau2.agent.llm_agent as llm_agent_mod  # noqa: E402
import tau2_deepseek_cli  # noqa: F401,E402  # DeepSeek 模型注册
from gepa.core.adapter import EvaluationBatch, GEPAAdapter  # noqa: E402
from openai import OpenAI  # noqa: E402
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.run import run_single_task  # noqa: E402

from optimizers.tau2_rollout import _TASK_POOL, make_config, set_task_pool  # noqa: E402

COMPONENT = "system_prompt"  # GEPA candidate 的组件名（与 APO 资源名一致）

# ---- 线程安全的指令注入 ----
# tau2 的 LLMAgent.system_prompt 直接读模块全局 AGENT_INSTRUCTION；
# GEPA evaluate 需要并行仿真（每任务 3-4 分钟），全局替换在多线程下竞态。
# 方案: patch LLMAgent.system_prompt 优先读 thread-local 指令，无则回退全局
# （保持 tau2_rollout 的全局替换语义不受影响）。
_instruction_local = threading.local()
_orig_system_prompt = llm_agent_mod.LLMAgent.system_prompt


def _thread_aware_system_prompt(self) -> str:
    inst = getattr(_instruction_local, "instruction", None)
    if inst is None:
        return _orig_system_prompt.fget(self)
    return llm_agent_mod.SYSTEM_PROMPT.format(
        domain_policy=self.domain_policy, agent_instruction=inst
    )


llm_agent_mod.LLMAgent.system_prompt = property(_thread_aware_system_prompt)

# ---- 诊断侧信息 ----
_DIAG_CATEGORIES = {
    0: "操作类型错误",
    1: "操作顺序错误",
    2: "操作参数错误",
    3: "操作对象错误",
    4: "先决条件未满足",
    5: "多余操作",
    6: "过早停止",
    7: "卡在循环中",
    8: "API 调用格式错误",
    9: "内部未知错误",
    10: "无失败",
}


def load_diagnosis_map(path: Path) -> dict[str, dict]:
    """加载 AgentRx 诊断汇总（summary.json）→ task_id → 诊断条目。"""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(e["task_id"]): e for e in data.get("entries", [])}


def _format_diagnosis(diag: dict) -> str:
    """诊断条目 → Actionable Side Information 字符串（注入反馈，非 ground truth）。"""
    cat = diag.get("predicted_category")
    cat_name = _DIAG_CATEGORIES.get(cat, f"类别{cat}")
    parts = [
        f"[AgentRx 诊断] 预测类别: {cat_name}({cat})",
        f"失败步数: {diag.get('predicted_step')}",
    ]
    if diag.get("evidence"):
        parts.append(f"证据: {diag['evidence']}")
    if diag.get("suggestion"):
        parts.append(f"建议: {diag['suggestion']}")
    return "\n".join(parts)


# ---- 数据实例 ----
class Tau2GEPAAdapter(GEPAAdapter[dict, dict, dict]):
    """GEPA 适配器：候选 prompt → tau2 仿真 reward；诊断注入 reflective dataset。

    DataInst:   {"id": "30"}（与 dev/val jsonl 条目一致，任务定义从任务池查）
    Trajectory: {"task_id", "reward", "termination_reason", "diagnosis"}
    RolloutOutput: {"task_id", "reward", "termination_reason"}
    """

    def __init__(self, diagnosis_path: Path | None = None,
                 inject_diagnosis: bool = True,
                 rollout_log: Path | None = None,
                 max_workers: int = 2):
        self.diag_map = load_diagnosis_map(diagnosis_path) if diagnosis_path else {}
        self.inject_diagnosis = inject_diagnosis
        self.rollout_log = rollout_log
        self.max_workers = max_workers  # evaluate 内并行仿真线程数
        self._lock = threading.Lock()
        if self.diag_map:
            print(f"==> GEPA 诊断加载 {len(self.diag_map)} 条"
                  f"（{'注入' if inject_diagnosis else '不注入'} reflective feedback）")

    # -- 记录 --
    def _record(self, task_id: str, prompt_hash: str, reward: float,
                cost: float, reason: str, duration_s: float) -> None:
        if self.rollout_log is None:
            return
        entry = {
            "ts": time.time(), "task_id": str(task_id), "prompt_hash": prompt_hash,
            "reward": reward, "cost_usd": round(cost, 6),
            "termination_reason": reason, "duration_s": round(duration_s, 1),
        }
        self.rollout_log.parent.mkdir(parents=True, exist_ok=True)
        with self.rollout_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- 核心：单任务仿真 --
    def _simulate(self, task_id: str, prompt_text: str) -> tuple[float, str, float]:
        """注入候选指令跑一次 tau2 仿真 → (reward, termination_reason, cost)。

        指令通过 thread-local 注入（见模块级 patch），多线程并行安全。
        """
        task = _TASK_POOL.get(str(task_id))
        if task is None:
            raise KeyError(f"任务池中不存在 task_id={task_id}；请先调用 set_task_pool()")
        start = time.time()
        _instruction_local.instruction = prompt_text
        try:
            run = run_single_task(make_config(task, prompt_text), task, seed=301)
        finally:
            _instruction_local.instruction = None
        rw = getattr(run.reward_info, "reward", None) if run.reward_info is not None else None
        reward = float(rw) if rw is not None else 0.0
        cost = float(getattr(run, "agent_cost", 0.0) or 0.0) + float(getattr(run, "user_cost", 0.0) or 0.0)
        reason = getattr(run, "termination_reason", "unknown") or "unknown"
        self._record(str(task_id), prompt_text[:40], reward, cost, reason, time.time() - start)
        return reward, reason, cost

    # -- GEPA 协议：evaluate --
    def evaluate(self, batch: list[dict], candidate: dict[str, str],
                 capture_traces: bool = False) -> EvaluationBatch[dict, dict]:
        prompt_text = candidate[COMPONENT]
        if self.max_workers > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                results = list(ex.map(
                    lambda data: self._simulate(str(data["id"]), prompt_text), batch))
        else:
            results = [self._simulate(str(data["id"]), prompt_text) for data in batch]

        outputs, scores, trajectories = [], [], [] if capture_traces else None
        for data, (reward, reason, _) in zip(batch, results):
            tid = str(data["id"])
            outputs.append({"task_id": tid, "reward": reward, "termination_reason": reason})
            scores.append(reward)
            if trajectories is not None:
                diag = self.diag_map.get(tid)
                trajectories.append({
                    "task_id": tid, "reward": reward,
                    "termination_reason": reason,
                    "diagnosis": diag if diag else None,
                })
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    # -- GEPA 协议：reflective dataset（ASI 注入点）--
    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict, dict],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        assert components_to_update == [COMPONENT], components_to_update
        assert eval_batch.trajectories is not None, "reflective dataset 需要 capture_traces=True"
        items: list[dict[str, str]] = []
        for traj in eval_batch.trajectories:
            tid = traj["task_id"]
            reward = traj["reward"]
            feedback = f"任务 {tid} 仿真{'成功' if reward == 1.0 else '失败'}，终止原因: {traj['termination_reason']}。"
            if reward != 1.0 and self.inject_diagnosis and traj.get("diagnosis"):
                feedback += "\n" + _format_diagnosis(traj["diagnosis"])
                feedback += "\n（注意：诊断为预测反馈，需在仿真中验证后采纳。）"
            items.append({
                "Task Input": f"tau2 retail 任务 {tid}（数据库操作类客服任务）",
                "Generated Output": f"reward={reward}，termination={traj['termination_reason']}",
                "Feedback": feedback,
            })
        if not items:
            raise ValueError("reflective dataset 为空")
        return {COMPONENT: items}


# ---- 反思 LLM ----
class DeepSeekLM:
    """GEPA LanguageModel 协议：openai 同步客户端调 DeepSeek（与 APO 臂同模型）。"""

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        )

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return resp.choices[0].message.content or ""
