#!/usr/bin/env python
"""M3: tau2 域 rollout agent（Agent Lightning Trainer 集成）。

让 agent-lightning 的 Trainer/APO 以 tau2 retail 域为执行环境做 rollout：
- task: 数据集条目（{"id", "task_input", "expected": null}，M1 dataset 适配器产物）
- prompt_template: APO 正在优化的资源（其 template 替换 tau2 LLMAgent 的 AGENT_INSTRUCTION）
- 返回值: DB 校验 reward（0.0/1.0）

实现要点:
- 通过临时替换 tau2.agent.llm_agent.AGENT_INSTRUCTION 注入候选指令
  （UserSimulator 是独立类，不受影响；domain_policy 保持任务环境原样）
- 每次 rollout 在运行目录追加一条 JSONL 记录（任务 id、候选指令 hash、reward、成本、
  终止原因）——诚实记录，供 metrics 复核
- 运行时不做 holdout 判定（由上层 dataset 入口保证）；这里提供 assert_not_holdout 辅助

用法（供 M3 runner 导入，不直接运行）:
  from optimizers.tau2_rollout import tau2_rollout, set_task_pool
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, cast

# ---- 依赖注入路径（tau2 与 DeepSeek 注册）----
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import tau2.agent.llm_agent as llm_agent_mod  # noqa: E402
import tau2_deepseek_cli  # noqa: F401,E402  # DeepSeek 模型注册 + evaluator 补丁
from agentlightning import rollout  # noqa: E402
from agentlightning.types import PromptTemplate  # noqa: E402
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.data_model.tasks import Task  # noqa: E402
from tau2.run import run_single_task  # noqa: E402

_ORIGINAL_AGENT_INSTRUCTION = llm_agent_mod.AGENT_INSTRUCTION

# 任务池：数据集适配器产物 → 完整 Task 定义（rollout 需要完整任务对象）
_TASK_POOL: dict[str, Task] = {}
_TASK_LOCK = threading.Lock()

# rollout 记录文件（每次运行进程设置一次）
ROLLOUT_LOG: Path = Path(os.environ.get(
    "P1_ROLLOUT_LOG", "runs/loop-apo/rollout_log.jsonl"
))


def set_task_pool(tasks: list[dict[str, Any]]) -> None:
    """从 dataset 适配器产物（含完整 task 定义的 tasks 列表）建立 id → Task 池。

    task_manifest 只有 id；完整定义从 results.json 的 tasks 字段读取，
    并在池中重建为 tau2 Task 对象（run_single_task 需要）。
    """
    with _TASK_LOCK:
        for t in tasks:
            tid = str(t["id"])
            if tid not in _TASK_POOL:
                _TASK_POOL[tid] = Task(**t)


def _lookup_task(tid: str) -> Task:
    with _TASK_LOCK:
        task = _TASK_POOL.get(tid)
    if task is None:
        raise KeyError(f"任务池中不存在 task_id={tid}；请先调用 set_task_pool()")
    return task


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _record(task_id: str, prompt_hash: str, reward: Optional[float],
            cost: float, reason: str, duration_s: float) -> None:
    entry = {
        "ts": time.time(),
        "task_id": str(task_id),
        "prompt_hash": prompt_hash,
        "reward": reward,
        "cost_usd": round(cost, 6),
        "termination_reason": reason,
        "duration_s": round(duration_s, 1),
    }
    ROLLOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ROLLOUT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def make_config(task: Task, prompt_text: str) -> TextRunConfig:
    """构造单任务运行配置（agent 指令 = 候选；user 保持基线配置）。"""
    llm_args = {
        "api_base": os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        "max_tokens": 8192,
        "temperature": 0.0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    model = f"openai/{os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"
    return TextRunConfig(
        domain="retail",
        agent="llm_agent",
        llm_agent=model,
        llm_args_agent=llm_args,
        user="user_simulator",
        llm_user=model,
        llm_args_user=llm_args,
        max_steps=80,
        timeout=300,
        max_retries=1,
        seed=301,
        max_concurrency=1,
        save_to=None,   # 单任务不落盘（记录由 ROLLOUT_LOG 负责）
        log_level="WARNING",
    )


@rollout
def tau2_rollout(task: dict[str, Any], prompt_template: PromptTemplate) -> float:
    """跑一次 tau2 retail 单任务仿真，返回 DB 校验 reward（0.0/1.0）。"""
    prompt_text = prompt_template.template
    prompt_hash = _prompt_hash(prompt_text)
    task_obj = _lookup_task(str(task["id"]))

    start = time.time()
    # 注入候选指令（try/finally 恢复，防止污染其他 rollout）
    llm_agent_mod.AGENT_INSTRUCTION = prompt_text
    try:
        run = run_single_task(make_config(task_obj, prompt_text), task_obj, seed=301)
    finally:
        llm_agent_mod.AGENT_INSTRUCTION = _ORIGINAL_AGENT_INSTRUCTION

    rw = getattr(run.reward_info, "reward", None) if run.reward_info is not None else None
    if rw is None:
        rw = getattr(run, "reward", None)
    reward = float(rw) if rw is not None else 0.0
    cost = float(getattr(run, "agent_cost", 0.0) or 0.0) + float(getattr(run, "user_cost", 0.0) or 0.0)
    reason = getattr(run, "termination_reason", "unknown") or "unknown"
    _record(str(task["id"]), prompt_hash, reward, cost, reason, time.time() - start)
    return reward


def assert_not_holdout(task_id: str, partition_manifest: Path) -> None:
    """holdout 守卫：任何优化入口触碰 holdout 任务直接抛错（SPEC 02 §1.2）。"""
    from data.partition import is_holdout

    if is_holdout(task_id, partition_manifest):
        raise PermissionError(
            f"holdout 任务 {task_id} 禁止进入优化调用（SPEC 02 防泄漏规则）"
        )
