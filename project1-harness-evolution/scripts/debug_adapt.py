#!/usr/bin/env python
"""调试: 定位 TraceToMessages.adapt 的 KeyError('role') 坏 span（M3 两臂同崩）。

在 adapt 抛 KeyError 时 dump 所有 span 的 gen_ai.prompt 相关属性，
找出哪条消息缺 role 字段、长什么样。

用法（agent-lightning venv + DeepSeek env）:
  PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
  .venvs/agent-lightning/bin/python scripts/debug_adapt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ1))
sys.path.insert(0, str(PROJ1 / "scripts"))

import tau2_deepseek_cli  # noqa: F401,E402  # DeepSeek 模型注册

from agentlightning import Trainer  # noqa: E402
from agentlightning.adapter import TraceToMessages  # noqa: E402
from agentlightning.algorithm.apo import APO  # noqa: E402
from agentlightning.types import PromptTemplate  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from evaluation.metrics import load_results  # noqa: E402
from optimizers.tau2_rollout import set_task_pool, tau2_rollout  # noqa: E402
from resources.loader import load_resources  # noqa: E402

RESULTS_JSON = Path("/media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json")
DATASETS = PROJ1 / "data" / "datasets"


def load_split(name: str) -> list[dict]:
    path = DATASETS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


class DebugAdapter(TraceToMessages):
    """在 KeyError 处 dump 所有 span 的 prompt 属性，保留原始 trace。"""

    def adapt(self, source, /):
        try:
            return super().adapt(source)
        except KeyError:
            print("!!! KeyError('role') — 以下 span 的 gen_ai.prompt 属性:", flush=True)
            for span in source:
                pkeys = sorted(k for k in span.attributes if "gen_ai.prompt" in k)
                if pkeys:
                    print(f"-- span {span.span_id} name={getattr(span, 'name', '?')}", flush=True)
                    for k in pkeys:
                        print(f"   {k} = {repr(span.attributes[k])[:400]}", flush=True)
            raise


def main() -> None:
    model = "deepseek-v4-flash"
    dev = load_split("dev")[:1]
    val = load_split("val")[:1]
    results = load_results(RESULTS_JSON)
    set_task_pool(results["tasks"])
    v0 = load_resources(0)
    seed = PromptTemplate(template=v0["system_prompt"], engine="f-string")

    client = AsyncOpenAI(base_url="https://api.deepseek.com")
    algo = APO(
        client,
        gradient_model=model,
        apply_edit_model=model,
        val_batch_size=1,
        gradient_batch_size=1,
        beam_width=1,
        branch_factor=1,
        beam_rounds=0,  # 只跑 seed 的 val 评测，快速复现
        run_initial_validation=True,
    )
    trainer = Trainer(
        algorithm=algo,
        n_runners=1,
        initial_resources={"prompt_template": seed},
        adapter=DebugAdapter(),
    )
    print("==> fit(dev[:1], val[:1]) 开始（复现 adapt 崩溃）", flush=True)
    trainer.fit(agent=tau2_rollout, train_dataset=dev, val_dataset=val)
    print("==> fit 完成（未崩溃？）", flush=True)


if __name__ == "__main__":
    main()
