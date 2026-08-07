# SPEC 01 — M0/M1：Agent Lightning 接入与 tau2 基线

- 版本：v0.1（草稿）
- 日期：2026-08-07
- 前置：SPEC 00 评审通过

## 1. M0 目标：验证 Agent Lightning + DeepSeek 可用

### 1.1 现状核查（已完成）

- vendor 版本：`f0a77cf`（2026-07-16），含 `agentlightning/algorithm/apo/`（apo.py + prompts/）与完整示例 `examples/apo/`、`examples/minimal/`。
- 示例依赖：OpenAI-compatible API（`AsyncOpenAI`）——DeepSeek 满足。
- 环境：`.venvs/agent-lightning/bin/activate` 存在，README 记录基础导入通过。

### 1.2 实施步骤

1. 激活 `.venvs/agent-lightning`，安装示例依赖（如缺）：参考 `examples/apo/README.md` 与 `docs/tutorials/installation.md`（agentlightning[apo] 附加依赖）。
2. 在 tmux 中运行官方最小示例 `examples/minimal/`，确认 rollout + trace 落盘。
3. 复制 `examples/apo/` 到 `project1-harness-evolution/optimizers/sandbox/`，把 `AsyncOpenAI()` 改为 DeepSeek endpoint（`base_url=https://api.deepseek.com`，key 从 `.secrets/deepseek.env` 读取，不写死）。
4. 运行 `room_selector_apo.py`，确认 APO 完成至少一轮（gradient → candidate → val 评测），日志落盘到 `runs/loop-sandbox/`。
5. 记录：模型、token 消耗、成本、每轮结果。

### 1.3 M0 验收

- [ ] 官方 APO 示例在 DeepSeek 上完成 ≥1 轮完整训练（beam search 有输出，val 指标更新）
- [ ] 产物目录 `runs/loop-sandbox/` 有 trainer 日志、算法日志、最终 resource（优化后 prompt）
- [ ] 无 GPU 参与（全程 CPU + API）
- [ ] 发现的环境/API 兼容问题记录到 `docs/`（如 `troubleshooting`）

## 2. M1 目标：tau2 任务接入 Agent Lightning + 基线

### 2.1 设计：三个适配层

```text
层 1  dataset 适配：tau2 task 描述 → Agent Lightning Dataset[Task]
层 2  agent 函数：Task → 执行（DeepSeek function calling，tau2 工具）→ 结果
层 3  reward：结果 → 标量 reward（DB 状态校验 + 任务成功判定）
```

参考官方示例 `room_selector.py` 的形态：`task_input` 入、`expected_choice` 出，用 **agent 函数 + Trace** 表达。tau2 自有 runner 已有完整工具环（`tau2/tools`），可选方案：

- **方案 A（推荐）**：agent 函数内封装 tau2 的模拟执行（复用 `tau2` 的 `--agent-llm` 链路），把 tau2 执行结果包成 Agent Lightning Task/Trace。改动最小、reward 直接用 tau2 的 DB 校验。
- **方案 B**：在 Agent Lightning 内重写 tau2 工具环（工作量大，M1 不做，留作 GEPA 阶段评估）。

### 2.2 tau2 任务格式 → Agent Lightning Dataset

- tau2 侧：任务定义 + `simulations[]`（含完整轨迹、reward 明细）。
- Agent Lightning 侧：jsonl，字段 `{"id": ..., "task_input": {...}, "expected": ...}`（参考 `room_tasks.jsonl`）。
- 交付：`data/datasets/tau2_retail.jsonl` 生成脚本 `data/build_agent_lightning_dataset.py`，保持 tau2 任务 id 可回溯（`task_id ↔ simulation id` 映射表落盘）。

### 2.3 reward 设计（M1 版本）

- `reward = 1` 当且仅当 tau2 DB 校验通过（沿用现有 Reward 1.0 语义）；否则 0。
- 记录结构化：reward、token、成本、轨迹文件路径、诊断结果（若已跑 AgentRx）。
- 多目标扩展（成本、副作用）在 M2 定义，M1 先只做成功率主 reward。

### 2.4 采集规模

- 目标 40–80 条自然任务（现有 20 条 retail + 新增）；每条约几美分。
- 从 `tau2` 可重复生成的 retail 任务池采样，固定 seed，保证可复现。
- 全部轨迹落 `/media/imc/data/yzy/agent/project1/`（大文件）或 `data/runs/`（小文件）。
- 采集完成前，`data/partition.py`（M2）定稿并锁定划分（防泄漏）。

### 2.5 基线产物

- `runs/loop-baseline/`：每条任务 {任务 id, reward, token, cost, 轨迹, 诊断结果}
- `data/baseline_summary.json`：成功率、平均成本、失败样本清单（失败样本 = 闭环的候选输入）
- 失败样本进入 AgentRx 六阶段诊断，产出 `data/diagnostics/`（结构化：类别、失败步骤、证据、建议）

### 2.6 M1 验收

- [ ] `build_agent_lightning_dataset.py` 生成 dataset，id 可回溯 tau2
- [ ] Agent Lightning runner 在 ≥40 条任务上完成 rollout，reward 与 tau2 原生结果一致（抽查 ≥3 条人工核对）
- [ ] 失败样本全部跑通 AgentRx 诊断，结果结构化落盘
- [ ] 基线 summary 可复现（固定 seed、固定模型）

## 3. 与调研报告的对应

- 调研报告 2.3 第 1 项"tau2 task 与 Agent Lightning dataset 的适配" → 本 spec 2.1/2.2
- 调研报告 4.1"第一阶段最小闭环和算法基线" → M0 + M1
