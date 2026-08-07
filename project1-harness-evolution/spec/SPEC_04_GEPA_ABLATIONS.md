# SPEC 04 — M4/M5：GEPA 接入与消融、最终报告

- 版本：v0.1（草稿）
- 日期：2026-08-07
- 前置：M3 闭环至少一轮完成且对照臂数字落盘

## 1. M4 目标：GEPA 接入（调研报告 4.2 的第二阶段高级优化器）

### 1.1 GEPA 与 AgentRx 的对接（报告 2.2 的互补关系）

```text
AgentRx 回答"为什么失败"（类别/步骤/证据）
        ↓ 适配器
GEPA 的 Actionable Side Information（诊断性文本反馈）
        ↓
反思 → 定向变异 → 评测新候选 → Pareto 前沿更新
```

- GEPA 官方安装：`pip install gepa`（独立 venv 或 agent-tools venv，评估后定）。
- 独立 API 可嵌入：GEPA 不强制重写 Agent，通过 evaluator/adapter 接入我们的 rollout（tau2 执行链）。
- 优化对象扩展到多资源联合：`system_prompt` + `tool_policy` + `action_strategy`（M2 已定义序列化）。

### 1.2 交付物

- `optimizers/gepa_adapter.py`：把 AgentRx 诊断 JSON → Actionable Side Information 字符串。
- `optimizers/gepa_evaluator.py`：在 dev/val 上运行候选的 rollout 评测接口（成本守卫复用 M2）。
- `optimizers/run_gepa.py`：入口，输出 Pareto 前沿候选 + 每候选 val 指标。

### 1.3 GEPA 与 APO 的关系

- 不重复实现 APO：GEPA 作为第二阶段独立优化器接入，做**方法对比**而非替换。
- 若 GEPA 与 Agent Lightning 集成成本过高（如 API 形态不兼容），备选：`dspy.GEPA`（DSPy 集成版，调研报告 4.3 已确认支持 GEPA）。

## 2. 消融矩阵（M5 核心产出）

### 2.1 设计

| 臂 | 优化器 | 反馈信号 | 优化对象 | 目的 |
|---|---|---|---|---|
| baseline | 无 | 无 | 无 | 零点 |
| apo-plain | APO | 原始轨迹 | system_prompt | 隔离诊断贡献 |
| apo-diagnosis | APO | AgentRx 诊断 | system_prompt | **本项目方案 A** |
| gepa-diagnosis | GEPA | AgentRx 诊断（ASI） | 三资源联合 | **本项目方案 B** |
| gepa-plain（可选） | GEPA | 原始轨迹 | 三资源联合 | GEPA 自身诊断增益 |

### 2.2 统一协议

- 同一模型（deepseek-v4-flash）、同一 dev/val/holdout 划分、同一 seed。
- 每臂固定预算（token/成本上限一致），避免"烧钱取胜"。
- 全部在 holdout 集上做最终同协议评测（M3 各臂在 val 上的决策不受 holdout 影响）。

### 2.3 输出

- `reports/ablation_2026-08-xx.md`：成功率、成本、副作用三列并排 + 置信说明（样本量小，不做统计显著性宣称）。
- 每个数字可追溯：`runs/loop-<臂名>/` 原始文件 → `evaluation/metrics.py` 重算。
- 诚实结论模板：提升/持平/回退均如实；若未提升，结论 = "闭环工程可行但方法尚未产生收益"（DEVELOPMENT_SCOPE 2.3 原话）。

## 3. M5 最终交付清单

| 交付物 | 位置 |
|---|---|
| spec 全部评审闭环记录 | `spec/` |
| 复现脚本（一键跑通 M1→M4 主链路） | `scripts/run_loop.sh`（tmux 包装） |
| 消融报告 | `reports/ablation_*.md` |
| 面试叙事（问题定义、关键模块、消融、失败分析） | `reports/interview_notes.md` |
| DEVELOPMENT_SCOPE 2.3 逐项对照表 | `reports/acceptance_checklist.md` |

## 4. M4/M5 验收

- [ ] GEPA 至少完成一轮优化，产出 Pareto 候选 + val 指标
- [ ] 消融矩阵 ≥4 臂（baseline/apo-plain/apo-diagnosis/gepa-diagnosis）全有数字
- [ ] holdout 全程未触碰（日志可证）
- [ ] 一键复现脚本在干净环境说明下可运行
- [ ] 验收清单逐项勾选，未达成项如实标注

## 5. 风险

| 风险 | 对策 |
|---|---|
| GEPA 与 DeepSeek/现有链集成失败 | 备选 dspy.GEPA；再备选退回 APO 单优化器但保留诊断反馈差异 |
| 多资源联合优化导致评测爆炸（3 资源 × 候选数） | 首轮 GEPA 先只优化 system_prompt，联合优化留到有预算后 |
| 40–80 条样本消融结果波动大 | 固定 seed 多次采样报 Pass@1 区间，报告明确样本量限制 |
