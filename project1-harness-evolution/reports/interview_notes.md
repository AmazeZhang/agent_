# 项目一面试叙事（agent 算法岗）

- 日期：2026-08-07（数字待四臂完成后回填）
- 定位：秋招 agent 算法岗的项目叙事——问题定义、关键模块、消融、失败分析

## 1. 一句话

用 trace 驱动 agent 自进化：在 tau2-bench 仿真环境里采集失败轨迹，AgentRx 做失败诊断，
APO / GEPA 两个优化器把诊断变成提示词改进，门控接受，形成"执行→诊断→改进→评测"闭环。

## 2. 问题定义

- 为什么需要自进化：静态提示词工程上限低；agent 系统需要从自己的失败轨迹中学习
- 核心问题：**失败轨迹 → 可操作的改进信号**的转换
- 技术栈选择：Agent Lightning（APO，prompt 优化）+ AgentRx（trace 失败诊断）+ GEPA（进化式 prompt 优化）+
  tau2-bench（零售客服多轮对话仿真，40 任务基线）
- 本项目的独特点：把 AgentRx 的**失败诊断**（类别/失败步/证据）做成 Agent Lightning 和 GEPA 的**可操作反馈信号**，
  与"原始轨迹直接当反馈"（plain）对比——这是消融设计的核心问题

## 3. 数据与实验设计

- tau2-bench retail 40 任务 → 60/20/20 划分（dev/val/holdout），holdout 全程隔离
- 消融矩阵 5 臂：
  baseline（无优化） / apo-plain / apo-diagnosis / gepa-plain / gepa-diagnosis
- 每臂同模型（deepseek-v4-flash）、同任务、同评测协议、固定预算

## 4. 关键模块

### 4.1 Trace 适配层（本项目修过的核心）
- Agent Lightning 的 TraceToMessages：span trace → OpenAI 对话消息
- **修了三个上游插桩缺陷**（AgentOps 0.4.21）：
  1. 带 tool_calls 的 assistant 消息漏记 role → 推断修复
  2. tool 消息（tool_call_id）漏记 role → 推断修复
  3. 多 tool_calls 的最后一个 call 字段缺失（id/name/arguments）→ 兜底修复
- 面试点：如何定位（debug adapter dump 全部 span 属性，5 轮验证）、如何修复（适配层修，不动插桩库）

### 4.2 诊断注入
- AgentRx 诊断摘要（10 类失败类别）→ 注入两种形态：
  - APO 臂：诊断感知适配器，把诊断放进 LLM 优化上下文
  - GEPA 臂：诊断作为 Actionable Side Information 进 reflective feedback
- 关键设计：诊断是**预测性反馈**（"建议在仿真中验证后采纳"），不是 ground truth——诚实性设计

### 4.3 线程安全指令注入（GEPA）
- tau2 的 AGENT_INSTRUCTION 是模块全局，GEPA 多线程并行仿真有竞态
- 方案：patch LLMAgent.system_prompt 读 thread-local，无值时回退全局

### 4.4 回归门控
- 候选提示词在 val 独立重跑 → 成功率/成本 vs 基线（90% / $0.058）→ 接受则版本更新+CHANGELOG，拒绝则记录

## 5. 结果（待回填）

| 臂 | val 成功率 | 成本 | gate | 版本 |
|---|---|---|---|---|
| baseline | 0.90 | $0.058 | — | v1 |
| apo-plain | ? | ? | ? | ? |
| apo-diagnosis | ? | ? | ? | ? |
| gepa-plain | ? | ? | ? | ? |
| gepa-diagnosis | ? | ? | ? | ? |

## 6. 失败分析与诚实结论

- 样本量小（40 任务），不做统计显著性宣称
- 若未提升：结论 = "闭环工程可行但方法尚未产生收益"（DEVELOPMENT_SCOPE 2.3 原话）
- 工程层最大的失败与修复：AgentOps 插桩缺陷导致的两臂同时崩溃 → 调试方法论是亮点

## 7. 面试高频问答备选

- 为什么用 tau2 不用 SWE-bench？（多轮对话 + 失败样本可控；SWE-bench 在项目二）
- 诊断反馈 vs 原始轨迹，哪个更好？怎么证明？（消融矩阵就是回答）
- 你的贡献和上游的边界？（适配层修复 vs 自研适配器/门控/线程注入）
- 下一步？（更多样本、多资源联合优化、诊断质量评估）
