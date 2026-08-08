# 项目一：Agent Harness 自进化开源项目调研报告

> 调研日期：2026-08-07  
> 调研目标：寻找具有较高 GitHub 认可度、能够真实部署运行，并可用于补全项目一“诊断—优化—验证—版本更新”闭环的开源项目。

## 1. 调研背景

当前项目一主要基于以下开源项目：

- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)：Agent 轨迹采集、任务运行、资源管理与训练基础设施；
- [Microsoft AgentRx](https://github.com/microsoft/AgentRx)：Agent 失败轨迹分析与根因诊断；
- [Sierra Research tau2-bench](https://github.com/sierra-research/tau2-bench)：面向工具调用 Agent 的任务环境与评测基准。

目前已经完成或基本完成：

- DeepSeek 接入 AgentRx；
- tau2 轨迹转换；
- 六阶段失败诊断流程；
- 真实失败样本与 few-shot 数据构建；
- 严格的失败类别、失败步骤和汇总评测；
- 多仓库隔离环境、配置、运行脚本和实验报告。

但项目当前主体仍停留在：

```text
Agent 运行或读取已有轨迹
→ AgentRx 失败诊断
→ 诊断结果评测
```

尚未完全形成：

```text
Agent 执行
→ 发现失败
→ 诊断根因
→ 生成候选修改
→ 在验证集重新运行
→ 检查收益与回归
→ 接受或拒绝候选
→ 更新 Harness 版本
```

因此，本轮调研重点不是继续寻找另一个“失败分析工具”，而是寻找可以承担以下能力的项目：

1. 自动生成 Prompt、工具描述或工作流候选；
2. 利用执行轨迹和文本反馈指导优化；
3. 在任务集上自动运行、比较和选择候选；
4. 支持 API 模型，尤其是 DeepSeek 或 OpenAI-compatible API；
5. 可以与 Agent Lightning、AgentRx 和 tau2-bench 二次集成；
6. 能产生可复现、可量化、适合面试展示的实验结果。

## 2. 核心结论

本轮调研得到三个关键结论。

### 2.1 不需要从零开发完整自进化算法

Agent Lightning 的新版本已经内置 APO（Automatic Prompt Optimization）算法，实现了：

```text
Rollout
→ Trace 与 reward
→ 文本梯度/失败批评
→ Prompt 改写
→ 验证集评测
→ Beam Search 选择
→ 更新 Prompt 资源
```

这意味着项目一可以先使用 Agent Lightning 自带 APO 补齐最小优化闭环，而不必自己重新实现整套 Prompt 搜索算法。

### 2.2 GEPA 是与项目一最匹配的外部优化器

GEPA 能读取完整执行轨迹、错误信息和文本诊断，通过反思、变异和 Pareto 选择优化 Prompt、工具定义、代码、配置和 Agent 架构等文本资源。

它和 AgentRx 的互补关系非常清晰：

- AgentRx 回答“为什么失败”；
- GEPA 根据失败原因回答“应该怎样修改”；
- tau2-bench 回答“修改后是否真的更好”；
- Agent Lightning 负责运行、追踪、资源版本和调度。

### 2.3 仍然需要完成有价值的自研开发

成熟优化器不能直接理解当前 tau2 数据、AgentRx 诊断格式和业务约束。以下部分仍然需要自行开发：

- tau2 task 与 Agent Lightning dataset 的适配；
- AgentRx 诊断到优化器反馈的结构化转换；
- 可优化 Harness 资源的定义；
- reward、多目标指标与约束设计；
- 非法候选过滤；
- 训练集、验证集和测试集隔离；
- 回归门控和版本发布机制；
- APO、GEPA、无诊断版本等消融实验。

这些部分不是简单“胶水代码”，而是项目最有独立贡献和面试价值的部分。

## 3. 候选项目总览

以下 GitHub 数据采集于 2026-08-07，Star 数会随时间变化。

| 项目 | GitHub Star | 最近推送 | 核心能力 | 部署难度 | 与项目一适配度 | 建议定位 |
|---|---:|---|---|---:|---:|---|
| [Agent Lightning](https://github.com/microsoft/agent-lightning) | 17,457 | 2026-07-16 | Trace、资源管理、APO、SFT、RL | 中 | 10/10 | 核心运行框架 |
| [GEPA](https://github.com/gepa-ai/gepa) | 6,019 | 2026-08-06 | 反思、变异、Pareto 搜索、任意文本资源优化 | 低—中 | 10/10 | 核心高级优化器 |
| [DSPy](https://github.com/stanfordnlp/dspy) | 36,676 | 2026-08-07 | GEPA、MIPROv2、SIMBA、LM 程序优化 | 中 | 8/10 | 可选算法框架 |
| [Opik](https://github.com/comet-ml/opik) | 21,175 | 2026-08-07 | Trace、评测、Dashboard、Agent Optimizer | 中—高 | 8/10 | 可选实验平台 |
| [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) | 3,211 | 2026-07-07 | AFlow、TextGrad、MIPRO、工作流生成 | 中 | 7/10 | 对照框架 |
| [PromptWizard](https://github.com/microsoft/PromptWizard) | 3,999 | 2025-10-14 | Prompt 与 few-shot 示例联合优化 | 低 | 6/10 | Prompt 基线 |
| [TextGrad](https://github.com/zou-group/textgrad) | 3,690 | 2025-07-25 | 文本梯度和 Prompt 优化 | 低 | 6/10 | 算法基线 |
| [MetaGPT/AFlow](https://github.com/FoundationAgents/MetaGPT) | 69,693 | 2026-01-21 | MCTS 搜索 Agent 工作流结构 | 高 | 6/10 | 结构搜索参考 |
| [Promptfoo](https://github.com/promptfoo/promptfoo) | 24,035 | 2026-08-07 | Agent 测试、回归、红队与 CI | 低 | 7/10 | 评测辅助工具 |
| [AgentEvolver](https://github.com/modelscope/AgentEvolver) | 1,520 | 2026-04-01 | 任务生成、探索、归因、策略训练 | 高 | 5/10 | 更偏 Agentic RL |
| [AgentJet](https://github.com/modelscope/AgentJet) | 233 | 2026-08-06 | 分布式 Agent RL 训练 | 高 | 4/10 | 更偏项目二 |
| [Meta Prompt Ops](https://github.com/meta-llama/prompt-ops) | 852 | 2026-04-22 | Llama Prompt 迁移和自动优化 | 低 | 5/10 | 模型迁移参考 |

需要注意：MetaGPT 的约 7 万 Star 属于整个 MetaGPT 仓库，并不等同于 AFlow 单项算法的认可度。评估时不能把整个仓库的 Star 直接包装成某个子模块的 Star。

## 4. 重点候选深入分析

## 4.1 Agent Lightning APO

### 定位

Agent Lightning 是当前项目已经使用的核心框架。最新版已经提供 APO 算法模块，因此它是迁移成本最低的方案。

官方安装方式：

```bash
pip install "agentlightning[apo]"
```

官方文档：

- [APO 算法说明](https://microsoft.github.io/agent-lightning/stable/algorithm-zoo/apo/)
- [Train the First Agent](https://microsoft.github.io/agent-lightning/latest/how-to/train-first-agent/)
- [Write the First Algorithm](https://microsoft.github.io/agent-lightning/stable/how-to/write-first-algorithm/)

### 核心流程

APO 使用文本梯度和 Beam Search 优化 Prompt：

1. 在训练任务上运行当前 Prompt；
2. 收集消息、轨迹、错误和 reward；
3. 使用 LLM 生成针对当前 Prompt 的文本批评；
4. 根据批评生成多个候选 Prompt；
5. 在验证集上评测候选；
6. 保留表现较好的若干候选；
7. 重复多轮并输出历史最优 Prompt。

官方示例使用 29 条验证任务，将准确率从 0.569 提升到 0.721；设置 8 个 runner 时约运行 10 分钟。该结果只能作为官方能力证明，不能作为本项目实验结果使用。

### 优点

- 当前项目已经采用 Agent Lightning，迁移成本最低；
- Rollout、Trace、Resource 和 Trainer 生命周期统一；
- 可以使用 CPU 加模型 API 完成 Prompt 优化；
- 已经具备训练集/验证集和候选选择逻辑；
- 支持自定义 Algorithm，便于后续接入 GEPA；
- GitHub 认可度高，项目活跃。

### 局限

- 当前 APO 主要优化单个 PromptTemplate；
- 不原生支持同时优化多个 Prompt、工具描述和 Harness 配置；
- 可能产生格式错误或不符合业务规则的 Prompt；
- 需要项目自行提供 dataset、reward 和轨迹适配；
- Beam Search 和 textual gradient 算法创新度有限。

### 适合承担的角色

第一阶段最小闭环和算法基线。

## 4.2 GEPA

### 定位

GEPA（Genetic-Pareto）是反思式文本优化框架。它不仅使用标量 reward，还能够读取完整轨迹、错误信息、性能分析和自然语言诊断，从而生成有针对性的候选修改。

官方安装方式：

```bash
pip install gepa
```

官方资源：

- [GEPA GitHub](https://github.com/gepa-ai/gepa)
- [GEPA 文档](https://gepa-ai.github.io/gepa/)
- [DSPy GEPA 说明](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/GEPA/overview.md)

### 核心流程

```text
从 Pareto 前沿选择候选
→ 在任务子集运行
→ 收集分数与可操作文本反馈
→ LLM 反思失败原因
→ 生成定向变异
→ 评测新候选
→ 更新 Pareto 前沿
→ 必要时合并不同候选的优势
```

GEPA 将诊断性的文本信息称为 Actionable Side Information。AgentRx 的失败类别、失败步骤、证据和修复建议可以转换成这种信息。

### 可优化对象

- system prompt；
- planner prompt；
- tool description；
- MCP 工具定义；
- Agent skill；
- 工作流配置；
- 代码或规则文本；
- Agent 架构描述。

### 优点

- 与 AgentRx 诊断高度互补；
- 支持 API-only 模型，不要求访问模型权重；
- 适合昂贵、长轨迹的 Agent rollout；
- 支持多候选和 Pareto 选择；
- 优化过程可解释，每次修改可以追溯到失败轨迹；
- 独立 API 可嵌入现有系统，不强制重写 Agent；
- GitHub 活跃，已有 DSPy、Opik、MLflow 等生态集成。

### 局限

- 需要实现项目专用 Adapter 和 evaluator；
- 默认示例以 Prompt 和单任务评测为主；
- 复杂 Harness 资源必须先定义序列化方式和合法性约束；
- 搜索轮数较多时仍会产生较高 API 成本；
- 不等价于模型权重层面的 RL。

### 适合承担的角色

第二阶段高级优化器，以及项目一主要算法亮点。

## 4.3 DSPy

### 定位

DSPy 是模块化 LM 程序和自动优化框架，内置或集成：

- GEPA；
- MIPROv2；
- SIMBA；
- BootstrapFewShot；
- ReAct 等 Agent 模块。

安装：

```bash
pip install dspy
```

### 优点

- 社区认可度和成熟度高；
- 优化算法选择丰富；
- 适合做多模块 LM 程序优化；
- 文档、测试、版本发布和实验生态完善。

### 局限

- 采用自己的 Module、Signature、Predictor 和 Example 抽象；
- 若整体引入，可能与既有 Agent 框架（tau2、Agent Lightning）的 Trace 结构耦合较重，需要大量适配代码。

> ⚠️ 注：本文件原始版本在"若整体引入，可能"处截断（2026-08-08 修复时发现正文
> 尾部被追加约 46KB 空字节；三个历史会话转录中均无完整版，原始结尾不可恢复）。
> 上述结尾一句与下方小节为依据文档既有结构（4.1/4.2 均有"适合承担的角色"）补全。

### 适合承担的角色

作为集成 GEPA、MIPROv2 等优化器的宿主框架；项目一可借鉴其 Module/Signature 抽象，但不整体引入。

