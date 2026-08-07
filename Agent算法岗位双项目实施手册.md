# Agent 算法岗位双项目实施手册

> Harness 自进化 × Coding Agentic RL  
> 面向 2027 届秋季校园招聘的服务器实施、实验验证与面试准备方案  
> 版本：v1.0，2026-08-06

## 0. 最终决策

最终确定以下两个互补项目：

1. **面向有状态工具任务的 Trace 驱动 Agent 自进化与可靠性评测系统**
2. **基于可执行轨迹与过程奖励的 Coding Agent 后训练系统**

项目一解决 Agent 系统如何被观察、诊断和持续优化；项目二解决 Agent 模型如何通过可执行轨迹和强化学习真正更新策略参数。二者共同覆盖当前 Agent 算法岗位从 Harness、Trace、Evaluator 到 Agentic RL 后训练的完整链路。

| 项目 | 核心问题 | 技术主线 | 主要产出 |
|---|---|---|---|
| Harness 自进化 | Agent 为什么失败，应优化 Prompt、Tool、Memory 还是 Policy？ | τ-bench + AgentRx + Agent Lightning | 诊断闭环、可靠性评测、自动优化、回归门禁 |
| Coding Agentic RL | 如何利用代码执行反馈训练模型完成长程软件修复？ | SWE-agent + SWE-smith + rLLM | 可执行轨迹、SFT/GRPO、过程奖励、跨仓库验证 |

最终评价标准不是“运行了几个高 Star 仓库”，而是：

> 基于成熟开源组件，形成自己的问题定义、关键模块、算法改进、消融实验、量化结果和失败分析。

---

## 1. 岗位背景与项目目的

### 1.1 岗位背景

阿里巴巴、字节跳动及其他公司 Agent 算法岗位的共同要求，正在从“会搭建 Agent Demo”升级为“能够设计数据、环境、训练、评测和自进化闭环”。高频能力包括：

- 长程规划、工具调用、Memory、RAG、MCP/Skills 与多 Agent 协作；
- 轨迹采集、合成数据、失败归因、Evaluator/Verifier 和回归评测；
- SFT、DPO、PPO、GRPO 及 Agentic RL 训练；
- Harness、可执行环境、沙箱隔离、Trace 和过程级信用分配；
- Python/PyTorch、vLLM/SGLang、verl/rLLM、Ray、DeepSpeed/FSDP；
- Coding、Browser、Computer Use 等可验证或半可验证任务。

### 1.2 项目目的

1. 建立从任务环境、Agent 执行、轨迹诊断到自动优化的可复现闭环。
2. 完成一次真正更新模型参数的 Agentic RL 实验，而不是停留在 Prompt Engineering。
3. 围绕信用分配、失败轨迹利用、动态采样和循环控制形成个人创新点。
4. 沉淀可量化结果、消融实验、Demo、技术报告和面试问答。
5. 能够解释每个设计选择、指标变化、失败案例与工程取舍。

### 1.3 双项目关系

```text
项目一：任务执行 → Trace采集 → 失败归因 → 分类优化 → 回归门禁 → 可靠性提升
                                      ↓ 关键失败Turn / 优质轨迹
项目二：任务环境 → 多轨迹Rollout → 可执行Reward → SFT/GRPO → 策略更新 → 跨仓库评测
```

项目一生产结构化 Trace、失败类型和优化策略；项目二把可执行环境、轨迹与奖励接入模型后训练。项目一优先优化 Harness 和 Prompt 资源，项目二负责模型权重更新，避免内容重复。

### 1.4 研究边界

- 不从零实现完整大模型训练框架，基于 Agent Lightning、rLLM 或 verl 扩展关键模块。
- 不追求全量复现 SWE-smith 5 万级数据，使用精选仓库和可控任务规模完成严谨实验。
- 不把 LLM-as-a-Judge 作为唯一奖励，优先采用数据库状态、测试、构建和策略规则等可验证信号。
- 不把测试集信息泄漏到训练；合成任务可使用注入位置作为训练辅助信号，真实评测不可使用。
- 不以单次成功作为结论；固定种子、重复采样并报告 Pass@1、pass^k 和置信区间。

---

## 2. 推荐仓库结构

```text
agent-portfolio/
├── project1-harness-evolution/
│   ├── configs/
│   ├── agents/
│   ├── tracing/
│   ├── diagnosis/
│   ├── optimizers/
│   ├── evaluation/
│   └── dashboard/
├── project2-coding-agent-rl/
│   ├── environments/
│   ├── synthesis/
│   ├── rollouts/
│   ├── rewards/
│   ├── training/
│   └── evaluation/
├── shared/                 # 日志格式、模型客户端、成本统计、实验工具
├── experiments/            # 每次实验的配置、指标和说明
├── reports/                # 基线、消融、失败分析与最终报告
└── README.md
```

---

# 项目一：Trace 驱动 Agent 自进化系统

## 3. 项目概述

### 3.1 正式名称

**面向有状态工具任务的 Trace 驱动 Agent 自进化与可靠性评测系统**  
英文：**Trace-Driven Self-Evolving Tool Agent**

### 3.2 背景与问题定义

真实工具 Agent 失败时，单一成功率无法说明问题来自哪里。错误可能源于：

- 指令理解错误；
- 业务策略约束遗漏；
- 工具选择或调用顺序错误；
- Tool Schema 或参数错误；
- 历史状态遗忘；
- 长程规划错误；
- 工具失败后无法恢复；
- 环境超时或模拟器异常。

如果没有结构化 Trace 和回归评测，“自进化”很容易退化成反复修改 Prompt 并挑选好结果。

本项目建立以下闭环：

```text
执行 → Trace采集 → 失败定位 → 根因分类 → 分类优化 → 回归评测 → 接受/拒绝新版本
```

从 τ-bench 有状态工具任务生成轨迹，用 AgentRx 或自研诊断模块定位关键失败步骤，再由分类优化器选择 Prompt、Tool、Memory 或 Policy 优化动作，最后通过固定 Holdout 和可靠性指标决定是否接受新版本。

### 3.3 核心研究问题

1. 能否从长轨迹中准确定位第一个关键失败 Turn，而不只判断最终失败？
2. 能否根据失败类别选择正确优化对象，避免所有问题都修改 System Prompt？
3. 一次优化提高平均成功率时，是否破坏原先已经成功的任务？
4. 同一任务重复执行时，Agent 是否稳定，还是依赖偶然采样？
5. 可靠性提升是否以更多 Token、工具调用和延迟为代价？

## 4. 系统架构

| 层级 | 组件 | 职责 |
|---|---|---|
| Environment | τ-bench/τ²-bench | 提供有状态任务、工具、用户模拟器和策略约束 |
| Agent | ReAct 或 Plan-Execute Agent | 理解目标、选择工具、管理状态并给出最终答复 |
| Observability | Agent Lightning Trace | 记录 Prompt、Tool Call、Observation、Latency、Token、Reward Span |
| Diagnosis | AgentRx + 自研规则 | 失败定位、根因分类、置信度与证据抽取 |
| Optimizer | Category-conditioned Optimizer | 按错误类型修改 Prompt、工具描述、Memory 策略或训练资源 |
| Gate | Regression & Reliability Gate | 在开发集和 Holdout 上决定接受、回滚或继续优化 |
| Dashboard | 实验与案例看板 | 展示指标、成本、错误分布和典型轨迹对比 |

### 4.1 Trace 最小字段

```yaml
run:
  task_id: string
  domain: string
  seed: int
  agent_version: string
  model: string
  prompt_version: string

turn:
  turn_id: int
  role: string
  thought_summary: string
  tool_name: string | null
  tool_args: object | null
  observation_summary: string | null
  state_delta: object | null
  policy_hits: list[string]
  error: string | null
  latency_ms: int
  input_tokens: int
  output_tokens: int

result:
  reward: float
  sub_metrics: object
  terminal_reason: string
  diagnosis_label: string | null
  critical_failure_turn: int | null
  evidence: list[string]
```

### 4.2 失败标签体系

| 类别 | 典型表现 | 优先优化动作 |
|---|---|---|
| Instruction | 误解用户目标或约束 | Prompt、任务分解 |
| Policy | 违反退款、授权等业务规则 | Policy 提示、Verifier、硬门禁 |
| Tool Selection | 选错工具或调用顺序错误 | 工具路由、工具描述 |
| Tool Argument | 字段缺失、格式或实体错误 | Schema、参数校验、重试 |
| Memory/State | 忘记历史状态或使用过期信息 | 结构化 Memory、状态摘要 |
| Planning | 局部正确但长程路径错误 | Planner、里程碑、Policy 训练 |
| Recovery | 工具失败后没有纠正 | 恢复策略、失败示例、过程奖励 |
| Environment | 超时、服务或模拟器异常 | 重试或丢弃，不错误更新模型 |

## 5. 自进化方法

核心创新为 **Category-conditioned Optimizer**：诊断器输出错误类别、置信度和关键证据，优化器根据类别选择有限且可审计的动作，而不是让一个 LLM 自由改写整个 Agent。

1. 执行基线版本，在固定任务集上收集多次 Rollout。
2. 筛选失败和不稳定轨迹，定位关键失败 Turn 并进行根因分类。
3. 根据错误类别产生一个或多个候选优化补丁。
4. 先在开发集上筛选，再在不可见 Holdout 上回归验证。
5. 只有成功率、可靠性和安全指标满足门槛时才更新版本。
6. 保存被拒绝的优化及其退化原因，形成失败知识库。

### 5.1 优化动作映射

```text
Instruction错误  → System Prompt / Few-shot / Task Decomposition
Policy错误       → Policy文本 / Verifier / Hard Gate
Tool Selection   → Tool Router / Tool Description
Tool Argument    → JSON Schema / Validator / Retry Policy
Memory错误       → Structured Memory / State Summary
Planning错误     → Planner / Milestone / Policy RL
Recovery错误     → Recovery Prompt / Failure Example / Process Reward
Environment错误  → Retry / Discard / Environment Fix
```

### 5.2 RL 在项目一中的位置

项目一第一阶段以 APO 和 Harness 优化为主；当诊断结果显示错误来自长期规划或工具策略时，再进行小规模 Policy RL。

建议做一个明确的 RL 对照：

> 在相同任务和模型上比较 Outcome-only GRPO 与 Critical-turn Reward GRPO，验证失败定位是否改善信用分配和样本效率。

RL 训练的对象是模型在每个 Turn 选择推理或工具动作的策略，而不是训练环境本身。

## 6. 项目一指标与验收

| 维度 | 核心指标 | 建议验收目标 |
|---|---|---|
| 效果 | Pass@1、Task Success | 相对基线提升不少于 10% |
| 可靠性 | pass^k、重复运行方差 | 稳定成功任务比例明显提升 |
| 诊断 | Failure Localization Accuracy | 关键失败 Turn 定位准确率不低于 65% |
| 归因 | Root-cause Macro-F1 | 主要失败类别 Macro-F1 不低于 0.60 |
| 恢复 | Recovery Rate | 工具异常后的恢复率提升不少于 15% |
| 回归 | Regression Rate | 已成功任务退化率不高于 5% |
| 效率 | Tool Calls、Token、Latency、Cost | 效果提升不以超过 30% 成本增长换取 |

以上是项目目标，不是预先承诺的结果。简历只能写实际复现实验中获得且可追溯的数字。

### 6.1 分阶段实施

| 阶段 | 主要工作 | 阶段产物 |
|---|---|---|
| P1-0 环境 | 固定仓库提交、模型、任务与依赖，跑通 Mock 和少量真实任务 | 安装脚本、Smoke Test |
| P1-1 基线 | 50–100 个任务、多 Seed 运行 | Baseline Report |
| P1-2 Trace | 统一 Span 与轨迹格式，接入日志和可视化 | Trace Dataset v1 |
| P1-3 诊断 | 失败标签、关键 Turn 定位、AgentRx 对比 | Diagnosis Benchmark |
| P1-4 优化 | 分类优化器、候选补丁、回归门禁 | Self-evolution Loop |
| P1-5 RL 可选 | Outcome-only 与 Critical-turn Reward 对比 | 小模型 RL 消融 |
| P1-6 收尾 | 跨 Domain 验证、成本分析、Demo、技术报告 | 可展示作品集 |

---

# 项目二：Coding Agentic RL 后训练系统

## 7. 项目概述

### 7.1 正式名称

**基于可执行轨迹与过程奖励的 Coding Agent 后训练系统**  
英文：**Executable-Trajectory Coding Agent Post-training**

### 7.2 背景与问题定义

Coding Agent 必须在 Repository 中进行多轮搜索、编辑和测试。最终结果可以通过测试、构建和回归验证，因此非常适合 Reinforcement Learning with Verifiable Rewards。

但长轨迹会产生以下问题：

- 最终奖励稀疏；
- 失败轨迹中的正确前缀被错误惩罚；
- Agent 容易重复查看、重复修改或超时；
- 环境执行时间差异大，产生 Straggler；
- 全对或全错的 Rollout Group 缺乏有效相对优势；
- 训练集有效，但跨 Repository 泛化不足；
- Agent 可能通过删除测试、硬编码等方式 Reward Hacking。

本项目利用 SWE-smith 构建可控 Bug 和执行环境，利用 SWE-agent 或轻量 Agent 收集轨迹，再通过 rLLM/verl 完成 SFT 和 GRPO。重点不是追求大型榜单成绩，而是在固定算力内验证失败感知动态采样、关键 Turn 信用分配与状态感知循环控制。

## 8. RL 任务建模

| RL 元素 | Coding Agent 中的定义 |
|---|---|
| State/Observation | Issue、Repository 状态、历史操作、工具输出、测试结果、剩余预算 |
| Action | 搜索、查看文件、编辑 Patch、运行测试、回退修改、提交答案等模型 Token 序列 |
| Transition | 命令执行、文件变化、测试状态或错误信息导致的环境变化 |
| Episode | 从接收 Issue 到成功修复、主动提交、超时或超过最大步数 |
| Policy | Coding LLM 在当前上下文中生成下一次思考和工具调用的概率分布 |
| Reward | 可执行结果为主，测试进展、恢复和效率信号为辅 |

## 9. 数据与轨迹流水线

1. 选择 3–8 个规模适中、测试完善、许可证清晰的 Python Repository。
2. 构建 Docker 执行环境并记录基础镜像、依赖和测试命令。
3. 通过 LLM Rewrite、AST Mutation、PR Revert 或组合方式注入 Bug。
4. 只保留能够稳定导致至少一个测试失败且可被原始补丁修复的任务。
5. 生成 Issue 描述，划分 Train/Dev/Test，并按 Repository 进行 Held-out 隔离。
6. 使用教师模型或当前策略采集多条轨迹，保存完整工具交互和环境状态。
7. 通过 Patch 应用、测试、回归和安全规则自动标注奖励及失败类型。
8. 从验证成功的优质轨迹构建 SFT 数据，再进行 On-policy GRPO。

### 9.1 推荐数据记录格式

```yaml
task:
  task_id: string
  repo: string
  base_commit: string
  issue: string
  split: train | dev | test
  mutation_type: rewrite | ast | revert | combined
  oracle_patch: string | null
  fail_to_pass_tests: list[string]
  pass_to_pass_tests: list[string]

trajectory:
  trajectory_id: string
  policy_version: string
  seed: int
  turns: list[Turn]
  final_patch: string
  terminal_reason: success | submit | timeout | max_steps | environment_error

evaluation:
  patch_applies: bool
  build_success: bool
  fail_to_pass_score: float
  pass_to_pass_score: float
  regression_count: int
  prohibited_change: bool
  reward_components: object
  total_reward: float
```

## 10. 奖励设计

总奖励采用“终局可执行结果主导、过程奖励有限塑形、违规行为硬门禁”的原则：

```text
R = w1·R_resolve + w2·R_progress + w3·R_localize + w4·R_recovery
    - λ1·C_regression - λ2·C_loop - λ3·C_invalid - λ4·C_cost
```

约束：

- `R_resolve` 权重最高；
- 安全、越权和篡改测试使用 Hard Gate；
- 过程奖励必须通过消融证明有效；
- 成本项初期只作为评测指标，确认无副作用后再加入训练。

| 奖励项 | 计算方式 | 风险控制 |
|---|---|---|
| Resolve | Fail-to-Pass 全部通过且 Pass-to-Pass 无回归 | 主奖励，隐藏测试验证 |
| Progress | 目标测试新增通过比例减去新增回归比例 | 低权重，避免只追局部改善 |
| Localization | 训练期编辑是否命中合成 Bug 文件或函数 | 仅用于合成训练，不用于真实测试 |
| Recovery | 首次失败后是否利用反馈纠正并成功 | 要求环境状态确实改善 |
| Loop Penalty | 相同状态下重复命令、查看或 Patch | 允许测试后状态变化的合理重复 |
| Invalid/Safety | 非法工具、越权路径、篡改测试、删除关键文件 | 优先 Harness 阻止并判失败 |
| Cost | 归一化步骤、Token、时间和测试次数 | 初期只评测，稳定后小权重加入 |

### 10.1 终局奖励建议

```text
完整修复且没有回归：        +1.00
目标测试全部通过但引入回归： 0.00 或直接失败
只修复部分目标测试：         0.00～0.30
Patch 无法应用：             -0.20
篡改或删除测试：              Hard Fail
越权访问或高风险操作：         Hard Fail
```

具体权重必须通过开发集和消融确定，不能为了获得漂亮曲线反复针对测试集调参。

## 11. 三个核心创新

### 11.1 Failure-aware Dynamic Sampling

统计每类任务当前策略的组内成功率。全对组和全错组通常缺乏相对优势信号，因此优先采样成功率处于中间区间、存在有效策略分叉的任务，同时保留一定困难任务用于探索。

需要记录：

- 有效 Group 比例；
- 零方差 Group 比例；
- 成功轨迹比例；
- Reward 方差；
- Policy Entropy；
- 单位有效样本成本。

消融：

```text
Uniform Sampling
vs Difficulty Sampling
vs Failure-aware Dynamic Sampling
```

### 11.2 Critical-turn Credit Assignment

利用测试变化、文件状态和诊断器定位导致失败的关键 Turn，避免对整条失败轨迹统一赋予负优势。

处理方式：

- 正确前缀转化为 SFT 数据；
- 失败点构建反事实分叉；
- 对关键 Turn 分配局部奖励或负奖励；
- 同时保留最终 Outcome Reward 作为全局约束。

指标：

- 关键 Turn 定位准确率；
- 成功前缀利用率；
- 单位 Rollout 的有效训练 Token；
- 样本效率；
- 训练稳定性。

消融：

```text
Outcome-only
vs Milestone Reward
vs Critical-turn Reward
```

### 11.3 State-aware Loop Control

循环检测不仅比较命令字符串，还比较：

- 文件 Hash；
- Git Diff；
- 测试状态；
- 当前目录；
- Observation 摘要；
- 未解决错误集合。

只有在环境状态没有有效变化时才判定无效循环，并进行惩罚或提前终止。

指标：

- Repetition Rate；
- Timeout Rate；
- 平均轨迹长度；
- 平均 Tool Calls；
- 成功任务平均成本。

消融：

```text
No Control
vs String Deduplication
vs State-aware Loop Control
```

## 12. 训练方案

### 12.1 模型与规模

- 开发模型：Qwen Coder 1.5B/3B 级，用于快速验证管线和奖励。
- 正式模型：根据服务器显存选择 3B/7B 级，先 LoRA/QLoRA SFT，再 GRPO。
- SFT 数据：首版 500–2,000 条验证成功轨迹。
- RL 任务：首版 100–500 个。
- 每个任务采样 4–8 条轨迹。
- 正式实验固定 Seed，并保存 Policy 版本和权重同步时间。

### 12.2 训练顺序

1. Base 模型评测，确定任务难度与初始成功率。
2. 使用优质成功轨迹 SFT，建立稳定工具调用和基础修复能力。
3. 运行 Outcome-only GRPO 标准基线。
4. 加入 Failure-aware Dynamic Sampling。
5. 加入 Critical-turn Reward，观察 Reward Hacking 和训练波动。
6. 加入 State-aware Loop Control，验证成本和成功率变化。
7. 在未见 Repository 上进行最终冻结评测。

### 12.3 核心实验矩阵

| 编号 | 训练配置 | 回答的问题 |
|---|---|---|
| E0 | Base Model | 原始模型具备多少 Agent 能力？ |
| E1 | SFT | 可执行成功轨迹能带来多少提升？ |
| E2 | SFT + Outcome-only GRPO | 标准 Agentic RL 是否优于 SFT？ |
| E3 | E2 + Dynamic Sampling | 更有价值的 Group 能否提高样本效率？ |
| E4 | E3 + Critical-turn Reward | Turn 级信用分配能否减少错误惩罚？ |
| E5 | E4 + Loop Control | 能否降低超时与成本而不伤害成功率？ |
| E6 | Repo-held-out | 提升能否迁移到未见仓库？ |

## 13. 项目二评价指标

### 13.1 效果

- Resolve Rate / Pass@1；
- Pass@k；
- Fail-to-Pass；
- Pass-to-Pass；
- Regression Rate。

### 13.2 轨迹与行为

- Localization Accuracy；
- Patch Precision；
- Valid Rollout Ratio；
- Recovery Rate；
- Repetition Rate；
- Timeout Rate；
- Steps to First Edit；
- Test Efficiency。

### 13.3 系统与成本

- Tool Calls；
- Tokens；
- Latency；
- Environment Throughput；
- GPU 等待比例；
- Straggler P95；
- Rollout 失败重试率；
- Cost per Resolved Task。

### 13.4 泛化

- Repo-held-out Resolve Rate；
- 训练仓库与未见仓库的性能差距；
- 不同 Bug 类型之间的迁移能力。

### 13.5 分阶段实施

| 阶段 | 主要工作 | 阶段产物 |
|---|---|---|
| P2-0 环境 | 跑通 SWE-agent、SWE-smith 和 Docker 沙箱 | 单任务端到端 Smoke Test |
| P2-1 数据 | 选择仓库、构建环境、合成并验证 Bug | Task Dataset v1 |
| P2-2 轨迹 | 教师或基线模型多轨迹采样、失败分类 | Trajectory Dataset v1 |
| P2-3 SFT | 成功轨迹过滤、格式规范、LoRA 训练 | SFT Model 与报告 |
| P2-4 RL 基线 | Outcome-only GRPO 与稳定性监控 | RL Baseline |
| P2-5 创新 | 动态采样、关键 Turn 奖励、循环控制 | 完整消融实验 |
| P2-6 泛化 | 未见仓库冻结评测、成本与失败分析 | Final Evaluation |
| P2-7 展示 | Demo、README、技术报告、面试材料 | 可投递作品集 |

---

# 服务器部署与工程规范

## 14. 推荐服务器规格

| 层级 | 建议配置 | 可完成范围 |
|---|---|---|
| 开发/基线 | 1×24GB GPU，16–32 vCPU，64GB RAM，1TB NVMe | 环境、轨迹、7B 量化推理、1.5B/3B SFT 与小型 RL |
| 推荐正式 | 1×48–80GB GPU 或 2×24GB，32 vCPU，128GB RAM，2TB NVMe | 3B/7B SFT、较稳定 GRPO、多环境并发 |
| 扩展实验 | 4–8×80GB GPU，64+ vCPU，256GB+ RAM | 大 Batch、多模型对比、系统吞吐实验 |

如果预算有限，优先保证内存、NVMe 和 CPU 环境并发，再增加 GPU。Coding Agentic RL 经常因 Docker 环境和测试执行成为瓶颈，而不只是显存不足。

## 15. 基础软件建议

- Ubuntu 22.04 或项目官方验证版本；
- NVIDIA Driver 与 CUDA 按 PyTorch/vLLM 兼容矩阵选择；
- Docker + NVIDIA Container Toolkit；
- Python 环境使用 uv 或 Conda 隔离；
- 模型服务使用 vLLM/SGLang；
- 训练采用 rLLM/verl；
- 任务调度使用 Ray 或轻量异步队列；
- 实验跟踪使用 Weights & Biases、MLflow 或本地等价方案；
- 使用独立数据盘或对象存储保存模型、轨迹和日志。

## 16. 服务器目录规划

```text
/data/agent-projects/
├── source/        # Git仓库
├── models/        # 基座、SFT与RL Checkpoint
├── datasets/      # 任务、轨迹和拆分清单
├── docker/        # 镜像缓存和环境工件
├── runs/          # 每次实验日志、配置、Seed与指标
└── artifacts/     # 报告、图表、Demo与可发布结果
```

## 17. 可复现性要求

- 每次实验保存 Git Commit、镜像 Digest、模型版本、数据版本、随机种子和完整配置。
- 基线、开发和测试任务 ID 固定，测试集在最终评测前冻结。
- Reward 代码必须有单元测试，覆盖正常、部分成功、回归、超时和违规案例。
- 对 API 模型记录提供商、模型快照日期、温度和 Token 上限。
- 所有结果从原始轨迹自动生成，避免手工复制指标。
- 定期检查密钥、用户数据、Repository 许可证和合成数据来源。

## 18. 安全要求

Coding Agent 必须运行在最小权限沙箱中：

- 默认断网或使用白名单网络；
- 只挂载当前任务目录；
- 禁止挂载宿主 Docker Socket；
- 限制进程数、CPU、内存、磁盘和执行时间；
- 对高风险命令做 Harness 硬阻断；
- 训练和评测使用独立凭证；
- 保存安全违规 Trace 用于审计，但不得让策略从漏洞中持续获益。

---

# 联合时间表与验收

## 19. 建议 16 周路线

| 周次 | 重点 | 里程碑 |
|---|---|---|
| 1–2 | 服务器、仓库、版本与基础环境 | 两个项目均完成 Smoke Test |
| 3–4 | 项目一基线与 Trace | 首份基线报告和轨迹数据 |
| 5–6 | 项目一诊断与自进化闭环 | 诊断基准、优化器、回归门禁 |
| 7 | 项目一最终实验与 Demo | 项目一可写入简历 |
| 8–9 | 项目二环境与 Bug 合成 | 首批可执行任务集 |
| 10–11 | 轨迹采集与 SFT | SFT 模型和对比报告 |
| 12–13 | Outcome-only GRPO 基线 | RL 训练曲线与稳定性分析 |
| 14–15 | 三个创新点消融 | 动态采样、信用分配、循环控制结果 |
| 16 | 跨仓库评测、报告与面试材料 | 最终作品集和答辩稿 |

## 20. 每周实验节奏

1. 周初确定一个可证伪的问题和对照实验，不同时修改多个核心变量。
2. 先运行小样本 Smoke Test，确认日志、Reward 和资源占用正确。
3. 正式运行前锁定配置并估算 GPU 时长与 API 成本。
4. 实验结束自动生成指标、曲线和失败案例列表。
5. 周末写一页结论：假设、结果、解释、反例、下一步。

## 21. 最终验收清单

- [ ] 两个仓库均可通过一条命令复现最小实验。
- [ ] 环境、数据、训练、评测和可视化均有清晰 README。
- [ ] 至少一套公开或可发布的小规模数据或轨迹样例。
- [ ] 项目一具备基线、诊断、优化、回归和成本五类结果。
- [ ] 项目二具备 Base、SFT、GRPO 与至少两个创新组件的消融。
- [ ] 所有核心数字可从原始结果自动重算。
- [ ] 提供不少于 10 个成功案例和 10 个失败案例的人工分析。
- [ ] 提供 5–10 分钟 Demo 视频或现场演示脚本。
- [ ] 完成中文技术报告、英文 README 和一页项目海报。

---

# 简历、作品集与面试准备

## 22. 简历表述原则

- 用问题、方法、个人贡献和量化结果描述，不以 Star 数代替贡献。
- 明确哪些模块来自开源仓库，哪些由自己设计和实现。
- 只写已经完成且能解释代码、公式和实验的内容。
- 同时报告效果、可靠性、成本和泛化，避免只给最佳单次分数。
- 准备失败实验；面试官通常会从异常和取舍判断项目深度。

## 23. 项目一最终简历要素

- 任务环境与规模：Domain、任务数、重复采样次数；
- Trace 和失败体系：记录粒度、标签数、诊断准确率；
- 自进化方法：分类优化器、回归门禁、候选选择；
- 结果：Pass@1/pass^k、恢复率、回归率、成本变化；
- 个人创新：Critical-turn 诊断或 Reliability-Cost 联合目标。

## 24. 项目二最终简历要素

- 任务与数据：仓库数、合成 Bug 数、有效任务率、轨迹数；
- 训练：基座模型、SFT 数据量、GRPO 配置和服务器规模；
- 奖励：可执行终局奖励、过程塑形和 Reward Hacking 防护；
- 创新：动态采样、关键 Turn 信用分配、状态感知循环控制；
- 结果：Resolve Rate、回归率、重复率、成本、跨仓库泛化。

## 25. 必须能回答的面试问题

1. 为什么项目一不用 RL 解决所有问题？什么情况下 APO 比更新权重更合适？
2. Agentic RL 与单轮 Reasoning RL 在 State、Action 和 Credit Assignment 上有什么区别？
3. PPO 与 GRPO 的目标函数、优势估计、KL 和显存开销有什么差异？
4. 为什么失败轨迹不能整体赋予相同负奖励？
5. 过程奖励如何避免改变最优策略或导致 Reward Hacking？
6. 同一任务一组 Rollout 全对或全错时，GRPO 还能学到什么？
7. 为什么必须做 Repository 级 Held-out，而不是随机划分任务？
8. 如何区分模型失败、工具失败和环境失败？
9. 异步 Rollout 如何提高吞吐，又会带来什么 Policy Staleness？
10. 如果结果没有提升，如何判断问题在数据、Reward、探索、优化器还是环境？
11. Token-level 与 Sequence-level Policy Gradient Loss 有什么差异？
12. 为什么可执行 Reward 通常比 LLM-as-a-Judge 更适合 Coding Agent？
13. 如何设计隐藏测试和沙箱，防止 Agent 投机？
14. 如何证明过程奖励真的提高了样本效率，而不是只提高训练集 Reward？
15. GRPO、DAPO、PPO 在本项目中的选择依据是什么？

---

# 附录 A：指标口径

| 指标 | 定义 |
|---|---|
| Pass@1 | 每个任务一次采样的平均成功率 |
| pass^k | 同一任务连续 k 次均成功的任务比例，用于衡量可靠性 |
| Pass@k | 同一任务 k 次采样中至少一次成功的概率 |
| Regression Rate | 原本成功或原测试通过的样本在新版本中退化的比例 |
| Recovery Rate | 出现可恢复失败后最终成功完成任务的比例 |
| Valid Rollout Ratio | 格式、工具、环境和终止状态均有效的轨迹比例 |
| Repetition Rate | 被状态感知检测器判为无进展重复的动作比例 |
| Cost per Resolved Task | 总模型与环境成本除以成功任务数 |

---

# 附录 B：关键风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| 范围过大 | 长期停留在部署多个框架 | 先完成最小闭环，每阶段设置停止条件 |
| Reward Hacking | 删除测试、硬编码、利用环境漏洞 | 隐藏测试、硬门禁、沙箱、人工审计 |
| 数据泄漏 | 使用 Bug 位置或测试答案评测 | 按 Repository 隔离，训练辅助信号不进入测试 |
| 算力浪费 | 大量全错或全对 Group | 动态采样、任务难度分层、早停 |
| 环境长尾 | 某些测试拖慢整个 Batch | 异步环境、超时、隔离重试、Straggler 监控 |
| 不可复现 | API 模型漂移、镜像变化 | 锁版本、保存 Digest、记录日期和配置 |
| 项目套壳 | 只能描述框架功能 | 自研模块、消融、失败分析、明确贡献边界 |

---

# 附录 C：单次实验记录模板

```markdown
# Experiment <ID>: <Title>

## Hypothesis

本实验试图验证什么？预期为什么会提升？什么结果会证伪假设？

## Changes

- 相对基线只修改了哪些变量？
- 哪些配置保持不变？

## Environment

- Git commit:
- Docker image digest:
- Model/checkpoint:
- Dataset version:
- GPU/CPU/RAM:
- Seed:
- Training framework:

## Training

- Tasks:
- Rollouts per task:
- Batch size:
- Learning rate:
- KL coefficient:
- Max turns/tokens:
- Sampling temperature:
- Reward weights:

## Results

| Metric | Baseline | Experiment | Delta |
|---|---:|---:|---:|
| Pass@1 | | | |
| Regression Rate | | | |
| Recovery Rate | | | |
| Repetition Rate | | | |
| Cost/Resolved Task | | | |

## Failure Analysis

- 成功案例：
- 失败案例：
- 新增失败模式：
- 是否观察到 Reward Hacking：

## Conclusion

- 假设是否成立？
- 最可能的解释是什么？
- 还需要什么对照实验？
- 下一步是什么？
```

---

# 附录 D：参考项目与资料

- [Microsoft Agent Lightning](https://github.com/microsoft/agent-lightning)
- [Microsoft AgentRx](https://github.com/microsoft/AgentRx)
- [τ-bench / τ²-bench](https://github.com/sierra-research/tau2-bench)
- [SWE-agent](https://github.com/SWE-agent/SWE-agent)
- [SWE-smith](https://github.com/SWE-bench/SWE-smith)
- [rLLM](https://github.com/rllm-org/rllm)
- [ByteDance Seed DAPO](https://github.com/BytedTsinghua-SIA/DAPO)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [TRACE：Turn-level Credit Assignment](https://arxiv.org/abs/2607.13988)
- [BEACON：Milestone-guided Policy Learning](https://arxiv.org/abs/2605.06078)
- [阿里云强化学习开发指南](https://help.aliyun.com/zh/model-studio/rl-function-development-guide)
- [RollArt：Agentic RL System](https://www.usenix.org/conference/osdi26/presentation/gao)

> 实施时需再次检查各仓库最新版本、许可证、数据条款和官方推荐依赖，并在实验报告中记录实际使用的 Commit。
