# 开源生态调研：Agent 自进化 / 诊断驱动改进闭环

- 调研时间：2026-08-07
- 调研方式：Web 搜索（README / 论文摘要级信息），**尚未 clone 验证**；采用前需逐仓核查活跃度、license 和代码质量
- 调研目的：为项目一"改进策略"环节（闭环中目前唯一未接通的环节）寻找可借鉴或可接入的开源实现

## 一、结论先行

2026 年开源生态已出现一批与项目一闭环**同构**的项目（采集轨迹 → 失败根因诊断 → 自动改进 → 重跑 → 对照），且多篇论文/工作报告了可量化提升。"改进策略"环节**不需要从零自研**，选择变为"接入 vs 借鉴 vs 继续自研"。

## 二、直接同构项目

### 2.1 完整闭环类

| 项目 | 来源 | 闭环形态 | 与项目一的对应 | 可借鉴点 / 注意点 |
|---|---|---|---|---|
| [Hive Agent Dev Framework](https://github.com/syntax-syndicate/hive-agent-development-framework) | GitHub | **Execute → Evaluate → Diagnose → Regenerate** 四阶段循环；失败后由 coding agent 改 prompt/图结构/工具/约束后重部署 | 字面同构于我们的五环节 | 文档明确"evolution 只针对见过的失败类型，不是通用智能"——与项目一验收边界一致；决策日志是核心信号 |
| [Hermes Agent Self-Evolution](https://github.com/zuquanzhi/hermes-agent-self-evolution) | NousResearch，ICLR 2026 Oral，MIT | **GEPA 读执行轨迹理解"为什么失败"（不只是失败了）→ 针对性改进 skills/prompts/code**；DSPy 驱动；~$2–10/次；100% 测试通过门禁 + 人工 PR 审查 | **最接近项目一定位**："why 失败"诊断 + 改进 + 回归门禁 | 改进环节可直接借鉴其 GEPA/DSPy 管线；注意其诊断深度不如 AgentRx |
| [TraceRoot](https://github.com/traceroot-ai/traceroot) | YC S25 | OTel 追踪 → LLM-as-judge 检测器 → 根因定位 → 自动开修复 PR；确认的失败沉淀为 golden dataset 做离线 eval | 覆盖采集+诊断+改进 | 偏生产可观测性；golden dataset 机制与我们的可信轨迹导出思想一致 |
| [HALO](https://pypi.org/project/halo-engine/0.1.9/)（halo-engine） | PyPI | 采集 OTel 轨迹 → 识别跨执行**共性失败模式** → findings 报告 → coding agent 改 harness → 重部署循环 | 与"在 7 条失败轨迹上找共性"思路一致 | AppWorld 上改进了 Gemini 3 Flash / Sonnet 4.6 的 harness |
| [recursive-improve](https://github.com/kayba-ai/recursive-improve) | GitHub | 捕获每次 LLM 调用 → 轨迹失败模式分析 → 定向修复 → 每次循环前后 benchmark | 开箱即用的 ratchet 式循环 | 含隔夜自主循环 + 每轮分支 + 仪表盘 |

### 2.2 治理 / 研究类（可借鉴设计）

| 项目 | 来源 | 内容 | 借鉴点 |
|---|---|---|---|
| [Ratchet](https://github.com/amazon-science/Self-Evolving-Agents-Ratchet) | Amazon，arXiv 2605.22148 | 技能自进化五角色循环（router/critic/synthesizer/curator 等），治理机制：技能退休、active-cap、回滚、不可变 Verdict 记录 | **进化治理机制**（防退化、可回滚）正是我们对照评测协议缺的一块 |
| [RethinkSkill](https://github.com/HKUST-KnowComp/rethinkskill) | HKUST，MIT | 学术研究：成功/失败反馈如何塑造技能进化；**11 个进化出的技能全部来自包含失败轨迹的条件**；388 候选 × 3 模型 × 5 基准 | 直接背书项目一前提——"用真实失败轨迹做诊断"；可作为面试中的学术论据 |
| [SCOPE](https://arxiv.org/abs/2512.15374)（JarvisPei） | arXiv | 轨迹分析 → 合成自然语言准则；双流更新（战术纠错 vs 战略原则）；HLE 14.23%→38.64%，GAIA 32.73%→56.97% | 改进环节的学术范本；双流机制值得参考 |

### 2.3 平台 / 其他

- [Future AGI](https://github.com/BeamNawapat/future-agi)（Apache 2.0）：tracing + evals + 6 种提示优化算法（GEPA、PromptWizard、ProTeGi、Bayesian、Meta-Prompt、Random），生产轨迹反馈为训练数据——改进算法全家桶，可作算法对照来源。
- [TRACE](https://github.com/ScalingIntelligence/TRACE)：**τ²-Bench 上 +15.4pp（32.9%→48.3%）**，GRPO 训能力特定 LoRA + MoE 路由——**与项目二直接相关**（同一基准 + Agentic RL），同时是项目一"改进不止于 prompt"的另一条路（直接训权重）。

### 2.4 低质量/需甄别

- [Sandesh-raut/self-evolving-agent](https://github.com/Sandesh-raut/self-evolving-agent)、[JNK234/Self-evolving-agent](https://github.com/JNK234/Self-evolving-agent)：hackathon/demo 级，仅作思路参考，不宜采用。

## 三、诊断环节对照：我们的差异化

- 上述项目多采用简单 LLM-as-judge 判断失败；**AgentRx 的六阶段 invariant 提取 + 10 类失败分类 + 可审计证据日志在深度上优于它们**。
- 若借鉴/接入上述闭环，建议**保留 AgentRx 作为诊断环节**，只替换/补强"改进"环节——这是差异化所在。

## 四、建议

1. **不再从零自研改进优化器**；优先评估把改进环节替换为可插拔优化器（DSPy/GEPA 一类）或参考 Hive/Ratchet 的 loop 与治理设计。
2. 候选接入顺序建议：Hermes/GEPA（最贴近定位）→ Hive（四阶段 loop 完整）→ Ratchet（治理机制）。
3. 采用前必须验证：仓库活跃度、license、与 DeepSeek 的兼容性、能否在现有硬件上运行。
4. RethinkSkill 的"失败轨迹信息量更大"结论可直接写入项目叙事和面试素材。
5. 项目二后续训练阶段可参考 TRACE（τ²-Bench + GRPO）的实验设计。

## 五、附录：项目二 vendor 完整可用性核查（2026-08-07）

| 组件 | 上游 | 固定版本 | 状态 |
|---|---|---|---|
| SWE-agent | github.com/SWE-agent/SWE-agent | 3ea751c（2026-07-16） | ✅ 官方上游；已应用浅拉取补丁（`sweagent/environment/repo.py` 显示 modified，符合预期，可复现补丁在 `patches/sweagent-shallow-reset.patch`） |
| SWE-smith | github.com/SWE-bench/SWE-smith | 9b74ac0（2026-03-21） | ✅ 官方上游，干净 |
| rllm | github.com/rllm-org/rllm | 1d1109a（2026-07-23） | ✅ 官方上游，干净；**但 `rllm-base` venv 未安装 verl/torch/vllm/ray，训练栈尚未安装** |

- 隔离环境：项目目录内 `.venvs/`（swe-tools、rllm-base 等 6 个），swe-tools venv 中 `import sweagent` 通过。
- Docker：本 shell 无 docker.sock 权限（需 sudo/tmux 环境）；历史运行记录（feasibility、五任务 pilot、run1–8）证明 SWE-ReX 镜像与 rollout 链路此前可用。
- **结论：项目二 rollout/评测/净化链路完整可用；唯一未到位的是 `rllm[verl]` 训练栈，属计划内后续步骤（排在数据质量门槛之后）。**

## 资料来源

- 各项目 GitHub/PyPI 链接见上表；以下为本次调研命中的原始检索结果：
  - TraceRoot：https://github.com/traceroot-ai/traceroot
  - HALO：https://pypi.org/project/halo-engine/0.1.9/
  - Ratchet：https://github.com/amazon-science/Self-Evolving-Agents-Ratchet（arXiv:2605.22148）
  - Hermes：https://github.com/zuquanzhi/hermes-agent-self-evolution
  - TRACE：https://github.com/ScalingIntelligence/TRACE
  - RethinkSkill：https://github.com/HKUST-KnowComp/rethinkskill
  - SCOPE：https://arxiv.org/abs/2512.15374
  - Future AGI：https://github.com/BeamNawapat/future-agi
  - Hive：https://github.com/syntax-syndicate/hive-agent-development-framework
  - recursive-improve：https://github.com/kayba-ai/recursive-improve
  - EvoMap/evolver：https://github.com/evomap/evolver
  - 自进化基准 ICLR 2026：https://mlanthology.org/iclr/2026/guo2026iclr-selfevolving/
