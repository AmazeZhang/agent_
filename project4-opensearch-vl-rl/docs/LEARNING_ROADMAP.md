# OpenSearch-VL 系统学习路线与进度台账

> 建立日期：2026-08-17  
> 对应仓库：`F:\2027秋招项目\OpenSearch-VL`  
> 唯一进度基准：本文件。  
> 审计 commit：`c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`  
> 学习目标：以 OpenSearch-VL 为主轴，系统掌握多模态搜索 Agent、Agentic SFT、RLOO/GRPO、复合奖励、Fatal-aware RL、异步训练系统与实验分析，最终能够完整应对项目面试追问。

## 使用规则

- `[ ]`：尚未学习。
- `[-]`：正在学习或仍有关键疑问。
- `[x]`：已通过验收，能够脱离材料用自己的话解释。
- 每次只推进一个知识点；当前项未通过验收，不进入下一项。
- 每节按照“直觉 → 定义/公式 → 项目源码 → 示例 → 面试表达 → 检查题”学习。
- 完成一节后，更新本文件中的复选框、完成日期、掌握结论和遗留问题。
- 完成一节后，在本目录创建对应的独立复习文档，命名为 `序号-知识点.md`。
- 作者论文结果、本地复现结果和个人改造结果必须严格区分。

## 当前进度

- 当前阶段：第一阶段——项目背景与任务定义
- 当前章节：第 2 节——任务定义与应用场景
- 已完成：1 / 24
- 下一步：严格定义 OpenSearch-VL 的任务输入、输出、成功条件，并区分 VQA、知识密集型 VQA、图像检索和 Visual Deep Research。

---

## 项目主线

```text
研究背景与任务定义
    ↓
多模态 Agent 决策循环与工具环境
    ↓
多跳 VQA 数据构造与专家轨迹合成
    ↓
Agentic SFT 工具使用冷启动
    ↓
在线多轮 rollout 与 RLOO/GRPO
    ↓
复合奖励与长轨迹信用分配
    ↓
Fatal-aware Masking + One-sided Clamp
    ↓
异步分布式训练、评测与消融
    ↓
小规模复现、个人改造与面试表达
```

---

# 第一阶段：项目背景与任务定义

## [x] 第 1 节：项目背景与研究动机

复习文档：[`01-项目背景与研究动机.md`](01-项目背景与研究动机.md)

需要掌握：

- [x] 普通 VLM 单次推理的能力边界。
- [x] 为什么知识密集型视觉问题需要外部信息搜索。
- [x] 普通 VLM、Multimodal RAG 和 Multimodal Agent 的区别。
- [x] Search-R1、Vision-DeepResearch 与 OpenSearch-VL 的关系。
- [x] OpenSearch-VL 试图解决的数据、工具环境和长轨迹训练问题。

验收标准：

- [x] 能在 2 分钟内回答“为什么不用一个更强的 VLM 直接看图回答？”
- [x] 能解释该项目为什么不只是图像搜索项目。

完成日期：2026-08-18  
掌握结论：能够区分普通 VLM、固定 RAG 与多模态 Agent；理解 OpenSearch-VL 的目标是优化完整的多工具问题求解策略；能够解释在线环境故障如何导致普通结果奖励错误惩罚有效前缀，并说明 Fatal-aware 方法的动机。  
遗留问题：后续在第 16～17 节结合公式和源码，进一步区分 fatal token masking 与 one-sided advantage clamp 的精确作用。

## [-] 第 2 节：任务定义与应用场景

需要掌握：

- [x] 初始输入 `(I₀, q)` 和最终答案 `a`。
- [-] 普通 VQA、Knowledge-intensive VQA、图文检索、Visual Deep Research 的区别（需补牢 action 与 environment observation 的边界）。
- [x] 地标、艺术品、文档、实体溯源等典型任务。
- [x] 为什么评价目标是最终问答正确率，而不是单纯 Recall@K。

验收标准：

- [-] 能给出一个必须经过“视觉定位 → 检索 → 验证”才能回答的例子（需改成项目中存在的具体工具链）。
- [x] 能准确描述项目任务的输入、输出和成功条件。

## [ ] 第 3 节：项目总体架构

需要掌握：

- [ ] Policy、Agent loop、Tool Environment、Training 四层结构。
- [ ] Qwen3-VL、rLLM、verl、SGLang、Megatron-LM 的职责边界。
- [ ] Agentic SFT → Agentic RL → Evaluation 的完整链路。

验收标准：

- [ ] 能脱离材料画出系统框图。
- [ ] 能解释每个模块的输入、输出和相互关系。

---

# 第二阶段：多模态 Agent 算法基础

## [ ] 第 4 节：多轮决策建模

需要掌握：

- [ ] 历史/状态 `hₗ`、推理 `zₗ`、工具调用 `cₗ`、观察 `oₗ`、轨迹 `τ`。
- [ ] `aₗ ~ πθ(·|hₗ)` 的含义。
- [ ] MDP、POMDP 与该项目的对应关系。
- [ ] 数据集、环境、策略探索之间的区别。

核心公式：

```text
hₗ = (I₀, q, a₀, o₀, …, aₗ₋₁, oₗ₋₁)
aₗ ~ πθ(· | hₗ)
```

验收标准：

- [ ] 能说明模型在 RL 中究竟探索什么。
- [ ] 能解释 observation 为什么不是模型动作。

## [ ] 第 5 节：ReAct 与工具调用协议

需要掌握：

- [ ] `<think>`、`<tool_call>`、`<tool_response>`、`<response>` 的职责。
- [ ] 单轮一个工具调用的控制方式。
- [ ] 工具名和参数为何都属于模型生成动作。
- [ ] 格式错误、工具错误和答案错误的区别。

验收标准：

- [ ] 能手写一条合法的多轮 XML/JSON 工具轨迹。
- [ ] 能解释为什么格式本身也需要奖励约束。

## [ ] 第 6 节：多模态观察与 Active Visual Context

需要掌握：

- [ ] 文本 observation 与图像 observation 的区别。
- [ ] Crop、Sharpen 等新图片如何回到视觉编码器。
- [ ] 为什么历史图片需要保留。
- [ ] “Think with Image”的具体含义。

核心公式：

```text
𝓘ₗ = {I₀} ∪ {oₖ : k < l, oₖ ∈ 𝓞img}
```

验收标准：

- [ ] 能解释视觉工具为什么是 Agent 动作，而不只是数据预处理。

## [ ] 第 7 节：多模态工具环境与非确定性

需要掌握：

- [ ] Crop、Layout Parsing、Image Search、Text Search、Web Search。
- [ ] Perspective Correction、Super Resolution、Sharpen、Python Interpreter。
- [ ] Serper、Jina Reader、OCR 服务、LLM summarizer 的作用。
- [ ] 本地确定性工具与联网非确定性工具。
- [ ] 搜索漂移、API 超时、网页变化和环境非平稳性。

验收标准：

- [ ] 能比较 Search-R1 离线检索环境和 OpenSearch-VL 在线环境。
- [ ] 能提出缓存/冻结工具环境的复现方案。

---

# 第三阶段：数据构造与 Agentic SFT

## [ ] 第 8 节：为什么普通 VQA 数据不够

需要掌握：

- [ ] 直接看图回答、参数知识捷径、实体名泄露。
- [ ] 一次检索 shortcut 与真正多跳工具需求。
- [ ] Tool-demanding 数据的设计原则。

验收标准：

- [ ] 能判断一个样本是否真正需要 Agent。

## [ ] 第 9 节：Wikipedia 多跳 VQA 构造

需要掌握：

- [ ] Wikipedia 2/3/4-hop 路径采样。
- [ ] Anchor、Bridge、Answer node 的功能。
- [ ] Fuzzy Entity Rewriting。
- [ ] Source-anchor Visual Grounding。
- [ ] CLIP 筛图、staged filtering 和 enhancement subset。

核心结构：

```text
v₀ → v₁ → … → vₕ
```

验收标准：

- [ ] 能解释图片为什么绑定起始实体而不是答案实体。
- [ ] 能解释如何抑制 single-hop shortcut。

## [ ] 第 10 节：专家轨迹合成与拒绝采样

需要掌握：

- [ ] Claude 候选轨迹生成。
- [ ] GPT-4o correctness judge。
- [ ] GPT-5.4 process judge。
- [ ] Rejection Sampling 与轨迹质量控制。
- [ ] 好答案和好过程的区别。
- [ ] 数据开放与数据生成代码未开放的边界。

验收标准：

- [ ] 能完整复述 SearchVL-SFT-36K 的生产链路。

## [ ] 第 11 节：Agentic SFT 目标函数

需要掌握：

- [ ] 轨迹自回归概率分解。
- [ ] Generation mask `Mgen`。
- [ ] 为什么工具 observation 不参与生成损失。
- [ ] SFT 如何学习思考、选工具、填参数和停止。
- [ ] Imitation ceiling。

核心公式：

```text
πθ(τ | I₀,q) = ∏ₗ Pθ(aₗ | hₗ)
Pθ(aₗ | hₗ) = Pθ(zₗ | hₗ)Pθ(cₗ | hₗ,zₗ)

L_SFT = -Σₜ Mgen(yₜ) log πθ(yₜ | y<t, I)
```

验收标准：

- [ ] 能手工标出一条轨迹中哪些 token 计算 SFT loss。
- [ ] 能解释为什么不能训练模型复述环境返回内容。

---

# 第四阶段：在线 Agentic RL

## [ ] 第 12 节：为什么 SFT 后还需要 RL

需要掌握：

- [ ] Offline imitation 与 on-policy exploration。
- [ ] SFT 无法主动发现更优路径的原因。
- [ ] RL 数据为何只有图片、问题和答案。
- [ ] 当前策略在线生成轨迹的含义。

验收标准：

- [ ] 能解释 SFT 和 RL 各自解决什么问题。
- [ ] 能解释 RL rollout 不是重新生成训练问题。

## [ ] 第 13 节：Policy Gradient、PPO、GRPO 与 RLOO

需要掌握：

- [ ] Policy Gradient 与 advantage。
- [ ] PPO importance ratio 与 clipped objective。
- [ ] GRPO 组内标准化和无 critic 训练。
- [ ] RLOO leave-one-out baseline。
- [ ] 每个 prompt 采样多条轨迹的原因。
- [ ] 论文主讲 GRPO、脚本默认 RLOO 的差异。

核心公式：

```text
∇θJ(θ) = E[Aₜ ∇θ log πθ(aₜ | sₜ)]

rₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)

L_PPO = -E[min(rₜAₜ, clip(rₜ,1-ε,1+ε)Aₜ)]

Âᵢ_GRPO = (rᵢ - mean(r₁,…,rG)) / (std(r₁,…,rG)+ε)

Âᵢ_RLOO = rᵢ - (1/(G-1))Σⱼ≠ᵢ rⱼ
```

验收标准：

- [ ] 能从 Policy Gradient 推到 GRPO 的直觉。
- [ ] 能准确比较 GRPO 与 RLOO。
- [ ] 能解释为什么 GRPO 不需要 value model。

## [ ] 第 14 节：复合多轮奖励

需要掌握：

- [ ] Format reward。
- [ ] Accuracy reward。
- [ ] Query-quality reward。
- [ ] 乘法 format gate。
- [ ] Sparse reward、process reward 和 reward hacking。
- [ ] LLM-as-a-Judge 的成本、偏差和复现问题。

核心公式：

```text
r(τ) = r_fmt(τ)[0.8r_acc(τ) + 0.2r_query(τ)]
r_fmt = (1/(L+1))Σₗ r_fmt^(l)
```

验收标准：

- [ ] 能解释三个奖励缺一会出现什么问题。
- [ ] 能批判性分析 query judge 的局限。

---

# 第五阶段：Fatal-aware RL 核心算法

## [ ] 第 15 节：长轨迹错误信用分配

需要掌握：

- [ ] 前缀正确、后缀故障的典型轨迹。
- [ ] Outcome reward 无法定位失败步骤的问题。
- [ ] 整条丢弃和 hard masking 的信息浪费。
- [ ] 策略错误与外部环境错误的区别。

验收标准：

- [ ] 能构造一个普通 GRPO 会错误惩罚有效前缀的例子。

## [ ] 第 16 节：Fatal Step Detection 与 Token Masking

需要掌握：

- [ ] 连续三次错误的 fatal 判定。
- [ ] 单次错误后恢复时计数清零。
- [ ] 第一个 fatal step 的定位。
- [ ] Fatal token mask 与 observation mask 的区别。

核心公式：

```text
M_fatal,t^(i) = 1[t < fᵢ]
```

验收标准：

- [ ] 能手工标记一条多轮轨迹的 fatal step 和 token mask。
- [ ] 能对应源码解释 mask 如何生效。

## [ ] 第 17 节：One-sided Advantage Clamping

需要掌握：

- [ ] Fatal 轨迹为什么仍参与组内统计。
- [ ] 只保留 fatal 前缀正向 advantage 的原因。
- [ ] Mask 与 Clamp 的职责差异。
- [ ] Hard mask、Fatal mask、Fatal mask + Clamp 的区别。
- [ ] 只奖不罚可能引入的偏差。

核心公式：

```text
Âᵢ_fatal = max(Âᵢ, 0)
```

验收标准：

- [ ] 能完整推演一组 fatal/non-fatal rollout 的 advantage。
- [ ] 能回答“为什么 clamp 不用于所有轨迹？”
- [ ] 能用 3 分钟讲清项目的核心算法创新。

---

# 第六阶段：训练系统与源码实现

## [ ] 第 18 节：异步 Agent Rollout 系统

需要掌握：

- [ ] rLLM、SGLang、verl、Megatron-LM、mbridge、Ray 的分工。
- [ ] Rollout worker 与 training worker。
- [ ] Policy 权重同步。
- [ ] 工具并发与 GPU 推理并发。
- [ ] 异步训练中的 stale policy 和近似 on-policy。
- [ ] 长上下文和累积图像的显存成本。

验收标准：

- [ ] 能画出一次“采样 → 奖励 → advantage → 更新 → 权重同步”。

## [ ] 第 19 节：两阶段 checkpoint 与仓库断点

需要掌握：

- [ ] SFT 输出如何接入 RL。
- [ ] 当前 RL 脚本为何仍指向原始 Qwen checkpoint。
- [ ] 默认 RLOO 与论文 GRPO 的差异。
- [ ] README 十工具与 RL 九工具的差异。
- [ ] `visit` 注册、Python 工具命名和资源配置问题。

验收标准：

- [ ] 能列出端到端复现前必须修复的配置。
- [ ] 能区分作者设计、发布代码和我们的改动。

---

# 第七阶段：实验、复现与面试

## [ ] 第 20 节：Benchmark 与评测协议

需要掌握：

- [ ] SimpleVQA、VDR、MMSearch、LiveVQA。
- [ ] BrowseComp-VL、FVQA、InfoSeek。
- [ ] Pass@1、GPT-4o judge 和统一评测。
- [ ] Direct reasoning、RAG workflow、Agentic workflow baseline。

验收标准：

- [ ] 能解释每类 benchmark 主要考查什么能力。
- [ ] 能说明在线评测为何难以严格复现。

## [ ] 第 21 节：主结果与消融实验

需要掌握：

- [ ] 8B、30B-A3B、32B 主结果。
- [ ] Base → SFT → Vanilla GRPO → Fatal Mask → Clamp。
- [ ] Hard Mask 基本无收益的含义。
- [ ] 哪些结论可以由消融支持，哪些不能。

核心结果链：

```text
53.7 → 64.6 → 67.6 → 69.1 → 71.8
```

验收标准：

- [ ] 能完整解读消融表，而不是只背数字。
- [ ] 能严格区分论文数字和本地实验数字。

## [ ] 第 22 节：批判性分析与局限

需要掌握：

- [ ] 数据生成流水线未开放。
- [ ] 在线环境非平稳。
- [ ] GPT judge 依赖和偏差。
- [ ] Query reward 不覆盖全部视觉操作。
- [ ] 工具和配置不一致。
- [ ] 缺少多随机种子结果。
- [ ] Fatal 阈值和 one-sided clamp 的潜在偏差。

验收标准：

- [ ] 能提出至少三个具体局限及对应改进方向。

## [ ] 第 23 节：我们的复现与改造方案

计划完成：

- [ ] 固定或缓存联网工具返回。
- [ ] 跑通官方 checkpoint 多工具推理。
- [ ] 小样本 Agentic SFT。
- [ ] 把 SFT checkpoint 正确接入 RL。
- [ ] 小规模 RLOO/GRPO。
- [ ] Vanilla、Fatal Mask、Fatal Mask + Clamp 消融。
- [ ] 在线环境和冻结环境对比。
- [ ] 统计准确率、平均轮数、工具故障率和 fatal 率。
- [ ] 修复工具注册和配置一致性。

验收标准：

- [ ] 形成可执行实验计划、资源预算和成功标准。
- [ ] 明确哪些工作可以诚实写成个人贡献。

## [ ] 第 24 节：面试项目表达与追问答辩

需要完成：

- [ ] 30 秒项目简介。
- [ ] 2 分钟项目完整介绍。
- [ ] 5 分钟算法深入讲解。
- [ ] 项目架构图口述。
- [ ] GRPO/RLOO 公式讲解。
- [ ] Fatal-aware 算法讲解。
- [ ] 数据、系统、实验和局限追问。
- [ ] “你个人做了什么”回答。

验收标准：

- [ ] 在不看材料的情况下完成一轮模拟面试。
- [ ] 对主要追问能够先给结论、再给原理和证据。

---

# 学习记录

## 记录 0：路线建立

- 日期：2026-08-17
- 内容：完成 OpenSearch-VL 源码初步审计，建立 24 节系统学习路线。
- 已确认事实：
  - RL 数据实际为 7,992 条。
  - SFT、RL 数据和模型权重已开放。
  - 数据生成流水线尚未开放。
  - RL 脚本默认 RLOO，论文主要描述 GRPO。
  - RL 脚本默认模型路径未接入 SFT checkpoint。
  - README 与 RL 实际工具注册存在差异。
- 下一步：开始第 1 节“项目背景与研究动机”。

## 记录 1：完成项目背景与研究动机

- 日期：2026-08-18
- 完成章节：第 1 节。
- 已掌握：
  - 更大的 VLM 仍受参数知识、视觉可见信息和证据可靠性限制。
  - 固定 RAG 的流程由系统预设，Agent 的工具决策由模型策略控制。
  - OpenSearch-VL 的最终目标是知识密集型视觉问答，而非单纯图像 Recall@K。
  - 在线多模态工具故障可能造成“前缀正确、后缀失败”，从而引出 Fatal-aware 信用分配。
- 表述修正：Fatal-aware 方法不是无条件给前缀正奖励；它屏蔽 fatal 后缀，并将 fatal 轨迹的负 advantage 截断为零，正 advantage 仍可保留。
- 下一步：第 2 节“任务定义与应用场景”。

---

# 待维护的最终产物

- [ ] OpenSearch-VL 一页架构图。
- [ ] 数据构造流程图。
- [ ] Agent rollout 时序图。
- [ ] SFT observation masking 示例。
- [ ] GRPO/RLOO 公式推导笔记。
- [ ] Fatal-aware 算法手算示例。
- [ ] 训练系统组件图。
- [ ] 小规模复现实验表。
- [ ] 项目局限与改进清单。
- [ ] 30 秒、2 分钟和 5 分钟面试话术。
- [ ] 高频追问题库与答案。
