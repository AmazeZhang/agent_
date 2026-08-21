# 多模态 Agentic 训练候选项目

> 用途：记录 2027 秋招阶段可用于理解、缩小复现与简历定向投递的开源项目候选。后续讨论持续更新本文件，不把上游完整论文结果表述为个人复现结果。
>
> 更新时间：2026-08-17

## 1. 选择标准

理想候选同时满足：

1. 数据、训练代码、模型权重与评测尽可能公开；
2. 模型输入包含图像或视频，并在 rollout 中主动调用视觉、检索、代码或浏览器工具；
3. 采用 Agentic SFT / Cold-start + Agentic RL 双阶段训练；
4. 允许只复现少量数据、小模型或短程训练，不以复现论文完整规模为必要条件；
5. 简历价值以真实掌握、真实运行和可解释证据为基础，可以突出关键链路，但不得把上游方法写成个人原创或把未运行的完整实验写成个人结果。

## 2. 当前候选池

| 候选 | 开放情况 | Agentic 多模态 | 双阶段 | 当前判断 |
|---|---|---|---|---|
| OpenSearch-VL | SearchVL-SFT-36k、SearchVL-RL-8k、SFT/RL/推理/评测代码、8B/30B/32B 权重，Apache-2.0 | 图像输入；crop、OCR、图像增强、文本/图像/网页搜索、visit、Python | Agentic SFT + Fatal-aware GRPO/RLOO/PPO | 技术覆盖最完整、与多模态搜索岗位最匹配；系统较重 |
| PixelReasoner | 约 7.85k SFT 轨迹、RL Query、WarmStart/RL 权重、SFT/RL/评测代码 | 图像/视频；zoom、crop、select-frame 等像素空间操作 | Instruction Tuning + Curiosity-driven GRPO | 当前最均衡：论文与开源成熟、双阶段清晰、适合缩小复现 |
| OpenThinkIMG | 数据、模型、视觉工具服务、SFT/V-ToolRL 代码基本公开 | 多视觉工具与动态多轮调用 | SFT + V-ToolRL | 2B 底座更轻，但仓库未识别到明确许可证、维护与文档完整性弱于前两项 |
| DeepEyesV2 | SFT/RL 数据、模型和 veRL Agent 框架公开 | 图像操作、代码执行、搜索组合 | Cold-start SFT + Agentic RL | 方法强、知名度高，但官方链路包含大规模训练和 72B Judge，复现工程较重 |
| MMSearch-R1 | GRPO 代码、模型、FVQA/缓存数据部分公开 | 图搜与文搜，多轮真实搜索 | 以 RL 为主 | README 要求自行搭搜索工具且训练目录主要提供样例数据，不是完整即用双阶段方案 |
| VTool-R1 | 数据、训练和评测代码公开，Apache-2.0 | Python 视觉编辑工具、图文交错推理 | 主要是 RFT/RL | Agentic RL 清晰，但不符合“明确 SFT + RL 主线”的首选条件 |
| PyVision-RL | RL 数据、SFT checkpoint、RL 代码公开 | 图像与视频按需取帧/视觉交互 | 从公开 SFT checkpoint 进入 RL | SFT 数据生产与训练链路未完整开放，不满足完整双阶段复现标准 |

## 3. 与现有 Search-R1 项目的基线关系

现有 Search-R1 已具备的真实证据：

- 真实 Wiki-18 Retriever、Qwen2.5-1.5B LoRA、veRL GRPO 单卡训练与 checkpoint 恢复闭环；
- 官方宽松语义线 256 条留出集上，官方 Search-R1 3B GRPO 为 32/256，Base 为 20/256，配对 McNemar `p=0.0357`，证明环境能够观察上游 RL 效果；
- 自有严格 fork 线 train64nqh8 为 31/256，Base 为 37/256，尚无个人训练收益；
- 后续算法空间为 GRPO/GiGPO、Step Advantage、Structured Anchor State 与信用分配。

多模态候选相对 Search-R1 的主要增量：

- 从文本 Query/文档检索扩展为图像或视频输入及动态视觉 observation；
- 工具动作更丰富，覆盖 crop、zoom、OCR、增强、图搜、网页访问、代码执行或视频选帧；
- 可呈现视觉证据在多轮 rollout 中被重新加入上下文的机制；
- 更贴合多模态搜索、视觉 Agent、视频理解和智能创作岗位；
- SFT 冷启动与 RL 行为优化的职责分工通常更直观。

Search-R1 仍保留的优势：

- 当前已有真实训练、严格留出评测、配对统计和失败结论，可信度高于尚未运行的新项目；
- Retriever、Reward、GRPO/GiGPO、Advantage 与 Policy Loss 主线更聚焦，面试更容易讲深；
- 个人改进 Structured-state GiGPO 比纯上游缩小复现更有原创辨识度；
- 文本环境更容易形成稳定、可复验的训练闭环。

## 4. 当前排序

### 不考虑完整规模复现，只看项目上限与简历信号

1. **OpenSearch-VL**：技术覆盖最完整，最接近“多模态搜索 Planning + 工具调用 + SFT/Agentic RL”。
2. **PixelReasoner**：两阶段最清晰、开源成熟度高、缩小复现风险更低。
3. **DeepEyesV2**：方法和知名度强，但工程链路重。
4. **OpenThinkIMG**：模型更小，适合训练机制学习，但仓库成熟度和许可证较弱。

### 以秋招时间内形成可信证据为目标

1. **PixelReasoner 缩小复现**；
2. **OpenSearch-VL 关键链路/小规模复现**；
3. 继续完成现有 **Search-R1 GRPO/GiGPO**；
4. OpenThinkIMG；
5. DeepEyesV2。

## 5. 待讨论决策

- 新项目是替换 Search-R1，还是仅作为多模态岗位版本的定向补充；
- 更看重“多模态搜索与外部知识获取”（OpenSearch-VL），还是“视觉工具推理与图像/视频证据获取”（PixelReasoner）；
- 最低个人证据要求：推理评测、SFT 非零更新、短程 GRPO、checkpoint、Base/SFT/RL 同协议对照；
- 简历中严格区分“复现/缩小实验”“上游方法”和“个人修改”。
