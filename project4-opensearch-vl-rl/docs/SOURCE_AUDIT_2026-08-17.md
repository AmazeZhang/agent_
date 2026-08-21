# OpenSearch-VL 源码审计（2026-08-17）

## 结论先行

OpenSearch-VL **值得作为秋招主项目候选**，而且比单纯复现 Search-R1 更贴近“多模态 Agentic 训练”岗位：它同时覆盖图像理解、搜索与视觉工具调用、Agentic SFT、在线多轮 RL、长轨迹失败处理和分布式训练。

但当前仓库还不是“下载后一条命令端到端复现”的状态。更准确的定位是：**核心训练代码、训练数据、模型权重已经开放；数据生成流水线尚未开放；SFT→RL 的衔接和部分工具注册需要我们自行修补。**

项目建议评分：**8.5/10，适合作为主项目；前提是把源码差异修通，并做小规模对照实验，而不是只运行作者 checkpoint。**

本地仓库：`F:\2027秋招项目\OpenSearch-VL`

审计版本：commit `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`

---

## 1. 训练框架

### 1.1 Agentic SFT

- 基座：Qwen3-VL-8B / 30B-A3B / 32B，也提供 Qwen2.5-VL、Qwen3.5-VL 配置。
- 框架：仓库内置的 LLaMA-Factory fork。
- 调度与并行：Ray + DeepSpeed ZeRO-3。
- 默认训练方式：全参数训练，LLM、视觉塔和多模态 projector 均不冻结，不是 LoRA。
- 8B 官方配置：`cutoff_len=32000`、bf16、学习率 `2e-5`、8 epochs、单卡 batch size 1、gradient checkpointing。
- 数据观察 masking：ShareGPT 中工具返回使用 `observation` 角色，训练只学习模型生成的思考、工具调用和最终回答，不学习搜索结果/OCR 等环境输出。

入口配置：`SFT/examples/agentic_full/qwen3_vl_full_sft_8b_ray.yaml`

### 1.2 Agentic RL

- Agent 编排：rLLM `AgentWorkflowEngine`。
- 策略优化：verl。
- 异步 rollout：SGLang。
- 大模型训练：Megatron-LM。
- Hugging Face / Megatron 权重转换：mbridge。
- 集群调度：Ray。
- Actor/Reference 更新和在线真实工具调用均在训练循环内完成。

8B 单机脚本默认值：

- advantage estimator 默认是 **RLOO**，并非论文标题强调的 GRPO；可把参数改成 `grpo`。
- 每个 prompt 采样 8 条轨迹；prompt batch 256；mini-batch 64。
- 最大 prompt 4096 tokens，最大 response 70000 tokens。
- 256 个并行任务、2048 个并行工具调用。
- Actor 学习率 `1e-6`，KL controller 系数 `0.001`。

官方论文完整训练成本很高：8B SFT 使用 256 张 H20 约 2 天，RL 使用 64 张 H20 约 10 天、约 200 个优化 step。我们的目标不应是原样重做算力规模，而应是保持算法链路、缩小样本和步数。

---

## 2. 算法构成

整体链路：

`Qwen3-VL-Instruct → Agentic SFT → 多工具异步 rollout → 复合奖励 → Fatal-aware RLOO/GRPO`

### 2.1 Agentic SFT

SFT 学习完整的多轮行为轨迹：

1. 看图并生成 `<think>`；
2. 选择并生成 `<tool_call>`；
3. 接收图像或文本 observation；
4. 基于历史视觉上下文继续搜索；
5. 生成最终 `<response>`。

图像工具产生的新图会继续进入视觉上下文，文本搜索结果和 OCR observation 则参与后续推理但不计算生成损失。这一点比普通“图文问答 SFT”更像真正的多模态 Agent。

### 2.2 复合奖励

源码中的总奖励为：

`r = r_format × (0.8 × r_accuracy + 0.2 × r_query)`

- `r_format`：每一步是否严格遵守 think→tool_call / think→response 格式。
- `r_accuracy`：最终答案是否正确；默认调用 GPT-4o 类 judge，judge 不可用时退化为归一化 exact match。
- `r_query`：搜索查询是否相关、是否逐步推进、检索信噪比是否合理、图搜与文本搜索是否互补；依赖外部 LLM judge。

### 2.3 Fatal-aware 多轮 RL

这是项目最值得讲的算法点：

1. 连续 3 个错误步骤被判为 fatal cascade；单次错误后如果恢复，计数清零。
2. 从第一个 fatal step 开始，将后续 response token 的训练 mask 置零。
3. fatal 之前的有效前缀仍参与训练，而不是整条轨迹全部丢弃。
4. fatal 轨迹参与组内 advantage 统计，但其 advantage/return 被单边截断到不小于 0；因此有效前缀只会得到正向更新，最差退化为零梯度。

这比 Search-R1 式普通搜索 GRPO 多了一个明确面向长工具链故障的 credit assignment 设计，也比“整条失败轨迹 hard mask”保留了更多有效探索信号。

---

## 3. 数据集构成

### 3.1 SearchVL-SFT-36K

- 论文给出的精确规模：**36,592 条**多轮专家轨迹。
- 平均每条轨迹：**6.3 次工具调用**。
- 仓库配置包含 7 个子集：FVQA、PALACE、WebQA、LiveVQA、WikiArt、Wikipedia English、Wikipedia Chinese。
- 格式：ShareGPT，字段为 `conversations`、`images`、`system`、`tools`。
- 轨迹由 Claude Opus 4.6 在真实工具环境中生成，再用 GPT-4o 做答案正确性筛选、GPT-5.4 做过程质量筛选。
- 数据构造还包含 Wikipedia 2/3/4-hop 路径、实体模糊改写、source-anchor 图像对齐、CLIP 筛图、工具必要性过滤，以及约 10% 的图像退化/修复样本。

重要限制：

- Hugging Face 上的原始 JSON 和图片资产已经开放，但在线 Dataset Viewer 当前显示解析失败。
- **Wikipedia 路径采样、模糊改写、source-anchor grounding 等数据生成代码尚未开放**；README TODO 明确标为未完成。因此我们可以复现训练数据的使用，不能完整复现 36K 数据从零生成。

### 3.2 SearchVL-RL-8K

实际下载并逐行统计后，`rl_data.jsonl` 有 **7,992 条**，字段是 `question / answer / images / dataset`，并不是精确 8,000 条。

| 来源 | 条数 | 占比 |
|---|---:|---:|
| new_livevqa | 3,746 | 46.9% |
| WebQA | 1,507 | 18.9% |
| demo_1k | 1,000 | 12.5% |
| wiki_zh | 527 | 6.6% |
| wiki_en | 406 | 5.1% |
| palace | 369 | 4.6% |
| wikiart | 253 | 3.2% |
| new_fvqa | 184 | 2.3% |

注册脚本默认随机种子 42、按 90/10 切分，因此对应 **7,192 条 train + 800 条 test**。论文称 RL 样本与用于合成 SFT 轨迹的 VQA 样本互斥；仓库没有提供额外脚本让我们独立验证该去重过程。

---

## 4. 效果

以下都是作者论文在统一 Pass@1 + GPT-4o judge 下报告的结果，**目前还不是我们本地独立复现的结果**。

### 4.1 七个 benchmark 平均分

| 模型 | Agentic baseline | OpenSearch-VL | 提升 |
|---|---:|---:|---:|
| 8B | 42.0 | 56.6 | +14.6 |
| 30B-A3B | 47.8 | 61.6 | +13.8 |
| 32B | 48.0 | 63.7 | +15.7 |

8B OpenSearch-VL 比 SenseNova-MARS-8B 的 52.7 高 3.9 分。30B-A3B 在 VDR、MMSearch、FVQA、InfoSeek 上分别相对同基座 agentic baseline 提升 13.3、24.5、10.2、16.2 分。

### 4.2 SFT + RL 消融（8B，三项平均）

| 方法 | 平均分 |
|---|---:|
| Qwen3-VL-8B base | 53.7 |
| + SFT | 64.6 |
| + Vanilla GRPO | 67.6 |
| + Hard Masking | 67.7 |
| + Fatal Masking only | 69.1 |
| + Fatal Masking + One-sided Clamp | **71.8** |

这组消融非常适合项目讲述：SFT 提供工具使用冷启动，普通 GRPO 提供在线探索，fatal mask 和单边 clamp 再解决长轨迹中“前面做对、后面工具崩溃却整条受罚”的问题。

---

## 5. 源码与 README 的关键差异

这些问题不妨碍把它作为项目，反而可以成为我们“理解并工程化修复开源训练链路”的工作量：

1. **SFT→RL 没有在启动脚本中自动接通。** 四个 RL 脚本的 `MODEL_PATH` 都指向原始 `Qwen/Qwen3-VL-*-Instruct`，不是 SFT 输出 checkpoint。要复现论文两阶段路线，必须手动修改。
2. **默认算法名不一致。** 论文主打 fatal-aware GRPO，但发布脚本默认 `adv_estimator=rloo`；必须显式切到 `grpo` 才是论文叙述中的配置。
3. **工具数量不一致。** README 宣称统一 10 工具；RL 实际注册 9 个，缺少 `visit`。`VisitTool` 文件存在，但未接入 `get_all_tools()`；Python 工具注册名还是 `PythonInterpreter`，与 README 的 `python_interpreter` 不同。
4. **数据生成不是全开源。** 训练成品数据开放，但生成 36K/8K 的独立流水线仍在 TODO。
5. **SFT 集群说明不一致。** README 示例写 16×8 GPU，而 8B YAML 的 `ray_num_workers` 是 256；论文报告又是 32×8 H20。小规模复现时需要自己重写资源配置。

---

## 6. 对我们项目的建议

建议做“**受控小规模复现 + 开源链路修补 + 一组消融**”，而不是追求论文全量数字：

1. 使用官方 OpenSearch-VL-8B checkpoint 先跑通多工具推理和评测，确认环境闭环。
2. 用官方 8B 基座和 100～500 条 SFT 轨迹跑短程 cold start，验证 observation masking 和工具格式学习。
3. 把 RL 模型入口明确接到 SFT checkpoint，取几十到几百条 RL 样本跑少量 step。
4. 至少比较 `vanilla RLOO/GRPO`、`fatal mask`、`fatal mask + clamp` 三种设置；即便数据很小，也能展示我们真正理解了算法。
5. 修复 `visit` 注册、工具命名和配置一致性，并记录 API 失败率、fatal 比例、平均工具轮数、准确率四类指标。

最终项目故事可以是：**复现并工程化改造一个多模态深度搜索 Agent，通过 Agentic SFT 获得多工具冷启动，再用 fatal-aware RL 对长链工具交互做在线优化；针对开源仓库的阶段断点、工具不一致和外部 API 不稳定问题完成修复，并通过小规模消融验证有效前缀 credit preservation。**

这个故事具有多模态、Agent、SFT、RL、训练系统和工程诊断六个层次，明显比只复现一个文本搜索 GRPO 项目更完整。
