# P3 开源查询优化方法筛选与接入决策（2026-08-27）

## 结论

当前下一步不应继续单纯提高搜索率，也不应立即扩展 top-k 或启动新的长训练。
Aware-v2 已经稳定提高搜索行为，但两 seed 的 EM 净变化只有 `-1/+1`；seed2026 的检索—回答漏斗中，
`100` 个错误题已经搜索但没有命中答案证据，另有 `63` 个命中证据后仍答错。既有 query 的
top-3 加两篇候选只救回前一类中的 `10/100` 题，因此第一优先级是 query 规划、分解和改写。

开源方法中，**StepSearch 是最适合先做外部对照的方案**：它与 Search-R1 同源、使用 veRL、
Qwen2.5-3B、E5、Wiki-18、top-k=3，并明确优化多跳查询的信息增益与冗余。第一步应使用其公开的
`StepSearch-3B-Base` 权重接入我们现有的 Retriever，做 evaluation-only 兼容性 smoke 和固定
confirm256 对照；在看到外部策略确实改善 evidence-hit 前，不移植其 StePPO 或训练奖励。

## 1. 数据集前提澄清

`official-confirm256-v1` 的构建保证：

- 256 条真实 QA 标签；
- 与上游训练集、smoke、旧开发/确认集规范化重叠为 0；
- 固定 SHA 和七数据源配额；
- 使用真实 21,015,324-vector Wiki-18 Retriever 评测。

它**没有**保证每道题在当前 Retriever 下必然可检索，也没有保存 StepSearch 所需的
`search_keys` / `support_docs`。已有 question-as-query 审计结果是：

| 窗口 | 答案别名 lexical hit |
|---|---:|
| top-1 | 101/256（39.5%） |
| top-3 | 134/256（52.3%） |
| top-10 | 167/256（65.2%） |

这些数字是自动代理，不等价于“语料仅覆盖 65.2%”：未命中可能来自 query/embedding、Wikipedia
时间版本、答案别名规则，或多跳答案需要跨文档推导。项目的任务正是让 Agent 通过子问题分解、
查询改写和多轮检索提高可达证据率。后续结果必须同时按数据源和 question-as-query 可检索分层报告，
避免把语料覆盖、Retriever 召回和 Policy query 质量混为一谈。

## 2. 开源方法系统筛选

| 方法 | 核心机制 | 开源状态 | 与当前项目的关系 | 决策 |
|---|---|---|---|---|
| [StepSearch](https://github.com/Zillwang/StepSearch) | StePPO；查询关键词奖励；支持文档信息增益；重复文档惩罚；显式 plan/observation 循环 | Apache-2.0；代码、3B权重；Search-R1/veRL fork | 同 Qwen2.5-3B、E5、Wiki-18、top3；最贴近当前 query 瓶颈 | **P0：先评测公开权重** |
| [ReSearch/ReCall](https://github.com/Agent-RL/ReCall) | 文本推理引导何时/如何搜索；端到端 RL | 代码、模型、数据；深度定制 veRL，训练示例为 7B/4GPU | 能验证多轮反思，但没有 StepSearch 的明确 query/process supervision | P1 备选，不先移植 |
| [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) | 两阶段 outcome RL：先学调用搜索，再学最终答案 | MIT；训练/推理/权重/数据 | 第一阶段主要解决搜索激活；Aware-v2 已完成这一目标 | 不作为当前主改进 |
| [AutoRefine](https://arxiv.org/abs/2505.11277) | search 后显式 refine，过滤证据并规划后续 query；GRPO + retrieval reward | 论文可核验；未从论文页确认官方代码入口 | 同时针对 query 与证据整理，概念高度相关 | 无官方代码门禁前不作为“开源复现” |
| [EviNote-RAG](https://github.com/Da1yuqin/EviNoteRAG) | 搜索后生成支持证据笔记；entailment evidence-quality reward | Apache-2.0；完整代码/模型/数据 | 主要解决“命中证据仍答错”的 63 题 | P2，query 改善后再考虑 |
| [CaRR](https://github.com/THUDM/CaRR) | 原子 rubric + citation grounding + evidence chain reward | MIT；代码/轨迹/RL数据/权重 | 面向长上下文 deep search，需要外部 RM/API，规模过大 | 超出面试项目的适度改进边界 |

Search-R1 后续实证研究还发现 intermediate retrieval reward 的平均收益有限，而搜索引擎质量对
训练动力学影响很大。因此不能因为 StepSearch 有更多 reward 就直接认定有效；必须先用公开权重做
跨环境、同 Retriever 的行为验证。

## 3. StepSearch 源码兼容性审计

只读源码审计对象：`Zillwang/StepSearch` commit
`43215bab9118a4c8e01b15082f74b2aea30c1fc8`（下载到 `/tmp`，未加入项目 vendor）。

确认事项：

- 官方训练为 Qwen2.5-3B Base、Wiki-18/E5、retriever top-k=3、max_turns=5；
- 训练脚本使用 8 GPU、PPO/GAE critic、1120 steps，不符合本项目“最小单变量实验”资源边界；
- `step_information_gain` 将本轮检索文档与 golden `support_docs` 做 TF-IDF 相似度增量；
- 冗余惩罚按跨轮完全重复文档计算；
- `step_search_keys_match` 用 query 与 golden `search_keys` 的 token-F1 评分；
- 完整训练数据要求每条样本具有 `search_keys` 和严格对应当前 corpus 的 `support_docs`；
- 我们的 confirm256 只有 `ground_truth.target`，不能合法计算这两项训练 reward；
- 官方 Hugging Face 已公开 `Zill1/StepSearch-3B-Base`，可以先绕过训练做外部策略评测。

## 4. 推荐的最小开源方法实验

### Phase A：兼容性 smoke（8–16题）

- 模型：公开 `Zill1/StepSearch-3B-Base`；
- Retriever：复用现有 CPU Wiki-18 E5 服务，不修改 index、embedding 或 top-k；
- Prompt：采用 StepSearch 官方 plan/search/information/observation 协议；
- 只做 adapter，不修改模型权重；
- 物理 GPU0 禁用、GPU5 默认禁用；GPU 动作必须经过 preflight、tmux、`run_managed.sh`；
- 门禁：search tag 解析、Retriever success、非空 query、最多轮数、答案抽取、资源清理全部通过。

### Phase B：固定 confirm256 外部基线

如果 Phase A 通过，固定同一 Retriever/top-k/greedy 条件运行 confirm256。主要比较：

1. 题级 evidence-hit（主诊断指标）；
2. 搜索题中的 evidence-hit；
3. 多跳源 evidence-hit；
4. EM/F1（下游指标，不因搜索率上升自动成立）；
5. 搜索次数、重复查询、max-turn exhausted 和延迟。

该实验回答的是“StepSearch 公开策略能否在我们的 Search-R1 环境提高检索有效性”，不是与
Aware-v2 的严格同起点训练算法比较。外部模型在训练数据、prompt 和优化算法上不同，必须标记为
external open-source baseline。

### Phase C：只有外部策略显示 headroom 后才移植

若 StepSearch 相比 Aware-v2 明显提高 evidence-hit 且 EM 不退化，再选择一个最小组件移植：

- 优先移植显式 `plan -> search -> observation -> replan` 协议；或
- 只在有可靠 `support_docs/search_keys` 的训练子集上加入信息增益 reward；
- 保持现有 GRPO、FSDP、Retriever 和 top-k 不变，不直接复刻 8-GPU StePPO；
- 1-step smoke -> 5/10-step 单 seed -> 通过门禁后再做第二 seed。

若公开 StepSearch 在相同 Retriever 上 evidence-hit 也没有改善，则停止 query reward 训练，优先
审查语料/embedding 覆盖或按数据源报告不可检索层，不继续堆 reward。

## 5. 资源与安全边界

下一步尚未下载模型、尚未启动 GPU、尚未创建训练 Run。预计公开 3B 权重约数 GB；正式下载必须
放在 `/media/imc/data/project3-search-agent-rl/models/` 的新目录且拒绝覆盖。评测只使用重新确认
空闲的非零物理 GPU，使用全新 Run ID；任何不兼容或 smoke 失败保留证据并停止，不进入训练。

