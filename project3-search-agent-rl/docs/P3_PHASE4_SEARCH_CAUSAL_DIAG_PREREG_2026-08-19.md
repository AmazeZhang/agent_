# P3 Phase 4A：搜索失效因果诊断预注册（dated 2026-08-19）

**性质**：本文件为探索性诊断（Phase 4A）的**预注册**，预先固定四类诊断的定义、
指标、判断规则与固定输入。本阶段**不含任何 RL/SFT 训练**，不修改基线
checkpoint、预注册（`P3_PHASE2_PREREG_2026-08-16.md`）与正式结果
（final-confirm512 结论仅保留已提交的汇总结论，不重新运行、不逐题查看、
不用于调参）。所有诊断结果**不冒充、不替换预注册确认性结论**。

---

## 1. 实验边界（不变量）

- **诊断数据**：仅 official-confirm256-v1（dev256）。
  `datasets/searchr1-official-confirm256-v1/heldout.parquet` SHA256 =
  `ffebf468e756a673da267f5830cfc67f2e9c4dc44ec41c979a389c1efebfff60`
  （256 行；分源 nq 64 / hotpotqa 64 / popqa 32 / 2wikimultihopqa 32 /
  triviaqa 32 / musique 16 / bamboogle 16；标准答案字段
  `reward_model.ground_truth.target`）。
- **模型（固定三个）**：Qwen2.5-3B Base
  （`models/Qwen2.5-3B`）；正式 Step 300
  （`models/p3-formal-segment-100-300-gs300-merged-20260817b`）；官方 Search-R1 3B
  （`models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`）。
- **推理**：仅 GPU1，一律 `tmux + run_managed` 受管执行；**禁止 GPU0/GPU5；
  禁止任何六卡任务**。
- **解码固定**：seed 0、贪心解码、同一 tokenizer（Qwen2.5-3B Base）、同一
  问题顺序（dev256 文件行序）。
- **不改正式评测脚本**：诊断 2 使用独立新入口；正式
  `run_p3_eval_vllm_official.sh`/`run_p3_eval_vllm_official.py` 一字不改。

## 2. 诊断 1：Retriever 上限审计（纯 CPU）

对 dev256 **每题以原始 question 为 query** 调用真实 CPU Retriever
（`127.0.0.1:18080/retrieve`，Wiki-18 IndexFlatIP 21,015,324 向量），
分别请求 **Top-3 与 Top-10**：

- HTTP 检索成功率、超时（timeout=180 沿用）、错误计数；
- **lexical answer recall**：标准答案 alias（`ground_truth.target` 数组，
  归一化：小写、空白/标点折叠）作为子串出现在 Top-k 文档**拼接文本**中；
  报告 Top-1 / Top-3 / Top-10；按 7 分源统计；
- 文档得分分布（min/p50/p95/max）与检索延迟 p50/p95/p99；
- **声明**：lexical answer hit 仅为自动化代理指标，不表述为完整语义相关性。

**真实 search query 审计**（同一报告）：对三模型现有 dev256 episodes
（`runs/p3-eval-official-confirm256-{base3b,gs300*,official3b}-*`）中
`executed_search=true` 的步骤，分类：

| 类别 | 定义 |
|---|---|
| query 无效 | `<search>` 空 query / 格式错 / status=invalid_query |
| Retriever 成功但无答案证据 | status=success 且 Top-10 文档无标准答案 alias |
| Retriever 返回包含答案的证据 | status=success 且 Top-10 文档含标准答案 alias |

（Step300 的 dev256 episodes 若不存在，先经 GPU1 受管标准评测补跑一次，见 §3 备注。）

**判断规则**：Top-10 answer recall 低 → 优先修 Retriever/embedding/语料，
不先训 reward。

## 3. 诊断 2：反事实证据注入（GPU1 受管推理）

**独立新入口**（`scripts/run_p3_eval_counterfactual*.py/.sh`），不修改正式脚本。
同一批 dev256、三个模型 × 四个固定条件 = 12 个 run：

| 条件 | 定义 |
|---|---|
| no-evidence | 仅 base prompt（系统 + 问题），模型直接生成 |
| real-top3 | question-as-query 检索到的真实 Top-3 文档 |
| oracle-retrieved | 若 Top-10（question-as-query）中存在含标准答案 alias 的文档 → 注入该文档；无命中题**单独标记**，不伪造证据 |
| shuffled-evidence | 由其他题确定性置换来的 Top-3 文档（固定置换：第 i 题取第 (i+17) mod 256 题的 real Top-3） |

**evidence 条件统一构造**（固定模板，三个模型逐字一致）：

```
assistant: <search>固定query</search>
user: <information>固定文档</information>
assistant: <由模型继续生成 <answer>...>
```

（no-evidence 不含中间两行，其余完全一致。fixed query = 题目原始 question。）

- 四条件间问题、顺序、tokenizer、生成参数（seed 0 / greedy / max_new 256）
  完全一致；
- 评分：skyRL EM 语义（`ground_truth.target` alias 匹配，与正式评测同一
  归一化）、compliance（输出含 `<answer>`）；
- 报告：各条件 EM/compliance；**模型内**两两配对 McNemar 精确双侧 p；
  delta 报告 `real-top3 − no-evidence`、`oracle − no-evidence`、
  `real-top3 − shuffled`；分别统计「证据含答案」与「证据不含答案」题目的 EM；
- **声明**：本诊断是受控反事实探测，不称为确认性实验；oracle 条件仅用于
  上限估计，不声称可部署。

**判断规则**：oracle 显著提升而 real-top3 无提升 → 主要是 Retriever/query
问题；real-top3 已含答案仍不提升、oracle 也提升有限 → 阅读/上下文利用问题。

## 4. 诊断 3：搜索选择偏差（纯 CPU）

输入：三模型 dev256 episodes（Base、官方 Search-R1 已有；**Step300 若缺则
GPU1 受管补跑标准评测** `run_p3_eval_vllm_official.sh` 生成 episodes，
仅 dev256、不进 final-confirm512）：

- 搜索题 vs 不搜索题：分源构成、prompt 长度（token）、多跳源（2wiki/
  hotpotqa/musique）占比；
- **Base 在「官方模型选择搜索」的题目子集上的直接作答 EM**；
- **官方模型搜索子集在 oracle evidence 条件下的 EM**（复用诊断 2 的
  oracle run，对官方模型搜索子集切片）；
- 判断 `search→correct=0` 的归因：只在难题搜索（选择偏差）／Retriever 无
  证据（检索失败）／模型不会使用证据（利用失败）；
- **声明**：观察到搜索题更难只能说明选择偏差，不能直接证明搜索无效。

**判断规则**：evidence 可用且能提升、但策略不搜索 → 主要是奖励与信用分配
问题。

## 5. 诊断 4：候选奖励离线模拟（纯 CPU，不训练）

基于历史 rollout（训练 `runs/*/rollouts/*.jsonl(.audit.jsonl)`，group_n=5）
与诊断 1/2 结果，实现**可审计 reward simulator**：

```
R = R_answer
  + α · valid_retrieval
  + β · evidence_hit
  + γ · searched_and_correct_and_evidence_hit
  − λ · invalid_or_error
  − μ · redundant_search_count
```

预注册定义（诊断报告内展开）：

- `R_answer`：当前语义（EM=1.0 / format=0.1 / 无=0.0）；**候选评估
  format_score ∈ {0.1, 0.05, 0.0}**（含「降低当前 0.1」的方案）；
- `valid_retrieval`：执行 search 且 status=success 且非 error → 1 否则 0；
- `evidence_hit`：search 返回文档（Top-10 窗口）含标准答案 alias → 1 否则 0；
- `searched_and_correct_and_evidence_hit`：search 且 evidence_hit 且最终 EM=1
  → 1 否则 0；
- `invalid_or_error`：status ∈ {invalid_query, api_error, no_results} 或
  error_observation → 1 否则 0；
- `redundant_search_count`：episode 内 max(0, 搜索次数 − 1)。

**防 reward hacking 测试轨迹**（必须全部通过）：

1. 不搜索直接答对；
2. 有效搜索后答对（证据含答案）；
3. 搜索到相关证据但答错；
4. 搜索无关文档（证据不命中）后答对/答错；
5. invalid query；
6. 重复刷搜索（同 query 多次）；
7. 把标准答案直接写进 query；
8. 只输出正确格式但错误答案。

**硬性约束**（预注册判定规则）：

```
有效证据支持下搜索并答对 > 不搜索直接答对 > 有格式但答错 > 无效/重复搜索
```

且**不得**仅凭「调用了搜索」给予足以刷分的奖励（α 上限约束，见结果报告）。
候选若违反排序或存在可刷分路径 → 判为不合格。

**历史轨迹评估**（group_n=5，按 rollout 组）：每个候选系数集的 reward 分布、
组内方差、全同奖励组比例、搜索/不搜索轨迹的 advantage 方向（GRPO 组内
归一化下）、潜在作弊路径清单。**只推荐系数范围，不冻结最终系数**。

## 6. GiGPO 接入审计（只读代码核查）

只读核查当前 verl-agent fork（pin `20bd331b…` + patches 0001–0006）中 GiGPO
的真实实现与配置入口（`algorithm.gigpo.*` 等），明确：

- episode-level group advantage 现行为；
- step-level / group-in-group advantage（`gigpo_mode`、`step_advantage_w`、
  `gigpo_enable_similarity`、`gigpo_similarity_thresh`）现行为；
- 多轮 search/answer record 如何分配信用；
- Observation token 是否被 policy loss mask 排除（`response_mask`/`loss_mask`）；
- 与当前 GRPO 相比需要修改的配置与代码点；
- 能否保持现有 FSDP/offload/vLLM 架构。

输出 **Search-aware GRPO** 与 **Search-aware GiGPO** 两套候选方案（仅设计，
**不实施训练**）。

## 7. 决策规则（诊断完成后按序判定）

| 路径 | 结论 |
|---|---|
| Top-10 answer recall 低 | 优先修 Retriever、embedding 或语料，不先训 reward |
| oracle 显著提升、real-top3 无提升 | 主要是 Retriever/query 问题 |
| real-top3 含答案仍不提升、oracle 提升有限 | 主要是阅读/上下文利用问题 |
| evidence 可用且能提升、但策略不搜索 | 主要是奖励与信用分配问题 |
| 多个问题并存 | 明确优先级与最小改动实验 |

## 8. 交付物清单

1. 本预注册（先 commit/push，再执行诊断）；
2. Retriever recall 结果 JSON/Markdown；
3. 反事实推理脚本 + 结果（12 run episodes + 汇总表）；
4. 搜索选择偏差分析；
5. reward simulator + 防作弊测试；
6. GiGPO 接入设计；
7. `docs/P3_PHASE4_SEARCH_CAUSAL_DIAG_RESULT_2026-08-19.md` 汇总（含下一步
   最小训练实验：起始模型、数据集、reward 公式与系数范围、GRPO/GiGPO、
   步数/batch/GPU 预算、smoke 门禁与停止条件——**仅方案，不实施**）。

**完成诊断后停止**：不自动启动 SFT、GRPO、GiGPO 或任何六卡训练，等待单独
批准。
