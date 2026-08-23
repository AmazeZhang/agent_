# P3 v2 Step5 与既有对照线的严格逐题配对统计（2026-08-23）

## 0. 结论先行

在 official-confirm256-v1 上对已有单次 greedy 运行（相同题集、相同数据 SHA、
逐题按 `question_id` 严格配对）做精确 McNemar 检验：

| 配对（目标 vs 对照） | 1→1 | 0→0 | 对照对/目标错 | 对照错/目标对 | 净增 | 精确双侧 McNemar p |
|---|---|---|---|---|---|---|
| v2 Step5 vs Step0 | 50 | 163 | 15 | 28 | **+13** | 0.06599 |
| v2 Step5 vs GRPO10 | 53 | 157 | 21 | 25 | **+4** | 0.65874 |
| v2 Step5 vs GiGPO10 | 47 | 156 | 22 | 31 | **+9** | 0.27168 |
| GRPO10 vs Step0 | 46 | 163 | 19 | 28 | +9 | 0.24296 |

- 四组配对 **均不达 p<0.05**。v2 Step5 相对 Step0 的 +13 最接近（p=0.066），
  与 GRPO10 的 +4 完全在噪声内（p=0.66）。
- 因此"v2 Step5 是该评测集第一名"的表述维持为**方向性正面、当前第一名**，
  **不是显著性声明**（单次 greedy 运行，区间重叠，见
  `P3_V2_BEHAVIOR_EVAL_REPORT_2026-08-23.md` §6）。
- 本报告只做配对描述统计，**不用 Wilson 区间重叠代替配对检验**；上述 p 值
  全部来自逐题配对的精确双侧 McNemar（二项双侧）。

## 1. 配对完整性（integrity）

- 四个运行的题集一致：`question_id` 0..255，排序一致、无重复、无缺失。
- 四个运行的 `data_files.sha256` 全部相同（`ffebf468…`，heldout.parquet）。
- 运行元数据：全部 vLLM 原生 greedy（temperature=0.0）、seed=0、max_steps=4、
  history=4、topk=3、GPU1-only；v2 Step5 为 `p3-eval-v2-behavior-gs5-confirm256-20260823a`，
  Step0/GRPO10/GiGPO10 为 2026-08-20 的 clean 对照运行。

## 2. 行为分解（题级，n=256）

| 运行 | EM | 搜索 | 作答(offline) | 作答(env提交) | 搜索且对 | 未搜索且对 | 搜索但无作答(offline) | 步/题 | invalid\* |
|---|---|---|---|---|---|---|---|---|---|
| v2 Step5 | 78 | 233 | 232 | 189 | 69 | 9 | 24 | 2.80 | 2 |
| Step0 | 65 | 180 | 254 | 211 | 41 | 24 | 2 | 2.32 | 0 |
| GRPO10 | 74 | 161 | 224 | 224 | 46 | 28 | 32 | 2.13 | 0 |
| GiGPO10 | 69 | 140 | 233 | 233 | 32 | 37 | 23 | 1.95 | 0 |

\* invalid = episodes 级 `invalid_query`（检索失败/api 错误，`episodes.jsonl` 判定）；
v2 Step5 的 2 题（q5、q206）均已有作答，不在 §4 的 24 个未作答题内；
`results.json` 的 `invalid_search_calls` 是更窄的计数口径，与这里不同。
上表全部数值与 `gates/p3_v2_paired_stats_20260823.json` 逐项一致
（`scripts/p3_v2_paired_stats.py` 可复算）。

- **双作答口径**：`answered_offline`（results.json 合规口径，从拼接 raw
  action 抽取 `<answer>`——含混合回合中的草稿）与 `answered_env_committed`
  （真实提交 `<answer>` 回合，环境终态）。v2 Step5 的 232 vs 189 差异 = 43 题
  起草了答案但从未提交（混合回合、projection 选择 search）——行为信号，
  反事实评测中检验。
- **混合回合**（raw action 同时含 `<search>` 与 `<answer>`）：v2 Step5 169
  回合（164 题）、Step0 169 回合（165 题）——总数巧合相等但逐题分布不同
  （已核验非产物），GRPO10/GiGPO10 为 0。

## 3. 逐源不一致计数（discordants）

| 源 | Step0 对/v2 错 | Step0 错/v2 对 | GRPO10 对/v2 错 | GRPO10 错/v2 对 |
|---|---|---|---|---|
| 2wikimultihopqa | 2 | 3 | 3 | 3 |
| bamboogle | 0 | 2 | 1 | 1 |
| hotpotqa | 2 | 7 | 5 | 3 |
| musique | 1 | 0 | 1 | 1 |
| nq | 5 | 5 | 6 | 4 |
| popqa | 3 | 5 | 2 | 7 |
| triviaqa | 2 | 6 | 3 | 6 |
| 合计 | 15 | 28 | 21 | 25 |

v2 Step5 相对 Step0 的净增主要来自 hotpotqa（+5）、triviaqa（+4）、popqa（+2）、
bamboogle（+2）、2wiki（+1）；nq 打平；musique 净失 1。相对 GRPO10 的 +4 中
triviaqa（+3）、popqa（+5）为正，nq（−2）、hotpotqa（−2）为负。

## 4. v2 Step5 未作答题（24/256）终止原因分类

- 24 题全部为 **max_steps_exhausted**（4 步用尽未提交 `<answer>`），
  **invalid_query = 0**（无检索失败/api 错误/空查询导致的失败；运行内 2 个
  invalid 题 q5/q206 均已作答，不在此列）。
- 24 题的 raw action 中 **0 题含 `<answer>` 草稿**（`<answer>` 出现即被
  offline 口径计入），全部 4 步都在搜索/思考，最终未收敛到提交。
- 另注意：运行内另有 43 题 offline 有 `<answer>` 草稿但从未提交
  （`drafted_never_committed`，env 提交 189 = 232 − 43）—— 与这 24 题不同集。
- 搜索侧：24 题全部执行过搜索；未作答与"搜索后未能收敛到提交"对应，
  是行为变化（搜索更积极、部分轨迹未收敛），不是环境故障。

## 5. 声明边界

- 单次 greedy 运行、固定 seed；配对检验控制的是"逐题结果差"的偶然性，
  不控制"运行间其他条件"（同模型同配置，仅解码固定）。
- 所有 p 值如实报告；**不因 p>0.05 调参、不因 p<0.05 过度声称**；
  证据因果性不由本报告判定（见反事实检索评测
  `P3_V2_COUNTERFACTUAL_RETRIEVAL_2026-08-23.md`）。
- 产物：`gates/p3_v2_paired_stats_20260823.json`（含逐源不一致、行为分解、
  24 题未作答明细）；CPU-only 复算：`scripts/p3_v2_paired_stats.py`。
