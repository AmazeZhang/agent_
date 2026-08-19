# P3 Phase 4A 搜索失效因果诊断结果（Search Causal Diag）

日期：2026-08-19 ｜ 预注册：`docs/P3_PHASE4_SEARCH_CAUSAL_DIAG_PREREG_2026-08-19.md`（commit 751e8d9）
本文件为 Phase 4A 诊断汇总。诊断 1–4 全部为只读/离线分析 + 固定开发集推理，**不修改任何基线 checkpoint、预注册与正式结果**。

## 0. 固定实验边界（复述预注册 §1，全部遵守）

- 数据：仅 `official-confirm256-v1`（dev256，SHA `ffebf468…bfff60`，256 行，nq 64 / hotpotqa 64 / popqa 32 / 2wikimultihopqa 32 / triviaqa 32 / musique 16 / bamboogle 16）
- 模型：① Qwen2.5-3B Base；② 正式 Step 300（`models/p3-formal-segment-100-300-gs300-merged-20260817b`）；③ 官方 Search-R1 3B（`models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`）
- GPU 推理仅 GPU1（tmux + run_managed）；固定 seed=0、贪心解码、同一 tokenizer（Base）、同一问题顺序

## 1. 诊断 1：Retriever 上限审计（完成）

脚本：`scripts/p3_diag1_retriever_audit.py`；结果：`gates/p3_diag1_retriever_audit_20260819.json`（commit e7f1ca8）

### Part A：question-as-query（n=256）

- **HTTP**：Top-3/Top-10 均 256/256 成功，无超时/错误；延迟 p50=6.15s / p95=6.33s / p99=6.37s（单客户端串行；服务器 `max_concurrent_queries=64`，IndexFlatIP 768 维，21,015,324 向量）
- **lexical answer recall**（NFKC+casefold+去空白标点，答案别名子串，**自动化代理指标**）：
  - 总体：Top-1 **101/256（39.5%）**、Top-3 **134/256（52.3%）**、Top-10 **167/256（65.2%）**
  - 分源（Top-1/Top-3/Top-10）：nq 38/47/56；hotpotqa 18/27/35；popqa 13/21/25；2wikimultihopqa 7/10/16；triviaqa 22/25/27；musique 2/2/4；bamboogle 1/2/4
  - 弱源：musique/bamboogle（Top-10 均仅 4/16）、2wikimultihopqa（16/32）——多跳问题答案不在单文档子串中是结构性的（答案需跨文档推理）
- **文档得分分布**（Top-10）：min=0.7637 / p50=0.8284 / p95=0.8727 / max=0.9260——得分带窄且高，score 本身不可作为相关性强弱判别
- 结论（prereg §7 决策规则 1）：**Top-10 lexical 覆盖率 65.2% 为 Retriever 上限的代理下界**；对 34.8% 未命中问题，oracle 注入无法命中（见诊断 2 oracle 子集），检索链路对这些题不可能提供答案证据——多跳子集（musique/bamboogle/2wiki）需要查询重写或文档拼接，纯 question-as-query 的 wiki-18 语料难以覆盖

### Part B：真实 search query 三分类（dev256 episodes）

| 模型 | search_steps | invalid/failed | success | success 含答案证据 | success 无证据 |
|---|---|---|---|---|---|
| Base（Qwen2.5-3B） | 124 | **114（92%）** | 10 | 5 | 5 |
| SearchR1（官方） | 40 | 0 | 40 | **24（60%）** | 16 |
| Step300 | **0** | — | — | — | — |

- Base：114 次 `invalid_query` 且空 `<search></search>`（empty_query=114）——**Base 的搜索行为几乎全部是空查询格式错误**，与最终确认评测「Base 搜索 242 次全部答错」一致：Base 不会生成有效搜索查询
- SearchR1：40 次全部 success，60% 的检索结果含答案证据（lexical 代理）→ 官方模型会搜索且检索常能带回证据，但最终 EM 仍低（见诊断 3：证据可用性与利用问题）
- **Step300：0 次搜索**——正式训练 300 步后搜索行为完全消失（GPU1 受管补跑确认），与诊断 3/4 互相印证（训练把搜索优化掉了）

## 2. 诊断 2：反事实证据注入（12 runs，完成）

脚本：`scripts/run_p3_eval_counterfactual.py`（commit 6ddc16b）+ `scripts/analyze_p3_diag2_counterfactual.py`（commit 5e5b045）；结果：`gates/p3_diag2_counterfactual_analysis_20260819.json`（本轮 commit）
全部 run 经 GPU1 受管执行（run_managed 门禁，exit_code=0 全过）；共享同一 evidence cache（`diag_cache/dev256_top10_docs.json`，question-as-query Top-10，256/256 检索成功，SHA 校验通过）。**非确认性实验**（prereg §3），但效应量与配对显著性一致。

### 2.1 总表（EM，n=256；配对 McNemar 精确双侧）

| 模型 | no-evidence | real-top3 | Δ(real−no) | oracle | Δ(oracle−no) | shuffled | Δ(real−shuff) |
|---|---|---|---|---|---|---|---|
| Base | 22（8.6%） | 30（11.7%） | +3.1p, p=0.29 | **52（20.3%）** | +11.7p, **p=0.0002** | 2（0.8%） | 29:1, **p<0.0001** |
| gs300（Step300） | 48（18.8%） | 68（26.6%） | +7.8p, **p=0.019** | **105（41.0%）** | +22.2p, **p<0.0001** | 27（10.5%） | 47:6, **p<0.0001** |
| SearchR1（官方） | 34（13.3%） | 65（25.4%） | +12.1p, **p=0.0001** | **97（37.9%）** | +24.6p, **p<0.0001** | 22（8.6%） | 49:6, **p<0.0001** |

- **oracle 证据注入全模型显著提升**（+11.7 ~ +24.6p，McNemar p≤0.0002）→ **模型会用证据**：证据到位时 EM 接近翻倍
- **real-top3 vs shuffled 全显著**（正向差 29~49 对反向 1~6）→ 提升源于证据与问题/答案的相关性，排除「随便给文档」的混淆
- Base 的 real-top3 提升不显著（p=0.29）但 oracle 显著 → Base 的搜索链路失效（不会生成有效 query，诊断 1B：92% 空查询），但给足证据它仍能用
- compliance：gs300 全条件 100%；Base 48.8~68.4%（格式能力弱）

### 2.2 oracle 子集（167/256 题 Top-10 命中答案证据，与诊断 1 的 65.2% 一致）

| 模型 | 命中题 oracle EM | 同题 no-evidence EM | McNemar p | 未命中题（89）oracle EM |
|---|---|---|---|---|
| Base | 51/167（30.5%） | 20/167（12.0%） | p=0.0001 | 1/89（1.1%） |
| gs300 | 97/167（58.1%） | 39/167（23.4%） | p<0.0001 | 8/89（9.0%） |
| SearchR1 | 93/167（55.7%） | 29/167（17.4%） | p<0.0001 | 4/89（4.5%） |

- 证据命中题上 oracle 比无证据高 2.5~3.4 倍 → **阅读/上下文利用不是主要瓶颈**
- 未命中题 oracle 仅 1.1~9.0% → **Retriever Top-10 覆盖 65.2% 是硬上限**：34.8% 的题（musique/bamboogle/2wiki 为主）检索链路无法提供答案证据，与诊断 1 分源结果互相印证

## 3. 诊断 3：搜索选择偏差（完成）

脚本：`scripts/analyze_p3_diag3_selection.py`（commit d27d146）；结果：`gates/p3_diag3_selection_20260819.json`（本轮 commit）

### 3.1 搜索 vs 不搜索子集（dev256）

| 模型 | 搜索题数 | 搜索题 EM | 不搜索题 EM | 搜索题多跳占比 | 不搜索题多跳占比 | 问题长度均值 |
|---|---|---|---|---|---|---|
| Base | 124 | **0/124** | 20/132（15.2%） | 42.7% | 45.0% | 16.1 vs 15.9 |
| SearchR1 | 40 | **0/40** | 32/216（14.8%） | **55.0%** | 42.0% | 16.5 vs 15.9 |
| **Step300** | **0** | — | 49/256（19.1%） | — | 43.8% | 16.0 |

### 3.2 关键事实

1. **Step300 完全停止搜索**：正式训练 300 步后 dev256 上 0 次搜索，全部凭记忆作答（EM 19.1%，全条件 compliance 100%）。结合诊断 4（C0 基线 T2==T1，搜索轨迹 advantage 全负）→ **当前 reward/credit 体系训练掉了搜索行为**：搜索在组内比记忆直答更差时被 GRPO 逐步淘汰
2. **三模型实际搜索题 EM 全为 0**（Base 0/124、SearchR1 0/40）：实际 rollout 中搜索从没带来一次答对
3. **选择偏差存在但温和**：SearchR1 搜索题多跳占比 55% vs 42%（搜索选向难题）；Base 搜索子集在无证据条件下 10%（Base 直答 EM 4/40，与总体 8.6% 相当）
4. **归因关键对照（诊断 3 + 2 拼接）**：
   - SearchR1 搜索过的 40 题中，25 题 Top-10 含答案证据 → 这 25 题在 oracle 条件下 **15/25（60%）答对**
   - Base 搜索过的 124 题中 79 题含证据 → oracle 条件下 25/79（31.6%）
   - 即：**搜索题本身可答（证据在语料中），模型实际搜索却没答对 → 问题在「实际检索返回的证据不够/未命中」+「训练压制搜索」，而非「题不会做」**

## 4. 诊断 4：候选奖励离线模拟（已完成）

脚本：`scripts/p3_diag4_reward_simulator.py`（commit a10c50e）；结果：`gates/p3_diag4_reward_sim_20260819.json`

### 4.1 公式与候选系数

```
R = R_answer + α·valid_retrieval + β·evidence_hit + γ·searched_and_correct_and_evidence_hit − λ·invalid_or_error − μ·redundant_search_count
```
R_answer 当前语义：EM=1.0 / format=0.1 / 无=0.0；候选 format_score ∈ {0.1, 0.05, 0.0}。硬性排序约束：`有效证据支持下搜索并答对 > 不搜索直接答对 > 有格式但答错 > 无效/重复搜索`；α ≤ 0.05（不得仅凭「调用了搜索」给予足以刷分的奖励）；μ ≥ α+β+γ（重复刷搜索不能净赚）。

| 候选 | format | α | β | γ | λ | μ | T2−T1 | 检查 |
|---|---|---|---|---|---|---|---|---|
| C0-current-baseline | 0.1 | 0 | 0 | 0 | 0 | 0 | 0（T2==T1） | **不满足 T2>T1（这正是问题所在）** |
| C1-conservative | 0.1 | 0.02 | 0.05 | 0.10 | 0.10 | 0.20 | +0.17 | T4=1.02>T1 ✗ |
| C2-moderate | 0.1 | 0.02 | 0.10 | 0.20 | 0.20 | 0.35 | +0.32 | T4=1.02>T1 ✗ |
| C3-format-0.05 | 0.05 | 0.02 | 0.05 | 0.10 | 0.10 | 0.20 | +0.17 | T4=1.02>T1 ✗ |
| C4-format-0.0 | 0.0 | 0.02 | 0.05 | 0.10 | 0.10 | 0.20 | +0.17 | T4=1.02>T1 ✗ |
| C5-evidence-driven-a0 | 0.1 | 0 | 0.15 | 0.30 | 0.20 | 0.45 | +0.45 | T4=1.0≤1.0 ✓；T7=1.45>T1 ✗ |
| C6-spam-averse | 0.1 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | +0.35 | T4=1.05>T1 ✗ |

8 条防 hack 轨迹断言（T1 直答对 / T2 有效搜索+证据答对 / T3 相关证据答错 / T4 无关文档凭记忆答对 / T5 invalid query / T6 重复搜索刷分 / T7 标准答案写进 query / T8 只格式正确答错）。**勘误 2026-08-19（审阅反馈）：「C1–C6 全部 PASS」表述不成立**：

- **T7 答案泄漏没有通过**：任何候选里 T7==T2 by construction（query 含答案 + 检索命中 → 触发与 T2 相同的有效+证据+答对奖励），奖励层面不可区分，必须新增 **T7 ≤ T1 硬断言**
- **C1/C2/C3/C4/C6 中 α>0 使无关搜索也可能获得奖励**：T4（无关文档凭记忆答对）= 1.0 + α > T1 = 1.0，违反 **T4 ≤ T1 硬断言**（α>0 时"调用了搜索"本身就有正收益 → 可被刷分）
- C0 基线失败（T2==T1）为机制性发现：当前奖励不给证据支撑搜索任何额外信用

**冻结的 v1 公式（见 §8）正是为满足这两个硬断言而设计**：α=0（T4==1.0≤T1 通过）+ `new_answer_leak_in_query` 惩罚（T7 被清零证据奖励并额外 −0.20 → 0.80<T1 通过），其余断言 T2>T1（1.45>1.0）、T1>T3（1.0>0.25）、T3>T5（0.25>−0.20）、T8<T1（0.1<1.0）在 v1 下全部满足。**勘误 2026-08-19（工程实现确认）：T6 值应为 1.00 而非 0.55**——diag T6（2 次搜索、证据命中、答对）在 episode 级模型下 = 1.0 + 0.15 + 0.30 − 0.45 = 1.00 == T1（相等，guard 成立）；0.55 是「2 次无价值搜索刷分」情形（1.0 − 0.45）。step 归因实现通过「**冗余（第 2 次起）搜索步不获得 evidence 加分**」保证 T6≤T1 对任意多搜索组合普遍成立（详见 §9.1 语义澄清）。

### 4.2 历史轨迹评估（seg 0-100，n=29,838，off-policy）

- 历史搜索质量：searched 4,524 / valid 1,297（28.7%）/ evidence-hit 585（12.9%）/ correct_with_evidence_search 41（0.9%）；redundant_sum 0
- 所有 shaping 候选下 search_adv 均为负（−0.36 ~ −0.48）、nosearch_adv 为正（+0.06 ~ +0.09）→ **shaping 在历史 off-policy 轨迹上不能逆转搜索劣势**（γ 只触发 41/4,524）
- C0 基线：同组全同奖励组比例 82.5%；C1–C6 降到 ~51–55%（组内方差略升）——shaping 增加组内信号但有限
- 勘误 2026-08-19（审阅反馈）：原"推荐系数范围、不冻结"被冻结为 **Search-aware GRPO v1 公式**（§8）：在 C5-evidence-driven（α=0）基础上新增 `new_answer_leak_in_query` 惩罚，使 T4≤T1 与 T7≤T1 两个硬断言同时成立（§4.1 勘误）

## 5. GiGPO 接入审计（只读，file:line 已核实）

### 5.1 fork 内真实实现

- `vendor/verl-agent/gigpo/core_gigpo.py:138-171` `compute_gigpo_outcome_advantage` = episode_norm_reward（按 uid 组=同 prompt 的 n 个 traj，含跨 step 去重 seen_pairs）+ `step_advantage_w` × step_norm_reward（按 build_step_group 聚类）
- `:243-331` `build_step_group`：同一 uid 组内 anchor_obs 精确相同成一组；`gigpo_enable_similarity` 时 SequenceMatcher ≥0.95 文本相似并入组
- `:334-384` step_norm_reward：组内归一（mode 切换是否除 std）；`:87-132` compute_step_discounted_returns
- config 入口：`verl/trainer/config/ppo_trainer.yaml:250-254`（gigpo 配置块）；ray_trainer.py:344-358（GiGPO 分支）、:243（compute_advantage 签名）
- **episode 项权重硬编码 1.0**（GraphGPO recipe 才有 episode_advantage_w）——若需调整要改 core_gigpo.py

### 5.2 search 场景下 GiGPO step 分组语义（env manager anchor 已核实）

- `agent_system/environments/env_manager.py:62-78,148-164`：reset 时 anchor=obs（同组 5 traj 相同）；step 时 anchor=next_obs（依检索结果分化）
- 结论（勘误 2026-08-19，审阅反馈）：**step 0 全组 5 条共享同一 anchor → 1 个 step 组**；step ≥ 1 的检索 Observation 通常彼此不同（不同 query/结果分化）→ **step group 为单例**
- **单例组归一化 advantage 近似 0，不提供有效 step 相对信用** → 当前 fork 内 GiGPO 直接开启的预期收益有限（episode 项 ≈ GRPO，step 项退化为无信号）
- **GiGPO 推迟到 nested rollout 版本**：能够对同一 post-retrieval state 采样多个后续 action（如 group-in-group 或按 step 内多采样）时，step 组内才有足够样本做相对归一；届时再评估

### 5.3 Observation token 与损失

- `dp_actor.py:317-372`：multi_turn=False → response_mask = attention_mask[:, -response_length:] → **<information> observation 是 prompt token，构造性排除于 policy loss**。这是正确语义：Observation 是环境条件上下文，policy loss 只作用于模型生成的 action/answer token；模型通过 answer token 的梯度学习利用证据。**不将 Observation 纳入 loss（勘误 2026-08-19，审阅反馈）**
- 当前信用分配：`agent_system/reward_manager/episode.py:20-96` 把完整 episode reward 放在**每条 step record 最后一个有效 response token** 上 → 搜索动作与答案动作同分 → 这是"搜索无用"机制根源之一（诊断 2/3 数据将给出证据强度）

### 5.4 架构可行性

GiGPO 是 driver-side 纯 advantage 计算替换（critic-free），**FSDP/offload/vLLM/rollout 架构不变**；开启只需 ppo_trainer.yaml 配置 + adv_estimator 分支；可保持与当前 GRPO 完全相同的训练拓扑。

## 6. 两套候选方案（仅设计，未实施）

### 方案 A：Search-aware GRPO（最小改动）

- **Reward**：**冻结的 Search-aware GRPO v1 公式**（勘误 2026-08-19 冻结，见 §8）：`R = R_answer + 0.15·evidence_hit + 0.30·searched_and_correct_and_evidence_hit − 0.20·invalid_or_error − 0.45·redundant_search_count − 0.20·new_answer_leak_in_query`；format_score=0.1（本轮不改格式奖励）；α=0（无关搜索本身不加分）
- **实现点**（verl-agent fork）：
  1. RewardManager 需按 step 产出 shaping 分量（evidence/sce/invalid/redundant/answer-leak），挂到对应 step record（仿 episode.py 但按 step；R_answer 只在终止 answer step）
  2. 现有 `token_level_rewards` 广播语义保留：episode 分量仍在最后 token；shaping 分量放该 step 最后 token
  3. loss mask 不变（Observation 继续排除于 policy loss——勘误①，正确语义：Observation 是条件上下文，模型通过 answer token 梯度学习利用证据）
- **优点**：改动最小（reward 层 + 配置）；GRPO 语义完全不动；可直接对比当前基线
- **缺点**：仍不显式训练"读完证据再作答"（靠 answer-token 梯度间接学）；`new_answer_leak_in_query` 依赖 alias 归一化检测规则（§8 规则 3），规则外泄漏需审计字段事后检查

### 方案 B：Search-aware GiGPO（**推迟，本轮不实施**）

- 勘误 2026-08-19（审阅反馈）：当前 fork 内 GiGPO 在 search 场景 step≥1 的检索 Observation 通常不同 → step group 为单例 → 单例组归一化 advantage≈0，无有效 step 相对信用 → 直接开启预期收益有限
- **前置条件：nested rollout 版本**——能对同一 post-retrieval state 采样多个后续 action（step 组内多样本）时再启用；届时在 shaping reward 之上：
  1. `adv_estimator: gigpo` + `step_advantage_w: 0.3~0.5`（起步 0.3，避免 step 项噪声主导）
  2. 修改 `core_gigpo.py` 增加 `episode_advantage_w` 配置（当前硬编码 1.0）
- **本轮不实施**：adv_estimator 保持 GRPO；Observation 始终排除于 policy loss（勘误①）

## 7. 决策规则应用（prereg §7）

| 规则 | 证据 | 结论 |
|---|---|---|
| Top-10 覆盖率低 → 修 Retriever/embedding/语料 | Top-10 lexical 65.2%；未命中 89 题 oracle 仅 1.1~9.0% | **Retriever 覆盖是硬上限（34.8% 题），但非唯一瓶颈**。弱源全部为多跳（musique/bamboogle 4/16、2wiki 16/32）；wiki-18 语料与 e5 embedding 已固定，重嵌入/换语料成本高、收益不确定 → 降级为 P2 |
| oracle 提升 real-top3 无 → Retriever/query 问题 | oracle vs real-top3：gs300 +14.4p、SearchR1 +12.5p、Base +8.6p；Top-3 lexical 仅 52.3% | **检索召回不足（尤其多跳题 question-as-query）是 P1 问题**：real-top3 常带不回答案文档 |
| real-top3 含答案仍不提升、oracle 有限 → 阅读/上下文利用 | oracle 命中题上 gs300 58.1% vs 23.4%、SearchR1 55.7% vs 17.4%（p<0.0001） | **阅读/上下文利用不是瓶颈**——证据到位时模型会用（2.5~3.4 倍提升）|
| evidence 可用能提升但不搜索 → 奖励与信用分配 | Step300 搜索 0 次；诊断 4 C0 基线 T2==T1（搜索+证据答对与记忆直答同分）、搜索轨迹 advantage 全负（−0.36~−0.48） | **P0 问题：当前 reward 不给搜索任何信用，GRPO 训练掉了搜索行为**。最小干预 = 诊断 4 的 shaping 系数 + 信用分配 |
| 多问题并存 → 优先级 | 上述 | **P0 训练侧（reward shaping + credit）→ P1 检索侧（query 改写/多查询）→ P2 语料/embedding** |

### 因果链总述（四诊断拼接）

1. 训练（Step300）后模型完全放弃搜索（诊断 3），因为当前奖励系统给「搜索+证据+答对」与「记忆直答对」相同奖励（诊断 4 C0：T2==T1），而搜索轨迹在组内 advantage 全负 → GRPO 优化掉搜索
2. 这不是「模型不会用证据」：证据注入（oracle）时 gs300 命中子集 58.1%（诊断 2）
3. 也不是「题不会做」：SearchR1 实际搜索的 40 题中 25 题证据在语料里、oracle 下 60% 可答对（诊断 3）
4. 真实检索能带回应答证据但有限（Top-3 52.3% / Top-10 65.2%，诊断 1），多跳题尤其差——检索侧是第二约束

## 8. 下一步最小训练实验方案（仅设计，未实施，待单独批准）

**目标**：验证「reward shaping + 信用分配能恢复并保留搜索行为，并让搜索真正贡献 EM」的最小实验。

| 项 | 方案 |
|---|---|
| 起始模型 | **Qwen2.5-3B Base**（干净对照；Step300 已压死搜索，作为基线参照而非起点）。备选：Step300 继续训练（保留记忆 18.8% + oracle 利用 58%，但需先解除搜索惩罚） |
| 数据集 | `datasets/searchr1-upstream/train.parquet`（169,615 行）取确定性子集 ~20k 题（SHA 固定抽样），含多跳源配额 |
| Reward 公式 | **冻结 v1（2026-08-19）**：`R = R_answer + 0.15·evidence_hit + 0.30·searched_and_correct_and_evidence_hit − 0.20·invalid_or_error − 0.45·redundant_search_count − 0.20·new_answer_leak_in_query`；format_score=0.1、α=0（valid_retrieval 不单独奖励）、evidence_hit 只检查 Retriever 返回的真实 document 正文 |
| 算法 | **Search-aware GRPO only**（adv_estimator=grpo，不改 advantage 层）；**GiGPO 推迟**（勘误②：step≥1 检索 Observation 分化 → 单例组 → 无有效 step 信用；等 nested rollout 版本再评估） |
| 步数/batch | 工程 smoke 1~2 步（只验显存/在线 reward/梯度/checkpoint）→ 行为 smoke 5~10 步（另行预注册）→ 正式 50~100 步；`ppo_epochs=1, gamma=1.0, env.rollout.n=5, seed=0`（与历史分段一致） |
| GPU 预算 | 六卡 1,2,3,4,6,7 工程 smoke（设计见 §9；与已验证全参数 FSDP/offload/gpu_mem=0.60 架构一致）；**本轮只设计不执行，所有 GPU 动作另行批准** |
| smoke 门禁 | ① 12 条 CPU 测试 + 历史 rollout 离线回放 5 条硬门禁（T4≤T1、T7≤T1 在内，见 §9）；② 训练侧 audit 显示 valid search > 0 且 invalid 率 < 50%；③ dev256 采样评测搜索率 > 0 |
| 停止条件 | 连续 N 步搜索率不恢复（无有效搜索）或 dev256 EM ≤ 记忆基线 → 停止并回到诊断；搜索恢复但 EM 无提升 → 转向 P1 检索侧（query 改写） |
| 并行对照 | Step300（记忆路径）与官方 SearchR1（搜索但欠利用路径）作为两个失败模式的对照锚点 |

**预期实验判别**：
- 若 shaping 后搜索恢复且 EM 超 Step300 → reward/credit 假设成立；GiGPO 精化等 nested rollout 版本（勘误②）后再评估
- 若搜索恢复但 EM 不升（证据带回但答错）→ 转为 P1：检索 query 改写（多跳题多查询/子问题分解），reward 不动
- 若搜索不恢复 → 回到诊断（shaping 信号在组内仍被淹没 → 需要更大 γ 或 GiGPO step 项）

## 9. Phase 4B 交付门禁与 GPU 工程 smoke 设计（2026-08-19，本轮交付）

### 9.1 冻结的 Search-aware GRPO v1（单一实现源）

```
R = R_answer + 0.15·evidence_hit + 0.30·searched_and_correct_and_evidence_hit
    − 0.20·invalid_or_error − 0.45·redundant_search_count − 0.20·new_answer_leak_in_query
```

- 固定项：`format_score=0.1`（本轮不改格式奖励）；`valid_retrieval` 系数 **α=0**（无关搜索本身不加分）；`evidence_hit` 只检查 **Retriever 返回的真实 document 正文**，不得检查 query、error 文本或模型输出；`searched_and_correct_and_evidence_hit` 要求至少一次真实成功检索 + 返回证据命中 + 最终答案 EM 正确
- **answer-leak 防作弊**：`new_answer_leak_in_query`——normalized ground-truth alias 出现在 search query 中、且该 alias 原本不在 question 中；排除过短/空 alias（规则与阈值写入测试）；命中时**清零该次搜索的 evidence_hit 与 sce 奖励**并额外扣 0.20；问题本身包含答案 alias 不得误判；每条命中记录 `{question, query, alias}` 审计字段
- **step 归因**：R_answer 只放终止 answer step；evidence_hit / invalid_or_error / new_answer_leak / redundant_search 放对应 search step；sce 在终止 answer step 结算但经 episode metadata 关联真实成功检索；禁止把完整 shaping reward 复制给 episode 内每个 step；Observation token 继续排除于 policy loss
- **语义澄清 2026-08-19（工程实现）**：**冗余（第 2 次起）搜索步不获得 evidence 加分**（`evidence_credit=false`，该步仅计 −0.45）；否则「两次有效搜索 + 答对」= 1.15 > 1.0，违反硬门禁 ⑥（重复搜索 ≤ 直接答对）。`evidence_effective` 保持 true（sce 经 episode metadata 结算只要求 episode 内存在一次真实成功检索证据命中）。由此 diag T6（2 次有效搜索）恰为 1.00 == T1，与 §4.1 防 hack 表一致；任何多搜索组合总奖励 ≤ 1.00（第 2 步起边际恒为 −0.45）
- **分量记录**：每条 rollout 记录 `answer_reward / format_reward / evidence_hit_reward / searched_correct_bonus / invalid_penalty / redundant_penalty / answer_leak_penalty / total_reward`，并断言分量之和与训练 score 一致
- **实现隔离**：独立 patch `0007-search-aware-step-reward.patch`；独立配置 profile `search-aware-grpo-v1`；official-loose 基线配置/checkpoint/评测脚本与结果**不修改**；adv_estimator 仍为 GRPO；起始模型 Qwen2.5-3B Base（不从 gs300 继续）；FSDP/offload/vLLM 训练拓扑不变

### 9.2 CPU 门禁（12 条测试 + 历史回放 5 条硬门禁）

CPU 测试（纯函数，无 GPU/无 Ray）：
① 不搜索直接答对 = 1.0；② 有效证据搜索后答对 = 1.45；③ 无关搜索后靠记忆答对 ≤ 1.0（T4≤T1 硬断言）；④ 有证据但答错 = 0.25；⑤ invalid query = −0.20；⑥ 重复搜索 ≤ 直接答对；⑦ 答案泄漏 query ≤ 直接答对（T7≤T1 硬断言）；⑧ question 原本含 alias 不误判；⑨ error observation 不算 evidence；⑩ 无搜索 episode 不得获得任何搜索奖励；⑪ 多步 reward 只落在对应 step；⑫ Observation token 的 policy loss mask 保持 0。

历史 rollout 离线回放（seg 0-300 全部 *audit.jsonl，按 (traj_uid, env_step) 去重）硬门禁：
`useful-search-correct > direct-correct`；`irrelevant-search-correct ≤ direct-correct`；`answer-leak-search-correct ≤ direct-correct`；`redundant-search-correct ≤ direct-correct`；`invalid < format-wrong < direct-correct`。
报告项：组内 reward 方差、全同 reward 组比例、search action 的 advantage 方向、各 reward 分量触发次数、数值重复计算检查。

**执行结果 2026-08-19（95,718 episodes / 19,796 groups，`gates/p3_v1_reward_replay_20260819.json`）——5 条硬门禁全部通过**：
- ① useful（n=34，均恰为 1.45 的单次有效搜索答对）> direct（n=12,928，1.00）✓
- ② irrelevant（n=51）max=1.00 ≤ 1.00 ✓；③ leak-correct（n=0）与 ④ redundant-correct（n=0）在历史数据上**空门禁**（与 diag `redundant_sum=0` 一致：历史无 ≥2 次搜索 episode）；语义由单元测试 ⑥⑦ 与分量算术结构性保证（多搜索总奖励 ≤ 1.00）
- ⑤ invalid（n=2,411，均值 −0.20）< format-wrong（n=78,643，0.100）< direct（1.00）✓
- 重复计算检查：95,718/95,718 episode 放置和 == 分量和，0 不一致；分量合计 answer 1,301,300¢ / format 794,780¢ / evidence 8,835¢ / sce 1,020¢ / invalid −65,620¢ / leak −240¢ / redundant 0¢
- 组内方差均值 0.034；全同组比例 69.7%；search advantage 均值 −0.58 < no-search +0.03（历史策略下搜索轨迹在组内仍处劣势——v1 修正的是**未来训练**的 reward 归因，不回改历史策略）
- 注意：历史 audit 的 record_score 记录的是旧 reward，与 v1 总分不可比（未做跨代数值断言，属设计内）

### 9.3 六卡工程 smoke 设计（仅设计，不执行；所有 GPU 动作另行批准）

**不用**"GPU1 单卡全参数 smoke"方案（未经显存验证）。采用与已验证的全参数 FSDP/offload/gpu_mem=0.60 训练拓扑一致的方式，六卡 1,2,3,4,6,7 一至两步工程 smoke：
- 目的仅验证：显存（FSDP 全参数 + offload + vLLM rollout 共存）、reward 在线计算（v1 分量落位与 sum 校验）、非零梯度（audit 或 loss 变化）、checkpoint 与恢复
- **不用 2 步 EM 判断算法效果**（样本量不足，属行为问题）
- 工程 smoke 通过后，另行预注册 5~10 步行为 smoke（搜索率/分量触发/advantage 方向）
- 预计耗时（估算）：启动与预检 ~10 min；1~2 步六卡 rollout+训练 ~20~40 min；checkpoint+恢复验证 ~10 min；合计 **~40~60 min 单次**（不含排队/重试）
- 完成后退出验收：exit code、无残留 PID/端口/Ray、GPU 回基线

## 10. 停止声明

诊断与 Phase 4B 工程交付已完成。**不自动启动 SFT、GRPO、GiGPO 或任何六卡训练**，等待单独批准。
