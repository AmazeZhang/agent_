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
| C1-conservative | 0.1 | 0.02 | 0.05 | 0.10 | 0.10 | 0.20 | +0.17 | PASS |
| C2-moderate | 0.1 | 0.02 | 0.10 | 0.20 | 0.20 | 0.35 | +0.32 | PASS |
| C3-format-0.05 | 0.05 | 0.02 | 0.05 | 0.10 | 0.10 | 0.20 | +0.17 | PASS |
| C4-format-0.0 | 0.0 | 0.02 | 0.05 | 0.10 | 0.10 | 0.20 | +0.17 | PASS |
| C5-evidence-driven-a0 | 0.1 | 0 | 0.15 | 0.30 | 0.20 | 0.45 | +0.45 | PASS |
| C6-spam-averse | 0.1 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | +0.35 | PASS |

8 条防 hack 轨迹断言（T1 直答对 / T2 有效搜索+证据答对 / T3 相关证据答错 / T4 无关文档凭记忆答对 / T5 invalid query / T6 重复搜索刷分 / T7 标准答案写进 query / T8 只格式正确答错）：**C1–C6 全部 PASS**（T2>T1、T1>T3、T3>T5、T6≤T1、α≤0.05、T8<T1、μ≥α+β+γ）。C0 基线失败（T2==T1）为机制性发现：当前奖励不给证据支撑搜索任何额外信用。

T7 注记：答案写进 query 在奖励层面不可检测（T7==T2 by construction），缓解在 query 校验层而非奖励层。

### 4.2 历史轨迹评估（seg 0-100，n=29,838，off-policy）

- 历史搜索质量：searched 4,524 / valid 1,297（28.7%）/ evidence-hit 585（12.9%）/ correct_with_evidence_search 41（0.9%）；redundant_sum 0
- 所有 shaping 候选下 search_adv 均为负（−0.36 ~ −0.48）、nosearch_adv 为正（+0.06 ~ +0.09）→ **shaping 在历史 off-policy 轨迹上不能逆转搜索劣势**（γ 只触发 41/4,524）
- C0 基线：同组全同奖励组比例 82.5%；C1–C6 降到 ~51–55%（组内方差略升）——shaping 增加组内信号但有限
- 仅推荐系数范围（C1-conservative ~ C6-spam-averse 之间），不冻结

## 5. GiGPO 接入审计（只读，file:line 已核实）

### 5.1 fork 内真实实现

- `vendor/verl-agent/gigpo/core_gigpo.py:138-171` `compute_gigpo_outcome_advantage` = episode_norm_reward（按 uid 组=同 prompt 的 n 个 traj，含跨 step 去重 seen_pairs）+ `step_advantage_w` × step_norm_reward（按 build_step_group 聚类）
- `:243-331` `build_step_group`：同一 uid 组内 anchor_obs 精确相同成一组；`gigpo_enable_similarity` 时 SequenceMatcher ≥0.95 文本相似并入组
- `:334-384` step_norm_reward：组内归一（mode 切换是否除 std）；`:87-132` compute_step_discounted_returns
- config 入口：`verl/trainer/config/ppo_trainer.yaml:250-254`（gigpo 配置块）；ray_trainer.py:344-358（GiGPO 分支）、:243（compute_advantage 签名）
- **episode 项权重硬编码 1.0**（GraphGPO recipe 才有 episode_advantage_w）——若需调整要改 core_gigpo.py

### 5.2 search 场景下 GiGPO step 分组语义（env manager anchor 已核实）

- `agent_system/environments/env_manager.py:62-78,148-164`：reset 时 anchor=obs（同组 5 traj 相同）；step 时 anchor=next_obs（依检索结果分化）
- 结论：**step 0 全组 5 条共享同一 anchor → 1 个 step 组**；step ≥ 1 通常每组 1 条（检索结果分化），仅同 query 同结果或同 error obs 时成组
- 即 GiGPO 在搜索场景：episode 项 ≈ GRPO（组内归一），step 项在 step 0 无组内区分、step ≥ 1 近似单例归一 → **提升有限，但 step-level 信用分配为多轮 credit 提供了机制入口**

### 5.3 Observation token 与损失

- `dp_actor.py:317-372`：multi_turn=False → response_mask = attention_mask[:, -response_length:] → **<information> observation 是 prompt token，构造性排除于 policy loss**（不改则永远学不到"读证据"）
- 当前信用分配：`agent_system/reward_manager/episode.py:20-96` 把完整 episode reward 放在**每条 step record 最后一个有效 response token** 上 → 搜索动作与答案动作同分 → 这是"搜索无用"机制根源之一（诊断 2/3 数据将给出证据强度）

### 5.4 架构可行性

GiGPO 是 driver-side 纯 advantage 计算替换（critic-free），**FSDP/offload/vLLM/rollout 架构不变**；开启只需 ppo_trainer.yaml 配置 + adv_estimator 分支；可保持与当前 GRPO 完全相同的训练拓扑。

## 6. 两套候选方案（仅设计，未实施）

### 方案 A：Search-aware GRPO（最小改动）

- **Reward**：诊断 4 候选系数（推荐 C1-conservative 起步：α=0.02, β=0.05, γ=0.10, λ=0.10, μ=0.20, format=0.1；或 C5-evidence-driven 若诊断 2 显示证据可用是主瓶颈）
- **实现点**（verl-agent fork）：
  1. RewardManager 需按 step 产出 shaping 分量（valid/evidence/sce/invalid/redundant），挂到 step record（仿 episode.py 但按 step）
  2. 现有 `token_level_rewards` 广播语义保留：episode 分量仍在最后 token；shaping 分量可放该 step 最后 token
  3. loss mask 不变（Observation 仍排除——方案 A 不读证据，靠 shaping 奖励间接驱动）
- **优点**：改动最小（reward 层 + 配置）；GRPO 语义完全不动；可直接对比当前基线
- **缺点**：仍学不会"读完证据再作答"；T7（答案写进 query）需 query 校验层

### 方案 B：Search-aware GiGPO（信用分配+step 奖励）

- 在方案 A 的 shaping reward 之上：
  1. `adv_estimator: gigpo` + `step_advantage_w: 0.3~0.5`（起步建议 0.3，避免 step 项噪声主导）
  2. 修改 `core_gigpo.py` 增加 `episode_advantage_w` 配置（当前硬编码 1.0）
  3. **可选（需补丁）**：把 Observation token 纳入 response_mask——训练"读证据"；风险：context 长度 +2048、训练吞吐下降
  4. step 分组语义已核实：step 0 全组 1 组（≈GRPO），step≥1 多为单例（信用天然细粒度）
- **优点**：搜索动作与答案动作获得不同 credit；多轮 search/answer 可分别归因
- **缺点**：改动面大（advantage 层 + reward 层 + 可选 loss mask 补丁）；fork 内 GiGPO 未在 search 场景验证过

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
| Reward 公式 | `R = R_answer + α·valid + β·evidence_hit + γ·searched_and_correct_and_evidence_hit − λ·invalid_or_error − μ·redundant_count`；**起始系数 C1-conservative**（format=0.1, α=0.02, β=0.05, γ=0.10, λ=0.10, μ=0.20），探索范围 C1~C6（α∈[0,0.05], β∈[0.05,0.15], γ∈[0.1,0.3], λ∈[0.1,0.3], μ∈[0.2,0.45]），不冻结 |
| 算法 | **先 Search-aware GRPO**（方案 A：仅 reward 层改动，验证 reward 假设；诊断 4 显示 shaping 能拉开 T2>T1 且防 hack 全过）→ 若搜索优势仍不出现，上 **Search-aware GiGPO**（方案 B：step 信用，step_advantage_w∈[0.3,0.5]） |
| 步数/batch | smoke 2 步门禁 → 正式 50~100 步；`ppo_epochs=1, gamma=1.0, env.rollout.n=5, seed=0`（与历史分段一致） |
| GPU 预算 | 单卡 GPU1（3B FSDP 小 batch）可作 smoke；正式分段需单独批准六卡（边界：本轮不跑任何训练） |
| smoke 门禁 | ① 诊断 4 的 8 轨迹防 hack 断言在训练环境过；② 训练侧 audit 显示 valid search > 0 且 invalid 率 < 50%；③ dev256 采样评测搜索率 > 0 |
| 停止条件 | 连续 N 步搜索率不恢复（无有效搜索）或 dev256 EM ≤ 记忆基线 → 停止并回到诊断；搜索恢复但 EM 无提升 → 转向 P1 检索侧（query 改写） |
| 并行对照 | Step300（记忆路径）与官方 SearchR1（搜索但欠利用路径）作为两个失败模式的对照锚点 |

**预期实验判别**：
- 若 shaping 后搜索恢复且 EM 超 Step300 → reward/credit 假设成立，继续 GiGPO 精化
- 若搜索恢复但 EM 不升（证据带回但答错）→ 转为 P1：检索 query 改写（多跳题多查询/子问题分解），reward 不动
- 若搜索不恢复 → 回到诊断（shaping 信号在组内仍被淹没 → 需要更大 γ 或 GiGPO step 项）

## 9. 停止声明

诊断已完成。**不自动启动 SFT、GRPO、GiGPO 或任何六卡训练**，等待单独批准。
