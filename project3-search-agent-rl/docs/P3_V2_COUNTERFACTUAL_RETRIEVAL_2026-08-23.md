# P3 v2 Step5 证据使用反事实检索评测（2026-08-23）

## 0. 结论先行

**证据因果性检验通过：v2 Step5 的正确回答确实依赖真实检索证据。** 同一
merged 模型（`p3-v2-behavior-gs5-merged-20260823d`）、同一
official-confirm256-v1 题集、greedy temperature=0、GPU1-only，三个严格配对
条件（仅改变证据内容，协议/解码/步数预算不变）：

| 条件 | EM | 搜索率 | search→correct | 搜索且对(绝对) | 未搜索且对 |
|---|---|---|---|---|---|
| **real（原运行）** | **78/256 (30.5%)** | 233/256 (91.0%) | 0.296 | **69** | 9 |
| shuffled（错位证据） | 10/256 (3.9%) | 233/256 (91.0%) | 0.004 | 1 | 9 |
| no-evidence（中性空证据） | 15/256 (5.9%) | 233/256 (91.0%) | 0.026 | 6 | 9 |

- **real vs shuffled 配对（逐题）**：1→1=10，0→0=178，real 对/cf 错=**68**，
  real 错/cf 对=**0**；精确双侧 McNemar **p≈0**；real 独有的正确 68 题。
- **real vs no-evidence 配对**：1→1=11，0→0=174，real 对/cf 错=**67**，
  real 错/cf 对=**4**；精确双侧 McNemar **p≈0**；real 独有的正确 67 题。
- **real 的 69 个"搜索且正确"中：shuffled 下 68 个翻转，no-evidence 下 67 个
  翻转** —— 绝大多数正确回答随证据消失而消失。
- **内部对照完美**：23 个未搜索题目在三个条件下全部保持 9/9 正确 —— 证据
  操作只影响使用证据的轨迹，不污染无关轨迹。
- **协议连续性**：三个条件搜索率逐字节相同（233/256）、0 api_error、
  0 环境异常；q5 的第一步 `invalid_query` 在三个条件下原样保留（首步行为
  在见到证据前完全相同）。
- **证据内容审计**：shuffled 的证据 = 预注册映射 `(i+17) mod 256` 对应题目
  的真实检索结果，独立对 retriever 抽查 31/31 个搜索步文档 ID 全部匹配。
- 证据改变但答案文本不变（evidence-change-but-answer-unchanged）：
  shuffled 25 题、no-evidence 28 题 —— 少数（约 1/6），说明"不依赖证据也答
  对"是少数情况，不是主流。

**声明**：单次 greedy 运行内 p 值极显著（配对 68:0 与 67:4），但这是"证据
内容变化→正确率崩塌"的因果证据，不等价于"78 vs 74 的模型间差异显著"
（后者见配对统计报告，p=0.66）。**"搜索并答对"≠"因证据答对"的质疑已在
此评测中证伪：搜索并答对几乎全部依赖真实证据。**

## 1. 设计（预注册、固定映射、fail-closed）

三个条件共享同一 eval 管线（`scripts/run_p3_eval_v2.py` + v2 树门禁
0001..0007 逐字节重建 PASS、merged-model 门禁 PASS、GPU1-only、seed=0、
max_steps=4、history=4、topk=3、retriever health 21,015,324 vectors）：

- **real**：原运行（`p3-eval-v2-behavior-gs5-confirm256-20260823a`），零补丁。
- **shuffled**（`…-shuffled-20260823a`）：模型自己的查询先**真实执行**检索
  （真实状态保留：非 success 原样返回、绝不改写错误/空结果）；成功后证据
  替换为固定映射 `(i+17) mod 256` 题目的**真实检索结果**（以该题目文本为
  查询二次真实检索），计数/格式/成功状态保真；映射题目真实检索为空时按
  真实 `no_results` 包络返回。映射与结果 SHA **在 episode 循环开始前写入**
  `retrieval_condition_preregistration.json`。
- **no-evidence**（`…-noevidence-20260823a`）：每次成功搜索返回固定中性包络
  （"No relevant documents were found…"×3，文档 ID `noev-0..2`），不发起
  任何 HTTP；`invalid_query`/`api_error` 路径保持原样。
- 实现为 eval 进程内的运行时 monkey-patch（`install_retrieval_condition`），
  **vendor v2 树盘上零改动**（门禁逐字节验证）；只改证据内容，绝不改
  prompt/projection/解码/步数。
- 计数器（fallback 审计）：shuffled 691/691 次映射服务、0 次 fallback、
  0 次真实失败保留（retriever 全程健康）；no-evidence 665/665 次中性包络、
  0 HTTP、0 错误。
- 映射 SHA `93363b67…` 与独立复算 `(i+17) mod 256` 一致（预注册防篡改）。

## 2. 门禁判定（5 项全过 → 授权 fresh 10-step）

| 门禁条件 | 结果 |
|---|---|
| real EM > shuffled EM | 78 > 10 ✓ |
| real EM > no-evidence EM | 78 > 15 ✓ |
| real-only 正确 > 各 counterfactual-only 正确 | 68 > 0；67 > 4 ✓ |
| 协议/合规/作答率差异不可由环境错误解释 | 搜索率三条件逐字节相同 233/256；0 api_error；invalid 仅 q5 第一步（原样保留）；作答率差异（232→187/195）为行为变化 ✓ |
| ≥1 个可审计样本 | 大量：68/67 个翻转 + 文档级 31/31 独立核对 ✓ |

p 值如实报告（均为 ~0）；门禁本身不要求 p<0.05，此处事实满足。

## 3. 行为观察（证据被破坏后模型如何反应）

- 搜索轨迹不变（233/256、步数从 2.80 增到 3.60-3.70：无效证据下搜索轮次
  增加、收敛更慢）；env 提交答案从 189 降到 30-50，offline 合规从 232 降到
  187-195 —— 证据不可信时模型更少提交答案、更多步才放弃。
- true_redundant_rate 从 0.236 升到 0.65-0.66（无效证据→重复查询）。
- no-evidence-only 正确 4 题（real 证据反而"带偏"的少数轨迹）；shuffled-only
  正确 0 题（错位证据无任何正贡献）。

## 4. 产物与复算

- 运行：`runs/p3-eval-v2-behavior-gs5-confirm256-shuffled-20260823a/`、
  `runs/p3-eval-v2-behavior-gs5-confirm256-noevidence-20260823a/`
  （results.json + episodes.jsonl + retrieval_condition_preregistration.json）。
- 配对统计：`gates/p3_v2_counterfactual_stats_20260823.json`；
  复算：`CUDA_VISIBLE_DEVICES='' python3 scripts/p3_v2_counterfactual_stats.py`。
- 补丁逻辑 CPU 冒烟：`scripts/_smoke_counterfactual.py`（29 项断言，
  含注册路径/映射/保真/包络/计数器）。
- 配套文档：`P3_V2_PAIRED_STATS_REPORT_2026-08-23.md`（模型间差异的配对
  视角）、`P3_V2_BEHAVIOR_EVAL_REPORT_2026-08-23.md` §3（准确表述）。

## 5. 决策链

门禁 5/5 通过 → 按 2026-08-23 指令启动 fresh v2 10-step（Run ID
`p3-search-aware-clean-v2-grpo10-fsdp6-b66-n5-s0-20260823a`，
Qwen2.5-3B-Instruct Step0 从头、total=10、save_freq=5、配置指纹与 v2 Step5
一致、warmup 0.285、GPU 1,2,3,4,6,7）。
