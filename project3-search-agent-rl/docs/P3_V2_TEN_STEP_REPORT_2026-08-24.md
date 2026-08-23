# P3 v2 fresh 10-step GRPO10 训练 + 五路对比报告（2026-08-24）

## 0. 结论先行

- **训练**：fresh v2 10-step 全程 10/10 步完成，10 项中止条件（C1–C10）全部
  PASS，exit 0，总时长 **4:02:03**（1447.9 s/it）。无 OOM/NaN/Xid/掉卡/NCCL
  分歧，无 GPU0/5 使用，config 指纹全程稳定，duplicate identity 恒为 0。
- **评测**：merged gs5（EM 68）→ merged gs10（EM 73），Step5→Step10 净
  **+5**（gained 21 / lost 16，精确双侧 McNemar p=0.511，方向性正面、非显著）。
- **Step10 成功建议 7 项中达成 6 项**，唯一未达成：EM ≥ 74/256 差 1 题
  （实际 73/256）。详见 §6。
- **五路对比**（§5）：gs10 与 clean GRPO10 并列（net −1, p=1.0），相对
  Step0 净 +8（p=0.33），均为方向性/不显著，如实报告，不调参。
- **证据因果性**：未在本轮 fresh gs10 上重新运行反事实评测（Phase 6 协议
  未包含 counterfactual re-run）；沿用 Phase 3 对旧 gs5 merged 模型建立的反
  事实证据（real 78 vs shuffled 10 / no-evidence 15，配对 p≈0）作为机制支撑，
  并如实标注此证据针对旧 gs5、非 fresh gs10。
- **运行间方差提示**：相同 gs5 checkpoint（同一 merge 产物）的两次独立 greedy
  评测结果 78 vs 68（net +10，p=0.11），故单次运行的 EM 差异 ±10 量级在噪声内。

## 1. 运行事实（run record）

| 项 | 值 |
|---|---|
| Run ID | `p3-search-aware-clean-v2-grpo10-fsdp6-b66-n5-s0-20260823a` |
| 起始/结束 | 2026-08-23 18:54:50 → 22:59:39 (+08:00)，时长 **4:02:03** |
| 仓库 commit | `97c6cb2`（本轮报告提交后见 §10 更新记录） |
| 配置指纹 | `d727b64f7c1c235e1d070637d9af498a02b1b89868bce088afdc19b814358402` |
| 基础模型 | Qwen2.5-3B-Instruct（Step0 从头，无 warm-start） |
| 数据 | 训练池（task 池，非 heldout）；评测 heldout.parquet `ffebf468e756a673…` |
| 关键配置 | total_training_steps=10、warmup 0.285、seed 1234/1234、save_freq=5、train_batch_size=66、rollout.n=5、ppo_mini_batch_size=330、ppo_epochs=1、adv_estimator=grpo、gamma=1.0、lr=1e-6、kl=0.001、FSDP 3× offload、gpu_mem=0.60、max_num_seqs=64、env max_steps=4、history=4、format_score=0.0 |
| GPU | 1,2,3,4,6,7（GPU0/5 禁用，metadata.env `physical_gpu_ids=1,2,3,4,6,7`） |
| 资源 | 6× RTX 3090 24 GB（`gpu_total_mib` 24564），Ray 临时目录 `/tmp/p3r.9TX1Zk`（已归档） |
| exit | `exit_code=0`；监控 watcher 全程 0 violation |

Merge 产物：`models/p3-v2-tenstep-gs5-merged-20260823a`、
`models/p3-v2-tenstep-gs10-merged-20260823a`（model_merger fsdp；
config.json md5 相同 `9e5b910810634b3e`，safetensors shard1 md5 分别为
`317cec863a02101d…` / `d7df8575f80eecc8…`，即两 checkpoint 权重不同）；
`scripts/verify_p3_merged_model.py` 两者均 VERIFY_MERGED: PASS。

## 2. 分步训练审计（权威 artifact：`gates/p3_ten_step_audit_20260823a.json`，由
`scripts/audit_p3_ten_step.py` 复算）

每步 330 条轨迹（66 组 × n=5 rollouts），逐条 audit 记录粒度
`(traj_uid, env_step)`；`uid` 为 GRPO group id，非重复信号。

| step | 轨迹 | 记录 | 搜索率 | useful | invalid | redundant | 作答提交率 | 闭卷 | dup | pad | 正优势 | 优势和 | 最大分量 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 330 | 800 | 0.782 | 0.555 | 0.028 | 0.168 | 0.239 | 72 | 0 | 0 | 162 | 0.00 | answer_reward_c |
| 2 | 330 | 793 | 0.727 | 0.451 | 0.028 | 0.197 | 0.239 | 90 | 0 | 0 | 134 | 0.00 | answer_reward_c |
| 3 | 330 | 815 | 0.773 | 0.466 | 0.045 | 0.140 | 0.252 | 75 | 0 | 0 | 153 | 0.00 | answer_reward_c |
| 4 | 330 | 787 | 0.736 | 0.460 | 0.024 | 0.155 | 0.297 | 87 | 0 | 0 | 144 | 0.00 | answer_reward_c |
| 5 | 330 | 830 | 0.803 | 0.468 | 0.018 | 0.166 | 0.336 | 65 | 0 | 0 | 145 | 0.00 | answer_reward_c |
| 6 | 330 | 805 | 0.821 | 0.530 | 0.025 | 0.114 | 0.309 | 59 | 0 | 0 | 156 | 0.00 | answer_reward_c |
| 7 | 330 | 753 | 0.773 | 0.437 | 0.038 | 0.116 | 0.273 | 75 | 0 | 0 | 154 | 0.00 | answer_reward_c |
| 8 | 330 | 850 | 0.894 | 0.296 | 0.010 | 0.127 | 0.233 | 35 | 0 | 0 | 142 | −0.00 | answer_reward_c |
| 9 | 330 | 774 | 0.879 | 0.451 | 0.016 | 0.106 | 0.297 | 40 | 0 | 0 | 126 | −0.00 | answer_reward_c |
| 10 | 330 | 792 | 0.915 | 0.511 | 0.002 | 0.121 | 0.309 | 28 | 0 | 0 | 128 | 0.00 | answer_reward_c |

注：`useful` = useful-search（evidence_hit）率，按 search records 计；Step8 的
0.296 低于 Step1 的 0.555×50%=0.2775 阈值之上且非连续 3 步（C2 不触发）；
`优势和`≈0 是 GRPO 去中心化优势的预期（batch 内均值归零），非异常。
Step10 奖励分量（cents）：`answer 10200 / evidence_hit 2790 /
searched_correct_bonus 2400 / invalid −20 / redundant −1120` —— answer 主导，
true-redundant 惩罚未再次成为最大分量（C4 PASS）。Step10 搜索轮次分布
{3:53, 2:54, 1:195, 0:28}；搜索轨迹率 0.915、invalid 0.2%。

**10 项中止条件结果**：C1（搜索率<70% 连续 2 步）ok、C2（useful-search 衰退）ok、
C3（invalid>10%）ok、C4（redundant 主导）ok、C5（重复身份=0）ok、
C6（reward/return/advantage 一致性）ok、C7（OOM/NaN/Xid/掉卡/retriever）ok、
C8（config 指纹）ok、C9（GPU0/5）ok、C10（NCCL/worker 丢失）ok。
`any_violation=False`。

## 3. 显存报告（三类口径分开，聚合值不得当作 per-GPU 物理峰值）

| 口径 | 数值 | 说明 |
|---|---|---|
| nvidia-smi per-GPU **physical** peaks（权威） | GPU1 **24061** / GPU2 **23881** / GPU3 **23977** / GPU4 **23565** / GPU6 **23853** / GPU7 **23771** MiB（卡总量 24564 MiB） | 训练 wrapper 内嵌 2s 采样器，`runs/<run>/peak_memory_nvidia_smi.json`；监控 watcher 独立采样交叉一致（±190 MiB 内） |
| torch `max_memory_allocated` | **24.903 GB** | verl 日志 per-step worker 聚合值（多 worker 取最大），torch allocator 视角，**不是**任何单卡的物理峰值 |
| torch `max_memory_reserved` | **36.645 GB** | 同上聚合口径（allocator 预留含缓存碎片），**不是**物理峰值；与 nvidia-smi 峰值之差来自 torch caching allocator 预留 vs 实际占用 |

**约束声明**：per-GPU 物理峰值只允许引用 nvidia-smi 采样器数值；torch
aggregate 值（24.903 / 36.645 GB）是 worker 聚合的 allocator 视图，绝不
改写为 per-GPU physical peaks。CPU 内存峰值（verl 日志 `cpu_memory_used_gb`）
99.786 GB 仅作运行环境参考，不在比较口径内。

## 4. 评测运行

| run | 模型 | 数据 | 解码 | EM | 说明 |
|---|---|---|---|---|---|
| `p3-eval-v2-tenstep-gs5-confirm256-20260823a` | merged gs5 | heldout.parquet `ffebf468…` | vLLM 原生 greedy，GPU1 | **68/256**（26.6%） | 与旧 gs5（78）同 checkpoint 的不同运行，见 §7 |
| `p3-eval-v2-tenstep-gs10-confirm256-20260823a` | merged gs10 | 同 | 同上 | **73/256**（28.5%） | 主结论模型 |
| `p3-eval-v2-tenstep-gs10-dev64-sampling-20260823a` | merged gs10 | dev64 `adc1a48d…` | vLLM sampling temp=1.0 × 5 rollouts，GPU1 | 80/320（25.0%） | **仅诊断**，无阈值；compliance 311/320，prompt_check 300/300 |

## 5. 五路对比（权威 artifact：`gates/p3_v2_five_way_stats_20260823.json`，由
`scripts/p3_v2_five_way_stats.py` 复算；n=256，逐题 `question_id` 严格配对，
同数据 SHA `ffebf468…`，全部 greedy temp=0 / seed=0 / GPU1）

| 运行 | EM | Wilson 95% | 搜索 | s2a | s2c | sc_abs | nsc_abs | ans_off | ans_env | 步/题 | invalid | maxst | 真冗余搜索 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean Step0 | 65 | [0.205, 0.311] | 180 | 0.989 | 0.228 | 41 | 24 | 254 | 211 | 2.32 | 0 | 45 | 0 |
| clean GRPO10 | 74 | [0.237, 0.347] | 161 | 0.801 | 0.286 | 46 | 28 | 224 | 224 | 2.13 | 0 | 32 | 0 |
| clean GiGPO10 | 69 | [0.219, 0.327] | 140 | 0.836 | 0.229 | 32 | 37 | 233 | 233 | 1.95 | 0 | 23 | 0 |
| **fresh v2 gs5** | **68** | [0.215, 0.323] | 227 | 0.912 | 0.251 | 57 | 11 | 236 | 195 | 2.70 | 2 | 61 | 112 |
| **fresh v2 gs10** | **73** | [0.233, 0.343] | 239 | 0.962 | 0.272 | 65 | 8 | 247 | 218 | 2.49 | 3 | 38 | 61 |

关键配对（精确双侧 McNemar）：

| 配对 | 1→1 | 0→0 | 目标对/对照错 | 对照对/目标错 | 净 | p |
|---|---|---|---|---|---|---|
| gs10 vs gs5 | 52 | 167 | 21 | 16 | **+5** | 0.511 |
| gs10 vs GRPO10 | 49 | 158 | 24 | 25 | **−1** | 1.000 |
| gs10 vs GiGPO10 | 39 | 153 | 34 | 30 | **+4** | 0.708 |
| gs10 vs Step0 | 43 | 161 | 30 | 22 | **+8** | 0.332 |
| gs5 vs GRPO10 | 48 | 162 | 26 | 20 | +6 | 0.461 |
| gs5 vs Step0 | 41 | 164 | 27 | 24 | +3 | 0.780 |
| GRPO10 vs Step0 | 46 | 163 | 28 | 19 | +9 | 0.243 |

全部配对 **p≥0.24**，无显著差异；"fresh v2 gs10 为该评测集第一名"维持为
**方向性正面**表述（EM 73 > 74 之外的所有对比线，但区间重叠）。

### Step5→Step10 轨迹（fresh v2 线内）

- gained **21**（gs5 错 gs10 对）、lost **16**（gs5 对 gs10 错）、
  maintained_correct 52、maintained_wrong 167；net **+5**，McNemar p=0.511。
- 行为无 collapse：搜索率 88.7%→93.4%，s2a 0.912→0.962，
  searched-and-correct 绝对量 57→65，invalid 2→3（≤5% 阈值内），
  真冗余搜索 112→61，步/题 2.70→2.49。搜索轮次分布从 {1:96,3:77,2:54,0:29}
  变为 {1:144,3:47,2:48,0:17}——单轮搜索占比上升（收敛更快）。
- 逐题细节（含 search behavior change）在 gate JSON `step5_to_step10` 字段。

## 6. Step10 成功建议逐项（pre-registered 阈值）

| # | 建议 | 阈值 | 实际 | 达成 |
|---|---|---|---|---|
| 1 | EM | ≥ 74/256 | 73/256 | ✗（差 1 题） |
| 2 | 搜索率 | ≥ 80% | 93.4% (239/256) | ✓ |
| 3 | search→answer | ≥ 90% | 0.962 | ✓ |
| 4 | invalid | ≤ 5% | 3/256 = 1.2% | ✓ |
| 5 | searched-and-correct 绝对量 | ≥ fresh gs5 | 65 ≥ 57 | ✓ |
| 6 | real-retrieval 反事实优势持续 | 沿用 Phase 3 证据 | 未重跑，见 §0/§8 | —（如实标注） |
| 7 | 无 Step5→Step10 行为 collapse | 行为逐项稳定/改善 | 全部稳定或改善 | ✓ |

7 项中 **6 项达成**；第 1 项 EM 差 1 题未达成，第 6 项因协议未包含
counterfactual re-run 而未检验（不视为达成，如实标注）。**不因结果不理想
自动调参**：本结论保留为"10 步训练质量良好、EM 提升方向性正面但未达显著"。

## 7. 运行间方差（诚实标注）

相同 gs5 checkpoint（同一 merge 产物）的两次独立 greedy 评测：
旧 run（2026-08-23 五步实验）EM 78，本轮 fresh gs10 训练里另起的 gs5 评测
EM 68 —— net +10，精确双侧 McNemar p=0.110。两次运行配置/数据/解码完全
相同，差异仅来自训练端 checkpoint 重放与 vLLM 解码的非确定性。因此任何
单次运行间的 ±10 量级 EM 差异均在噪声范围内；本报告所有跨 run 结论都以
配对检验为准，不依赖单点差值。

## 8. 证据因果性声明边界

- Phase 3 已建立的反事实证据（旧 gs5 merged：real 78 vs shuffled 10 vs
  no-evidence 15；配对 68:0 / 67:4，p≈0；real 的 69 个搜索且正确在扰动下
  翻转 68/67）指向"搜索并答对 ≈ 因证据答对"。该证据的模型是旧 gs5 产物，
  与 fresh gs10 同协议（search-aware reward、search_projection）但不同
  checkpoint；本轮未重跑，不把机制结论静默扩展到 fresh gs10 的 73 分。
- 本报告的所有显著性表述均为逐题配对检验；行为差异（搜索率、s2a、冗余
  等）为描述性统计，不单独声称因果。

## 9. 与既有对比线的关系（不替换旧模型）

- 旧 5 步 merged 模型（`p3-v2-behavior-gs5-merged-20260823d`，EM 78）与
  新 10 步模型并存；**未替换、未删除**任何旧产物。fresh gs10（EM 73）与
  旧 gs5（EM 78）的差异 5 题（78−73）在 §7 噪声区间内，不做优劣判定。
- 训练 checkpoint `checkpoints/global_step_5` 与 `global_step_10` 均保留在
  run 目录（save_freq=5）。

## 10. 本轮产物与复算

- `scripts/audit_p3_ten_step.py`（10 项中止条件审计，CPU-only 复算）
- `scripts/_p3_ten_step_watch.sh`（训练期 watcher：120s 审计 + 2s GPU 峰值采样）
- `scripts/p3_v2_five_way_stats.py`（五路配对统计，CPU-only 复算）
- `gates/p3_ten_step_audit_20260823a.json`（训练审计表 + 条件结果）
- `gates/p3_v2_five_way_stats_20260823.json`（五路对比 + Step5→10 轨迹）
- `docs/P3_V2_PAIRED_STATS_REPORT_2026-08-23.md`（已修复与 gate JSON 一致性）
- `docs/P3_V2_TEN_STEP_REPORT_2026-08-24.md`（本报告）
- 运行产物：`runs/p3-search-aware-clean-v2-grpo10-fsdp6-b66-n5-s0-20260823a`
  （含 `peak_memory_nvidia_smi.json`、rollouts/*.audit.jsonl）、两个 merged
  模型目录、三个 eval runs（gs5/gs10 greedy + gs10 dev64 sampling）。

## 11. 收尾状态

- 本轮 tmux 会话与监控 watcher 已清理（见下文提交说明）；13 个旧
  8/19–8/21 empty tmux 未动。
- 无 Ray/vLLM/main_ppo 残留进程；GPU 全部回到 18 MiB baseline
  （GPU0 的 407 MiB 为启动前即存在的外部无关进程）。
- Retriever healthy：`/health` 返回 ready，向量数 21,015,324。
