# P3 Search-aware clean v2：5 步行为训练完成 + Step5 评测报告（2026-08-23）

## 0. 结论先行

**Search-aware clean v2 五步行为实验已完整闭环：5/5 训练 exit 0 →
Step5 merge + VERIFY_MERGED: PASS → 主评测（greedy official-confirm256-v1，
GPU1-only）EM 78/256 = 30.5%，为该评测集上所有对比线的最高值；dev64
sampling 诊断（5 rollouts）题级正确率 42.2%（27/64）。**

- 训练：`p3-search-aware-clean-v2-behavior-fsdp6-b66-n5-s5-20260823d`，
  5/5 at 2:02:15（1467.03 s/it），exit 0，graceful shutdown，GPU 回基线。
- 主评测对比（同一 256 题集、vLLM 原生 greedy、GPU1-only）：

  | 模型 | EM | EM rate | Wilson 95% | 作答 | 步/题 |
  |---|---|---|---|---|---|
  | **v2 Step5（本报告）** | **78/256** | **30.5%** | [25.2, 36.4] | 232/256 | 2.80 |
  | clean GRPO10（2026-08-22 对照） | 74/256 | 28.9% | [23.7, 34.7] | 224/256 | 2.13 |
  | clean GiGPO10（对照） | 69/256 | 27.0% | [21.9, 32.7] | 233/256 | 1.95 |
  | clean Step0-instruct 基线 | 65/256 | 25.4% | [20.4, 31.1] | 254/256 | 2.32 |
  | upstream official（对照） | 7/256 | 2.7% | [1.3, 5.5] | 255/256 | 3.80 |

  v2 Step5 是当前该评测集上的**第一名**（比 Step0 基线 +5.1pp，比 GRPO10
  +1.6pp，比 GiGPO10 +3.5pp），但 Wilson 95% 区间相互重叠 —— **方向性正面
  证据，单次 greedy 运行，不构成显著性声明**（见 §6 声明边界）。
- 评测过程中修复的工程问题：eval wrapper 的 v2 树门禁与结果标签停留在
  `v2-0001..0006`（v2-0007 加入后首次启动 exit 15），已与训练线统一为
  `v2-0001..0007`；`decoding_backend` 静态标签改为按 temperature 派生。
  训练代码无任何改动。

## 1. 行为实验（5 步，fresh Step0）

按 2026-08-23 授权（新 smoke 11/11 PASS 时自动继续）启动：

| 项 | 值 |
|---|---|
| Run ID | `p3-search-aware-clean-v2-behavior-fsdp6-b66-n5-s5-20260823d` |
| 门禁 | `total_training_steps=5`（exit 26）、`PROJECT3_BEHAVIOR_APPROVED=yes`（exit 27）、`resume_from` 空（exit 28）、save_freq=1、每步 `rollouts/{n}.audit.jsonl` |
| GPU | 1,2,3,4,6,7（GPU0/5 禁用） |
| 配置 | Qwen2.5-3B-Instruct、train_batch_size=66、rollout.n=5、mini_batch=330、ppo_epochs=1、lr=1e-6、kl=0.001、warmup=0.285、FSDP 3× offload、gpu_mem=0.60、max_num_seqs=64、seed 1234/1234、env_max_steps=4、history=4、GRPO gamma=1.0、format_score=0.0 |
| 结果 | **5/5 at 2:02:15（1467.03 s/it），exit 0**，graceful shutdown（register centers / 6 worker actors / TaskRunner 全部停止） |

每步审计（`rollouts/{n}.audit.jsonl`，v2 表示：trajectory_advantage +
record_score + is_padding）：

| 步 | records | advantage 范围 | avg | adv>0 |
|---|---|---|---|---|
| 1 | 800 | [-1.789, 1.789] | -0.045 | 382 |
| 2 | 793 | [-1.789, 1.789] | +0.002 | 335 |
| 3 | 790 | [-1.789, 1.789] | -0.030 | 347 |
| 4 | 781 | [-1.789, 1.789] | -0.014 | 378 |
| 5 | 829 | [-1.789, 1.789] | -0.054 | 330 |

- advantage 严格对称 ±1.789（GRPO 5-rollout 组 mean-0/std-1 归一化的特征），
  正负占比 ~42-48% 均衡，无异常聚集；record 数 781-829 在 66 uid × ~5 步的
  合理范围内。
- checkpoints `global_step_1..5` 每步 6/6（model/optim/extra_state）完整。

## 2. Step5 merge + verify

```
model_merger.py merge --backend fsdp \
  --local_dir  …/global_step_5/actor \
  --target_dir …/models/p3-v2-behavior-gs5-merged-20260823d
→ MERGE_EXIT=0（6 个 FSDP shard 全部加载，model+tokenizer 保存）

verify_p3_merged_model.py --merged-dir …/p3-v2-behavior-gs5-merged-20260823d
→ VERIFY_MERGED: PASS
```

## 3. 主评测：greedy official-confirm256-v1（GPU1-only）

| 项 | 值 |
|---|---|
| Run ID | `p3-eval-v2-behavior-gs5-confirm256-20260823a` |
| 数据 | `searchr1-official-confirm256-v1/heldout.parquet`（256 题，与 manifest SHA 核对，leakage overlap=0） |
| 模型 | `models/p3-v2-behavior-gs5-merged-20260823d`（VERIFY_MERGED: PASS 门禁通过） |
| 解码 | vLLM 0.8.5 native，**greedy（temperature=0.0）**，num_rollouts=1，seed=0，max_steps=4，history=4，topk=3 |
| 时长 | 561.5 s；exit 0；GPU1-only（峰值 13.5 GiB，物理 GPU 仅 1） |
| 完整性 | prompt_check 232/232 passed；retriever 21,015,324 vectors ready；offline rescore 250 matches / 6 mismatches |

总体与分源：

| 源 | n | EM | rate |
|---|---|---|---|
| 2wikimultihopqa | 32 | 13 | 40.6% |
| bamboogle | 16 | 5 | 31.2% |
| hotpotqa | 64 | 12 | 18.8% |
| musique | 16 | 0 | 0.0% |
| nq | 64 | 20 | 31.2% |
| popqa | 32 | 10 | 31.2% |
| triviaqa | 32 | 18 | 56.2% |
| **合计** | **256** | **78** | **30.5%** |

- 行为侧：233/256 搜索（91.0%），232/256 作答（90.6%），平均 2.80 步/题
  （716 步）。**v2 Step5 优势的准确表述**：overall EM 比 clean GRPO10 多 4 题
  （78 vs 74），搜索率和 searched-and-correct 绝对数量更高（233 vs 161 题搜索、
  searched-and-correct 69 vs 46），但未证明统计显著（Wilson 95% 区间重叠，
  见 §6）；**"执行过搜索并答对"不等于"因检索证据而答对"** —— 证据因果性由
  反事实检索评测（`docs/P3_V2_COUNTERFACTUAL_RETRIEVAL_2026-08-23.md`）检验。
  （对照澄清：clean GRPO10 的 search→correct = 0.286（46/161），并非 0；
  "search→correct = 0/62" 属于失败的 **Search-aware patched v1 GRPO10**
  （2026-08-22 报告，EM 11.7%/搜索率 24.2%），与 clean 线是不同协议，
  不构成 v2 Step5 的对比基线。）
- 24/256 未作答：行为诊断项（见 §6）。

## 4. dev64 sampling 诊断（GPU1-only）

| 项 | 值 |
|---|---|
| Run ID | `p3-eval-v2-behavior-gs5-dev64-sampling-20260823a` |
| 数据 | `searchr1-p3-dev64-v1/dev64.parquet`（64 题，leakage overlap=0） |
| 解码 | vLLM native **sampling（temperature=1.0）**，**num_rollouts=5** → 320 episodes |
| 时长 | 746.1 s；exit 0；GPU1-only（峰值 13.8 GiB）；prompt_check passed；offline rescore 312/8 |

- 题级：64/64 作答（100%），63/64 搜索（98.4%），**27/64 正确（42.2%）**
  （Wilson 95% [30.9, 54.4]；"正确" = 5 个 rollout 中至少一个 EM 命中）。
- 分源（episode 级 EM）：2wiki 9/40 (22.5%)、bamboogle 5/20 (25.0%)、
  hotpotqa 15/80 (18.8%)、musique 0/20 (0.0%)、nq 12/80 (15.0%)、
  popqa 17/40 (42.5%)、triviaqa 17/40 (42.5%)。
- 平均 2.83 步/题（905 步）；作答合规 291/320 (90.9%)。
- dev64 为本轮新建的诊断集，无历史参考；sampling 数字是行为基线，
  与 confirm256 greedy 主评测（不同题集、不同解码）不可直接比较。

## 5. 本轮修复的工程问题（eval 门禁/标签陈旧）

v2 线在 2026-08-23 加入 v2-0007（duplicate-record 源头修复 + NCCL 死锁根因
修复）后，eval 侧两处仍停留在 0001..0006：

1. `scripts/run_p3_eval_v2.sh` v2 树门禁：patch 列表与三处 "v2-0001..0006"
   文本未含 v2-0007 → 首次启动 **exit 15**（diff 到的 5 个文件恰为 v2-0007
   的 5 个文件）。已统一为 `v2-0001..0007`，并用与门禁相同的重建流程验证
   `pristine 20bd331b + 0001..0007 == 工作区`（IDENTICAL PASS）后重新启动。
2. `scripts/run_p3_eval_v2.py`：`--v2-dir` help 与 results `line` 字段的
   "v2-0001..0006" 陈旧文本 → 已改为 0001..0007；`decoding_backend` 静态
   标签 "vllm-native-greedy" 无条件写入 → 改为按 temperature 派生
   （greedy main vs sampling diagnosis）。
3. 归档修正：两个已完成 run 的 `results.json` 中受陈旧脚本影响的字段
   （`line`、`decoding_backend`）已就地原子改写为真实值并记录于此；两 run
   实际均运行在 0001..0007 树上（门禁逐字节验证），标签仅为旧脚本的
   元数据残留。

训练侧（v2-0007、dp_actor.py、run 脚本 REBUILD 门禁、D13 测试）本轮零改动。

## 6. 声明边界

- **单次 greedy 运行**：78/256 与 GRPO10 的 74/256 的 Wilson 95% 区间重叠，
  结论为"方向性正面 + 当前第一名"，不是显著性声明；如需显著性需要多 seed /
  更大评测集（final-confirm512 不在授权范围，未触碰）。
- **musique 两个评测均为 0**（greedy 0/16、sampling 0/20）：多跳组合查询在
  本线仍是最弱源，行为诊断项，本轮不自动调参。
- **24/256 未作答**（greedy）：与 Step0 的 254/256 作答相比有差距，属于
  行为变化（搜索更积极但部分轨迹未收敛到 `<answer>`），诊断项。
- **offline rescore mismatches**：greedy 6/256、sampling 8/320，属于
  检查器对"不可离线重算的搜索步"的诚实标注（与 smoke S6 相同的机制），
  不改变主指标。
- 本轮未触碰（授权范围外）：reward 系数、batch 尺寸、模型、Prompt、
  projection、GPU 拓扑、final-confirm512；不启动 Step10/50、不启动 GiGPO。
- 训练内搜索率等行为曲线见行为 run 的 audits/checkpoints；与
  `docs/P3_V2_DEADLOCK_ROOTCAUSE_FIX_2026-08-23.md`（死锁根因 + smoke 11/11）
  共同构成完整证据链。
