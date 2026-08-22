# Search-aware GRPO 10-step 实验结果报告（2026-08-22）

## 0. 结论先行

**方向判断：C —— 搜索率持续下降，search-aware shaping 不足，搜索行为坍缩。**

- 10 步训练内搜索率 83.3% → 23.6%（useful-search 轨迹 169 → 40/330）；评测搜索率 41.0%（patched Step0）→ 24.2%，且 **search→correct = 0/62**。
- EM 从 patched Step0 的 8.2% 微升到 11.7%，但全部 30 个答对题均来自**闭卷**轨迹（no-search→correct 15.5%），无一条搜索轨迹答对。该提升不能视为 search-aware reward 的正面证据。
- Search-aware GRPO10 的绝对成绩（EM 11.7%）远低于 clean GRPO10（28.9%）与 clean GiGPO10（27.0%），也低于 clean Step0（25.4%）。
- 依据对照边界（第六部分）：patched 与 clean 在 prompt / projection / format 语义上有字节级差异，本结果**主要与 patched Step0 比较**，clean 两线仅作辅助参照，不声称"唯一变量只有 reward"。
- 按指令**不启动 50 步、不启动 GiGPO、不修改 final-confirm512**。下一步建议回到诊断（见 §7）。

## 1. 运行标识

| 项 | 值 |
|---|---|
| 训练 Run ID | `p3-search-aware-instruct-grpo10-fsdp6-b66-n5-s0-20260822a` |
| 评测 Run ID | `p3-eval-vllm-official-search-aware-grpo10-confirm256-20260822a` |
| 模型 | Qwen2.5-3B-Instruct（与 clean-upstream 对照相同的 Step0 起点，fresh 启动，未从任何 checkpoint 恢复） |
| 数据 | 同一 `train.parquet`（searchr1-upstream），data.seed=1234 / trainer.seed=1234 |
| 训练配置 | train_batch_size=66、rollout.n=5、samples/step=330、mini_batch=330、ppo_epochs=1、adv_estimator=grpo、gamma=1.0、lr=1e-6、kl=0.001、warmup=0.285、FSDP 三 offload、gpu_memory_utilization=0.60、max_num_seqs=64 |
| GPU | 训练 1,2,3,4,6,7（GPU0/5 禁用）；评测仅 GPU1 |
| 配置指纹 | `6e861cdeea1f5f4c0fa9fa2e07ddc2698e5076363c5b6f4281ac0f843da2c4c4` |
| 模型 SHA256 | `d00d0d7ff192d167d7e20c9cf297a5d5e6ce8dae7334ad7560f15422990ea7cb`（shard1）+ `391039c2e2fe86d4714525f0715dd31713a205dc6fe4f6c187a2c87f51ae5040`（shard2） |
| 合并与校验 | model_merger merge --backend fsdp；verify_p3_merged_model.py：**VERIFY_MERGED: PASS**（param_count 3,397,103,616、NaN/Inf=0、tie_lm_head_matches_embed_tokens=true） |
| 训练耗时 | 10/10 steps，3:28:05，943.76 s/it，exit 0 |
| 显存峰值 | max_memory_reserved 36.65 GiB / GPU（allocated 24.97 GiB，step 3 起稳定） |
| Retriever | 127.0.0.1:18080，IndexFlatIP 21,015,324 向量，评测全程 ready，无 timeout |

## 2. 冻结的 Search-aware v1 公式（未修改任何系数）

`R = R_answer + 0.15·evidence_hit + 0.30·(searched∧correct∧evidence_hit) − 0.20·invalid_or_error − 0.45·redundant_search_count − 0.20·new_answer_leak_in_query`；format_score=0.1；valid_retrieval 不加分；evidence_hit 只查真实 Retriever 正文；Observation token 不进入 policy loss；trajectory return 按 traj_uid 汇总，同题 5 条 trajectory GRPO 归一后广播到该轨迹全部 action record。

## 3. 训练内行为趋势（rollouts/1-10.audit.jsonl）

| 指标 | S1 | S5 | S10 |
|---|---|---|---|
| 搜索率（轨迹级） | 83.3% | 48.8% | **23.6%** |
| 有效查询率 | 98.6% | 97.2% | 94.6% |
| invalid 率 | 1.4% | 2.8% | 5.4% |
| 搜索轨迹数 / 330 | 275 | 161 | 78 |
| useful_search 轨迹 | 169 | 87 | 40 |
| closed_book 轨迹 | 55 | 169 | **252** |
| search_no_evidence | 106 | 74 | 38 |
| R_answer 总和（分） | 6900 | 8300（峰值） | 5200 |
| format 总和 | 1270 | 2010 | 2730 |
| evidence_hit 总和 | 2490 | 1215 | 570 |
| sce 总和 | 1650 | 1350 | 420 |
| invalid 惩罚总和 | -180 | -240 | -160 |
| redundant 惩罚总和 | -17505 | -7425 | -1485 |
| leak 惩罚总和 | -380 | -100 | 0 |
| useful_search mean adv | 0.523 | 0.588 | 0.929（仅 35 条可见） |
| closed_book mean adv | 0.895 | 0.624 | 0.585 |

趋势解读：redundant 惩罚（-0.45/次）随训练放大，模型学会**减少搜索调用**这一"廉价"规避；R_answer 在 S5 见顶后回落；到 S10 有效搜索轨迹只剩 40/330，尽管残留的 useful-search 轨迹 advantage 仍最高（+0.929）——即 reward 与信用分配没有失败在"有用搜索的梯度方向"上，而是被总数量的坍缩淹没。训练无 OOM/NaN/Xid/掉卡，8 分量与 step 放置和、trajectory return 一致性检查全程通过（fail-closed 未触发）。

## 4. 评测结果（official-confirm256-v1，greedy temperature=0，GPU1-only，256 题）

| 指标 | 值 |
|---|---|
| EM（env） | **11.7%（30/256）** |
| EM（offline rescore） | 11.7%（92 匹配 / 164 不匹配与 env 一致判定） |
| 搜索率（question 级） | **24.2%（62/256）** |
| 搜索调用总数 | 62（0.24 次/episode） |
| 搜索成功率 | 58.1%（36 success / 26 invalid_query；api_error=0、no_results=0） |
| search → answer | **0.0%（0/62）** —— 搜索轨迹无一产出答案 |
| search → correct | **0.0%（0/62）** |
| no-search → correct | 15.5%（30/194）—— **全部 30 个答对题均出自闭卷** |
| answer compliance | 75.8% |
| invalid/empty/redundant/leak | invalid_query 26；empty 0；redundant 0（official-loose 无冗余门）；leak 0 |
| reward 8 分量触发 | 评测协议无 step reward，计数为 0（预期） |

对比训练内轨迹（有效查询率 94.6%）与评测（success 58.1%）：greedy 解码下模型生成的查询格式退化严重（invalid_query 26/62），且搜索后无法收敛到 `<answer>`——评测中"搜索→永不安答"（search→answer=0）是比搜索率下降更严重的信号。

## 5. 四向表（含 patched Step0 基准）

| 指标 | clean Step0 | **patched Step0** | clean GRPO10 | clean GiGPO10 | **Search-aware GRPO10** |
|---|---|---|---|---|---|
| EM | 25.4%（65） | **8.2%（21）** | 28.9%（74） | 27.0%（69） | **11.7%（30）** |
| 搜索率 | 70.3% | **41.0%** | 62.9% | 54.7% | **24.2%** |
| search→correct | 0.228 | **0.000（0/105）** | 0.286 | 0.229 | **0.000（0/62）** |
| no-search→correct | 0.304 | **13.9%** | — | — | **15.5%** |
| answer compliance | 99.2% | **59.0%** | — | — | **75.8%** |
| invalid 率 | 0% | **22.9%** | — | — | **41.9%** |

## 6. 对照边界（第六部分结论）

patched 与 clean 存在**字节级确认的语义差异**：prompt（SEARCH_PROMPT_PREFIX vs SEARCH_TEMPLATE_NO_HIS）、projection（passthrough vs search_projection）、format_score（0.1 vs 0.0）。patched Step0 本身在评测上即远低于 clean Step0（EM 8.2% vs 25.4%，搜索率 41.0% vs 70.3%），因此：

- Search-aware GRPO10 的**唯一可比基准是 patched Step0**；
- clean GRPO10 / GiGPO10 仅作辅助参照，**不得**声称"唯一变量只有 reward"；
- 从 patched Step0 出发的相对变化：EM 8.2% → 11.7%（+3.5pp，全来自闭卷），搜索率 41.0% → 24.2%（−16.8pp），search→correct 保持 0。

## 7. 结论与建议

- **A-D 判断：C（搜索率仍下降 → shaping 不足返回诊断）**。不满足 A/B（搜索率与 search→correct 未提高）；EM 提升虽满足 D 的局部条件，但完全来自闭卷且搜索率不升反降，归入 C 并如实标注。
- Search-aware step reward + trajectory-return 信用分配**未能改善真实搜索行为**：惩罚项（redundant −0.45、invalid −0.20）主导了策略演化方向，模型学会"少搜"而非"搜得好"；同时评测暴露 greedy 下查询格式退化与搜索后不答问题两个独立问题（后者在训练内不可见，因为训练 rollout 用采样解码且 max_steps 语义相同但温度不同）。
- 建议下一步（需另行批准，本轮不执行）：诊断训练/评测解码差异与"搜索后不答"成因（greedy 退化 vs 策略坍缩）→ 若确认 shaping 问题，重新设计惩罚梯度（如 redundant 改 0.20、invalid 依赖格式学习）并以 patched Step0 为唯一基准做消融；在修复"搜索后必答"与查询格式之前，扩大步数无意义。

## 8. 工程状态与收尾核验

- 工程 smoke 13/13 PASS（run `p3-search-aware-instruct-eng-smoke-fsdp6-b66-n5-s0-20260822a`，exit 0，30:47）。
- 训练 exit 0、gs5+gs10 checkpoint 完整、audit 1-10 全落盘、无 OOM/NaN/Xid/掉卡、配置指纹与计划一致。
- 评测 exit 0，results.json + episodes.jsonl + audit.json 落盘。
- GPU 已全部回到基线（18 MiB；GPU0 407 MiB 为他人进程，与本项目无关）。
- 本轮相关文件：`scripts/audit_p3_eval_trajectories.py`、`scripts/audit_p3_search_aware_step.py`、本报告、PROGRESS_SYNC。
