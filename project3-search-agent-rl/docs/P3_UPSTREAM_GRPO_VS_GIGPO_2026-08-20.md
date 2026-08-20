# P3 干净 upstream GRPO vs GiGPO：Instruct Step0 评测与训练启动（2026-08-20）

状态：**双 1-step smoke 均 PASS，GRPO10 训练进行中（GiGPO10 串行等待）**。
Search-aware Reward 线（patch 0007/0008 及 0009）已按用户指令冻结；本线为
干净 upstream 20bd331b（无 0001-0009 任何补丁）+ Qwen2.5-3B-Instruct 的
算法对照实验，唯一算法变量为 `algorithm.adv_estimator=grpo|gigpo`。

## 1. 前置资产

- **Qwen2.5-3B-Instruct 下载**（2026-08-20）：HF 直连超时、hf-mirror CDN
  ~13MB/min 太慢，最终改用 **ModelScope CDN**（cdn-lfs-cn-1.modelscope.cn，
  ~3-7MB/s）。SHA256 与 hf-mirror 侧 LFS hash **交叉验证一致**（模型内容相同）：
  - `model-00001-of-00002.safetensors`
    `67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2`
  - `model-00002-of-00002.safetensors`
    `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d`
  - `model.safetensors.index.json`
    `bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56`
- **verify 门禁适配 raw-HF 布局**（commit 097ea45 + 4dc872a）：
  - Qwen2.5-3B-Instruct 官方 HF 布局**没有** `special_tokens_map.json` /
    `added_tokens.json`（added tokens 在 tokenizer_config.json 的
    added_tokens_decoder 中）→ 两文件改为可选（仅报告）；
  - tied embeddings：raw HF 无独立 `lm_head.weight`，transformers 物化副本
    共享 embed_tokens → 缺失检查允许该 key 并断言物化副本与 embed_tokens
    **完全相等**（`tie_lm_head_matches_embed_tokens`）；
  - 官方 Search-R1 checkpoint 回归仍 PASS；Instruct PASS（param_count
    3,085,938,688 = tied 期望值，NaN=0）。

## 2. 共享训练 wrapper（commit 61fc0f2）

`scripts/run_p3_grpo_gigpo_shared.sh`：唯一算法变量
`PROJECT3_ADV_ESTIMATOR=grpo|gigpo`（fail-closed，无默认）。

| 维度 | 值 |
|---|---|
| clean tree | `vendor/upstream-20bd331b`，pin 20bd331b + pristine + 无
  `search_aware_step_reward` 标记（clean-line 契约，与 eval wrapper 同构） |
| GiGPO overrides | `algorithm.gamma=0.95`、`gigpo.step_advantage_w=1.0`、
  `gigpo.mode=mean_std_norm`、`gigpo.enable_similarity=true`、
  `gigpo.similarity_thresh=0.9`（全部 hydra 解析验证） |
| GRPO overrides | `adv_estimator=grpo`、`gamma=1.0`（显式） |
| 共同配置 | `env.env_name=search`、`max_steps=4`、`history_length=4`、
  `env.rollout.n=5`（group，env_manager.py:609 读取）、
  `actor_rollout_ref.rollout.n=1`（main_ppo.py:159 硬断言）、上游
  `search_projection` / skyrl `compute_score` / SEARCH_TEMPLATE（不覆盖）、
  train_batch_size=66（DP6 整除）、mini_batch=330、FSDP 三 offload、
  gpu_mem=0.60、max_num_seqs=64、lr=1e-6、kl=0.001、warmup=0.285、
  CPU E5 Retriever、GPU 1,2,3,4,6,7、`trainer.resume_mode=disable` |
| 门禁 | 同 search_aware_v1：CPU mem gate、retriever health（vectors=21,015,324
  + max_concurrent_queries=64）、GPU 映射、managed env、shard 存在性 |

**tmux 变量透传修复**（commit 73ffc47）：tmux server 复用首个 session 环境，
原 start_tmux_run.sh 只透传 `PROJECT3_EVAL_*`；训练变量
（`PROJECT3_ADV_ESTIMATOR` 等）会静默丢失 → 改为透传全部 `PROJECT3_*`
（DATA_ROOT/MIN_FREE_GIB 保留显式导出）。

## 3. Step0（Qwen2.5-3B-Instruct，未训练）greedy 评测

run `p3-eval-upstream-clean-step0-instruct-confirm256-20260820a`（GPU1，
official-confirm256-v1 heldout.parquet 256 题，temperature=0、1 rollout，
PROJECT3_EVAL_MODEL/TOKENIZER 均指向 Instruct，exit 0，523s）。

| 指标 | Step0 Instruct | 官方 Search-R1 ckpt（对照） |
|---|---|---|
| EM（env 口径） | **65/256 = 25.4%** | 7/256 = 2.73% |
| 搜索率 | 177/256 = 69.1% | 256/256 = 100% |
| 搜索成功 | 333/333 = 100% | 685/685 = 100% |
| search→answer | 0.989（question 级） | 0.996 |
| search→correct | 0.232（question 级） | 0.027 |
| no_search→correct | 0.304（79 题闭卷） | — |
| answer_compliance | 254/256 = 99.2% | — |
| 422 / 空查询 | **0 / 0**（Instruct 无查询退化） | 修复后 0 |
| prompt gate | 177/177 PASS | 254/254 PASS |
| offline 一致 | 253/256 | 一致 |

分源 EM：triviaqa 43.8%、nq 32.8%、2wiki 31.3%、popqa 28.1%、bamboogle
18.8%、hotpotqa 12.5%、musique 0%。

**读法**：Instruct base 闭卷能力远超官方 checkpoint（25.4% vs 2.73%），但
搜索策略不完整（69% 搜索，30% 闭卷直答）。这正是 10 步 GRPO/GiGPO 要观察的
方向：搜索率是否上升、EM 是否维持/提升。

## 4. 双 1-step smoke（均 PASS）

| run | 结果 |
|---|---|
| `p3-grpo-gigpo-smoke-grpo-20260820a` | exit 0；global_step_1 完整
  （model/optim/extra × 6 rank + data.pt）；rollout 搜索成功 |
| `p3-grpo-gigpo-smoke-gigpo-20260820a` | exit 0；global_step_1 完整；
  457 次 Batch search 成功；step 683,127 tokens、adv timing 0.008ms/token
  （compute_advantage 走 gigpo 分支）、throughput 77.9 tokens/s |

smoke 后 worker SIGTERM 属 clean-upstream 关闭行为（patch 0003 不应用，
已知差异）；exit_code=0 且 checkpoint 完整。

## 5. 10 步训练（进行中）

串行执行（各需 6 卡已验证 FSDP 拓扑）：GRPO10 先行，GiGPO10 随后，**均从
同一 Instruct Step0 新启动**（resume_mode=disable，同一数据 seed 1234）。

| run | 配置 | 状态 |
|---|---|---|
| `p3-grpo-gigpo-10step-grpo-20260820a` | grpo，10 步，save_freq=5 | 进行中 |
| `p3-grpo-gigpo-10step-gigpo-20260820a` | gigpo，10 步，save_freq=5 | 等待 |

完成后：`scripts/model_merger.py` 合并 global_step_10 → confirm-256 评测
Step0/GRPO10/GiGPO10 → 报告 EM、搜索率、有效查询率、search→answer、
search→correct、GRPO reward 方差、GiGPO step-group 大小/有效比例、
episode/step advantage、资源状态。**10 步仅判断方向，不新增前置阶段，
不启动 50 步；结果出来后停止报告。**
