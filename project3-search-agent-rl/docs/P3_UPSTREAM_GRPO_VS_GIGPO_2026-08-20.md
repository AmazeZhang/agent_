# P3 干净 upstream GRPO vs GiGPO：Instruct Step0 评测与 10 步对照（2026-08-20）

状态：**双 1-step smoke 均 PASS；GRPO10 与 GiGPO10 训练均完成（exit 0）；
confirm-256 三向评测（Step0/GRPO10/GiGPO10）完成；本线按指令停止报告。**
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

## 5. 10 步训练（均完成）

串行执行（各需 6 卡已验证 FSDP 拓扑）：GRPO10 先行，GiGPO10 随后，**均从
同一 Instruct Step0 新启动**（resume_mode=disable，同一数据 seed 1234）。
每步 ~24 min（timing_s/step 1133-1473s），10 步约 4 小时/线。

| run | 配置 | 结果 |
|---|---|---|
| `p3-grpo-gigpo-10step-grpo-20260820a` | grpo（γ=1.0），10 步，save_freq=5 | **exit 0**；global_step_5+10 完整；merge PASS |
| `p3-grpo-gigpo-10step-gigpo-20260820a` | gigpo（γ=0.95、step_adv_w=1.0、mean_std_norm、sim=true、thresh=0.9），10 步 | **exit 0**；global_step_5+10 完整；merge PASS |

- merge 产物（`scripts/model_merger.py --backend fsdp --local_dir <gs10>/actor`）：
  GRPO10 → `models/p3-grpo10-grpo-instruct-merged-20260820`（权重 SHA256
  `ed98e0fd…`/`ebe6a5e1…`）；GiGPO10 →
  `models/p3-grpo10-gigpo-instruct-merged-20260820`（`1e8a1816…`/`7ee74981…`）。
- verify_p3_merged_model.py 两模型均 PASS：param_count 3,397,103,616（独立
  lm_head 副本）、NaN=0、tie_lm_head_matches_embed_tokens ✓。
- 训练 reward 轨迹（critic/score/mean）：GRPO 0.181→0.254（峰值 0.318@step5）；
  GiGPO 0.182→0.230（峰值 0.318@step5）。episode reward 分布（rollout jsonl）：
  GRPO mean 0.173 / std **0.391**；GiGPO mean 0.182 / std **0.397**。
- advantage 动态范围（critic/advantages max/min）：GRPO max 2.1-3.5 /
  min −2.7~−4.1；GiGPO max 3.8-5.8 / min −4.5~−6.2 —— γ=0.95 步级 advantage
  放大生效（adv 计算走 gigpo 分支，smoke 已证 0.008ms/token）。
- valid_action_ratio 轨迹：GRPO 0.704→0.976；GiGPO 0.703→0.960。
- GiGPO step-group：名义 group=`env.rollout.n=5`，hydra 日志确认
  `enable_similarity=True, similarity_thresh=0.9, step_advantage_w=1.0`；
  **实际相似度聚类分布未落盘**（rollout jsonl 仅 input/output/score/step），
  无法报告逐组大小，以 advantage 范围 + valid_action_ratio 间接佐证。

## 6. confirm-256 三向评测（均 GPU1 greedy、temperature=0、1 rollout）

run：Step0 `p3-eval-upstream-clean-step0-instruct-confirm256-20260820a`；
GRPO10 `p3-eval-upstream-clean-grpo10-confirm256-20260820a`；
GiGPO10 `p3-eval-upstream-clean-gigpo10-confirm256-20260820b`。

| 指标 | Step0 | GRPO10 | GiGPO10 |
|---|---|---|---|
| **EM（env 口径）** | 65/256 = 25.4% | **74/256 = 28.9%** (+3.5pp) | 69/256 = 27.0% (+1.6pp) |
| 搜索率（episode 级） | 180/256 = 70.3% | 161/256 = 62.9% | 140/256 = 54.7% |
| 搜索成功 | 333/338 = 98.5% | 285/289 = 98.6% | 237/242 = 97.9% |
| search→answer（question 级） | 0.989 | 0.800 | 0.835 |
| **search→correct**（episode 级） | 0.228 | **0.286** (+0.058) | 0.229（持平） |
| no_search→correct | 0.316（76 题） | 0.295（95 题） | 0.319（116 题） |
| answer_compliance | 254/256 = 99.2% | 224/256 = 87.5% | 233/256 = 91.0% |
| steps/episode | 2.32 | 2.13 | 1.95 |
| 422 / 空查询 | 0 / 0 | 0 / 0 | 0 / 0 |
| prompt gate | 177/177 PASS | 159/159 PASS | 137/137 PASS |
| offline 一致 | 253/256 | 256/256 | 256/256 |

分源 EM：triviaqa 0.375/0.375、nq 0.375/0.328、2wiki 0.375/0.313、popqa
0.344/0.313、bamboogle 0.313/0.250、hotpotqa 0.156/0.188、musique 0.063/0
（GRPO10/GiGPO10 顺序）。

**读法（10 步方向判断）**：
1. 两算法 10 步后 EM 均上升，**GRPO 略胜**（+3.5pp vs +1.6pp）；GRPO 增益主要
   来自搜索题答对率提升（search→correct 0.228→0.286），GiGPO 增益主要来自
   更多闭卷直答（no_search 占比 30%→45%）。
2. **搜索率两线均降**（70.3%→62.9%/54.7%）：训练 reward 对闭卷快答信号更强
   （闭卷 EM 基线 ~0.30 高于搜索正确率），策略向少搜索漂移——与自训 Step300
   坍缩同向，10 步仅轻微漂移未坍缩（search→answer 仍 0.80+，搜索成功 98%）。
3. **GiGPO 未见超越 GRPO**：γ=0.95 步级 advantage 确已生效（动态范围翻倍），
   但 10 步内 EM 低 1.9pp。
4. offline 一致性 GRPO/GiGPO 达 256/256（优于 Step0 253/256）：训练后格式
   违规减少。

## 7. 资源与收尾

- 全部 GPU 回 18 MiB 基线（含 GPU1 eval）；训练/eval tmux session 均正常退出
  （exit 0 / status 0）；Retriever 未受影响（21,015,324 向量就绪）。
- 按用户指令：**10 步仅判断方向，不新增前置阶段，不启动 50 步；
  结果已报告，本线停止报告。**
