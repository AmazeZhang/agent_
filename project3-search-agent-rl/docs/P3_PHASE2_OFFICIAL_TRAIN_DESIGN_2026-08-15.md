# P3 第二阶段设计：3B 官方宽松语义复现训练（多卡 FSDP）

**日期**：2026-08-15（第一阶段官方模型验证 PASS 之后）
**状态**：设计稿，供审阅；**未启动任何训练**。
**依据**：`docs/P3_OFFICIAL_CHECKPOINT_PREREG_2026-08-15.md`（PASS，p=0.0357）；
`docs/P3_EXPERIMENT_LINES_2026-08-15.md`（官方宽松语义线定义）。

## 0. 目标与用户拍板约束

- 目标：用官方宽松语义在真实 Wiki-18 retriever 上训练 Qwen2.5-3B，Step 0/50/100/300
  门禁（Step 50/100 停训→GPU1 评测→续训），最终 checkpoint 在盲测
  final-confirm512 上预注册检验。
- 硬约束（用户给定，2026-08-15）：
  1. **主线全参数 FSDP**，不默认 LoRA；LoRA 仅作显存失败后的降级方案；
  2. **global train batch 最低 64、目标 128**；**rollout.n=5**；不接受 batch 8；
  3. 物理 GPU **1,2,3,4,6,7**（6 卡），GPU0/5 禁用；
  4. Step 50 正常退出后用 GPU1 评测，**不与训练并发**（不压 retriever）；
  5. `official-confirm256-v1` 降级为 **dev 集**（可反复用于中期门禁）；
     最终 checkpoint 必须在**新建盲测 `final-confirm512`**（512 题）上做一次
     预注册检验；
  6. 执行顺序：设计 → 官方训练语义实现 → 3B 单步显存 → 六卡 1 步+恢复 →
     Retriever 并发测试 → 冻结 resolved config → 预注册 → Step 0–50。

## 1. 训练语义：official-loose（训练侧）

评测侧宽松语义已就绪（`run_p3_eval_vllm_official.py`：raw action 直达 skyrl
SearchEnv、无投影无惩罚、format_score=0.1）。训练侧由代码核查确认以下事实：

| 严格线现状（代码位置） | 官方宽松线改动 |
|---|---|
| `agent_system/environments/env_manager.py:613-618` `make_envs` 硬编码 `projection_f = partial(search_projection)`，manager 先投影再执行，`valids` 写入 `is_action_valid` | **vendor patch 0005**：`make_envs` 按 `config.env.projection`（默认 `strict`）选择投影；`projection=official` 时用透传投影 `(actions, [True]*len(actions))` —— raw action 直达 skyrl SearchEnv，valids 恒 true（宽松线无无效概念，与评测侧一致） |
| `verl/trainer/ppo/ray_trainer.py:1224-1227` 惩罚应用，由 `actor_rollout_ref.actor.use_invalid_action_penalty`（默认 True）+ `invalid_action_penalty_coef`（-0.1）控制 | **纯 config**：`use_invalid_action_penalty=false` → 无惩罚，**不改代码** |
| env reward 内 `format_score=0.1`（patch 0004） | 不变（官方论文口径，训练与评测共用 env） |

**结论**：训练侧宽松语义 = patch 0005（约 10 行）+ 两条 config。valids 恒 true
的副作用（`valid_action_ratio` 恒 1.0 的日志指标）可接受，在宽松线语义下该指标无意义。

## 2. 多卡架构（verl 原生支持，无 fork 架构改动）

- `trainer.n_gpus_per_node=6`、`trainer.nnodes=1`：RayResourcePool 起 6 个 worker
  （`verl/trainer/main_ppo.py:131`，已核验）。
- 每 worker：actor FSDP shard + ref FSDP shard + **独立 vLLM rollout engine**
  （`tensor_model_parallel_size=1`，V0 引擎，与评测/rollout 同路径）。
- GRPO 组大小：fork 硬约束 `actor_rollout_ref.rollout.n==1`
  （`main_ppo.py:173`）保持；`config.env.rollout.n=5` → `group_n=5`
  （`env_manager.py:606-614`，每 prompt 5 个平行 env 采样 5 条 rollout）。

## 3. Batch 语义（global samples = prompts × group_n）

| 档位 | prompts/step | × group_n=5 | samples/step |
|---|---|---|---|
| 最低 | 13 | 5 | **65**（≥64 ✓） |
| 目标 | 26 | 5 | **130**（≥128 ✓） |

- `data.train_batch_size` = prompts 数（13 起步 → 26 目标）；`env.rollout.n=5`。
- `ppo_mini_batch_size=65/130`（GRPO 全量 mini）、`ppo_micro_batch_size_per_gpu=1`。
- 优化步吞吐：130 samples × ~2304 tokens ≈ 300k tokens/step；6 卡 FSDP + 6 vLLM
  engine，预计每步 5–10 分钟（3B 全参，60-70% 时间在 rollout/retrieval）。
  Step 50 预计 4–8 小时（含 2 次停训评测）。

## 4. 显存预算（每卡 24GB，4090D；全参数主线）

| 分量 | 每卡占用（FSDP 6 卡分片） |
|---|---|
| actor 权重+梯度+Adam（3.06B 全参，bf16 权重 + fp32 Adam） | ~7.6 GB（6.1+3.1+36.7 总 /6） |
| ref 模型（bf16，无优化器） | ~1.5 GB |
| vLLM rollout（3B bf16 权重每卡全量 6.1GB + KV cache） | ~12.3 GB（gpu_mem 0.5×24） |
| 激活（micro 1 × 2304） | ~1–2 GB |
| **合计** | **~22–23 GB（临界）** |

风险与降级（按顺序尝试，均在"3B 单步显存"步骤实测决定）：
1. `gpu_memory_utilization` 0.6 → 0.5 → 0.45（KV cache 缩小，rollout 批变小但 step 数不变）；
2. `param_offload=true`/`optimizer_offload=true`（CPU 换显存，速度损失 ~20–40%）；
3. **LoRA 降级**（用户指定兜底）：`lora_rank=32` + `target_modules=all-linear` +
   现有一切 LoRA 配置（param_offload=true），batch 约束不变；仅当全参数在
   6 卡 24GB 上不可行时启用，并在 resolved config 与预注册中明确标注降级。

## 5. 数据

- `datasets/searchr1-upstream/train.parquet`（169,615 行 = NQ 79,168 + HotpotQA
  90,447；即官方训练集 `PeterJinGo/nq_hotpotqa_train` b7d80ab）全量作为 prompt 池，
  `data.shuffle=false` 顺序取（与现有训练一致）；`filter_overlong_prompts=true`。
- 50–300 步 × 65–130 prompts 消费 ≤39k 条，远小于池容量，无重复窗口问题。
- val：dev 集评测不依赖训练 val 路径（中期门禁用独立 eval 入口 +
  `official-confirm256-v1`；训练侧 `val_before_train=false`、`test_freq=-1` 关闭
  训练内评测，避免与评测线混用）。

## 6. 训练入口与 gates（独立 wrapper，不动严格线）

- 新脚本 `scripts/run_p3_grpo_official_exp.sh`（镜像 `run_p3_grpo_fix_exp.sh`
  结构），overrides 差异：
  - `trainer.n_gpus_per_node=6`；`actor_rollout_ref.actor.fsdp_config.param_offload=false`、
    `optimizer_offload=false`（全参数主线）；
  - `env.rollout.n=5`；`data.train_batch_size=13`（起步，目标 26）；
  - `env.projection=official`（patch 0005）、`actor_rollout_ref.actor.use_invalid_action_penalty=false`；
  - `actor_rollout_ref.model.path=models/Qwen2.5-3B`（本地已有）、
    **不带 lora_rank**（全参数；降级路径再带）；
  - `trainer.experiment_name=p3_grpo_official_3b_fsdp6_loose_n5_s0`；`save_freq=1`；
  - 其余（V0、max_model_len 2304、topk 3、timeout 180、lr 等）沿用训练基线，
    官方超参（lr/kl/optimizer 细节）落地前从官方 Search-R1 配置提取核验并写入
    resolved config。
- gate：`CUDA_VISIBLE_DEVICES` 必须是 `1,2,3,4,6,7`（**禁止含 0 或 5**，且卡数=6）；
  patch 0005 已应用；retriever health；veRL commit pin 20bd331b；
  `run_managed.sh` 受管（已有 gpu_ids 列表支持）+ preflight 逐卡空闲检查。
- vendor patch 0005：`0005-search-env-loose-projection.patch`（新文件，含
  单元测试或与 0001-0004 同等待验证）。

## 7. 分段运行编排（Step 0/50/100/300）

| 步 | 动作 | 评测 | 判定（预注册时细化） |
|---|---|---|---|
| Step 0 | 基线 | 已有 Base-3B 官方线 dev 结果（20/256，official-confirm256-v1）直接作为基线（受管、SHA 在案） | — |
| Step 50 | 训练正常退出（checkpoint global_step_50）→ **停训** → GPU1 评测 | dev 集，官方宽松线 eval 入口 | 与 Step 0 配对 McNemar：正向且 p<0.05 → 继续；无变化 → 诊断（预注册固定） |
| Step 100 | `trainer.resume_mode=resume_path` 续训至 100 → 停训评测 | 同上 | 趋势一致（Step 50/100 均正向或 Step 100 更强）→ 继续至 300 |
| Step 300 | 主门禁，最终 checkpoint | **final-confirm512 盲测**（新建，排除全部已用集，预注册配对检验） | PASS/INCONCLUSIVE/FAIL-TO-OBSERVE 三档（预注册固定） |

- resume 依赖 verl 原生 `resume_from_path`（严格线已用）；
  六卡 resume 在"六卡 1 步+恢复"步骤先行验证。
- 停训→评测→续训全程由 run_managed 受管；GPU1 在训练期间被训练占用时评测
  只在停训窗口进行（不并发）。

## 8. Retriever 并发方案

- 压力面：6 卡 × 每卡 envs。训练 130 samples/step（26 prompts × 5 group）时
  并发检索 ≈ 130 环境（+val 16）——低于 256（已 wedged 观察），但高于
  32（已验证安全负载）。24 线程 CPU retriever 是否扛得住 130 并发需实测。
- "Retriever 并发测试"步骤（六卡 1 步之后）：用真实 retriever 短跑 1 步，
  判据：0 超时 + p99 检索延迟可接受 + health 存活。不通过则依次：
  a) retriever 扩容（`serve_p25_cpu_retriever.py` OMP/线程 24→48，CPU 核数先确认）；
  b) 训练侧 env 检索并发限流（保留语义，仅并发控制）；
  c) 再评估 batch（用户约束 batch≥64 是下限，不降）。

## 9. 执行序列与每步验收（用户给定顺序）

| # | 步骤 | 验收 |
|---|---|---|
| 1 | 本设计（提交供审阅） | 用户批准设计 |
| 2 | 官方训练语义实现（patch 0005 + wrapper） | CPU 逻辑测试 + 严格线不受影响（现有测试全绿） |
| 3 | 3B 单步显存 | 单卡（GPU1）跑通 1 优化步（全参数 FSDP 6 卡配置的单卡模拟或预检 nvidia-smi），峰值显存 < 22GB；失败则按 §4 降级顺序 |
| 4 | 六卡 1 步 + 恢复 | GPU 1,2,3,4,6,7 跑 1 步 → 正常退出 → resume 续跑 1 步成功 |
| 5 | Retriever 并发测试 | §8 判据 |
| 6 | 冻结 resolved config | config 快照 + SHA 记录（含超参来源核验） |
| 7 | 第二阶段预注册 | 提交（先于任何 Step 50+ 评测）；含 Step 50/100/300 判据 + final-confirm512 盲测协议 |
| 8 | Step 0–50 训练 | 受管 6 卡；Step 50 停训 → GPU1 dev 评测 → 判定 → （通过则继续 100/300） |

## 10. 开放项（在落地步骤中逐一关闭）

1. 官方 Search-R1 训练超参（lr、KL 系数/方式、warmup、optimizer）：从官方
   verl 配置/论文提取，写入 resolved config（§9-6）；
2. 全参数 6 卡显存实测（§4 预算是否成立）；
3. retriever CPU 核数与扩容可行性（§8）；
4. final-confirm512 构建（domain `searchr1-p3-final-confirm-v1`，排除
   dev32/confirm256/official-confirm256-v1/上游 train，512 题配额放大）；
5. Step 0 基线的复用 vs 重跑（预注册中固定：复用受管 Base 结果，run id/SHA 在案）。

## 11. 声明边界

- 本设计是官方宽松语义线的训练入口设计；严格 fork 线（LoRA + 投影 + 惩罚）
  全部现有产物不动；
- 训练成功与否由预注册门禁判定，本设计不预设结果；
- final-confirm512 盲测是最终 Checkpoint 的唯一验收，dev 集数字不作终审。
