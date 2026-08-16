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

| 档位 | prompts/step | × group_n=5 | samples/step（ppo_mini_batch_size） |
|---|---|---|---|
| 最低 | **66** | 5 | **330** |
| 目标 | **132** | 5 | **660** |

- `data.train_batch_size` = prompts 数（66 → 132，用户审阅拍板值）；
  `env.rollout.n=5`；`ppo_mini_batch_size=330/660`（GRPO 全量 mini）、
  `ppo_micro_batch_size_per_gpu=1`。
- 优化步吞吐：330/660 samples × ~2304 tokens ≈ 760k–1.5M tokens/step；
  6 卡 FSDP + 6 vLLM engine，预计每步 10–25 分钟（3B 全参，60-70% 时间在
  rollout/retrieval）。Step 50 预计 8–20 小时（含 2 次停训评测窗口）。

## 4. 显存预算（每卡 24GB，4090D；全参数主线）

| 分量 | 每卡占用（FSDP 6 卡分片） |
|---|---|
| actor 权重+梯度+Adam（3.06B 全参，bf16 权重 + fp32 Adam） | ~7.6 GB（6.1+3.1+36.7 总 /6） |
| ref 模型（bf16，无优化器） | ~1.5 GB |
| vLLM rollout（3B bf16 权重每卡全量 5.79GB + KV cache） | 见 §4.1 实测 |
| 激活（micro 1 × 2304） | **实测 5.53 GiB**（见 §4.1；原 1–2 GB 估计已证伪） |
| **合计** | **~22–23 GB（临界）** |

### 4.1 六卡 smoke 实测（2026-08-16，详见 PROGRESS_SYNC）

- vLLM profiling 画像（6 卡一致，全驻留时）：weights 5.79 GiB + non_torch 0.01 GiB +
  **激活峰值 5.53 GiB**（profiling 批次 = max_num_batched_tokens 2304 全量进
  eager 模型，enable_chunked_prefill=false）= 非 KV 合计 **11.33 GiB**。
- KV 预算公式（每卡 23.99 GiB）：`gpu_mem_utilization × 23.99 − 11.33`。
  0.40 与 0.45 均实测失败（−1.73 / −0.53 GiB → 0 blocks → `initialize_cache`
  abort）；**未自动调整**（用户规则），报告后经用户批准改道。
- **官方架构适配（offload + gpu_mem=0.60 + max_num_seqs=64）成功**：
  profile `official-offload-smoke`（独立命名，不覆盖 0.40/0.45 记录）：
  - vLLM 画像（同配置下）：weights 5.79 / non_torch 0.01 / **activation 0.47 GiB** /
    **KV 7.83 GiB**；GPU blocks **14259** / CPU blocks 7281 / max concurrency 99x；
    init 3.75s。**注意**：0.47 GiB 是"offload + max_num_seqs=64"**组合配置**的
    观测结果——两个变量同时改变，不能严格宣称由 offload 单独导致（单变量归因未
    实测，见 §10 开放项）。
  - 各卡峰值（1s 采样）：GPU1 19,191 / GPU2 20,335 / GPU3 20,219 /
    GPU4 19,367 / GPU6 20,611 / GPU7 19,509 MiB（峰值在 optimizer/checkpoint
    gather 阶段，均 <85% 卡容量，无 OOM）。
  - 一次通过：exit 0、optimizer step 1 完成、checkpoint global_step_1
    （model+optimizer+extra_state world_size_6 + data.pt）完整、退出清理干净。
  - 显存机理（组合配置观测，非单变量归因）：训练阶段 VRAM 呈"engine asleep
    ~2.2 GB（参数在 CPU）→ wake 13–16 GB（rollout）→ 19–20.6 GB
    （optimizer/gather）"周期；与全驻留失败档相比，vLLM 画像中 activation
    峰值 5.53 → 0.47 GiB、KV cache 预算 −0.53 → +7.83 GiB。
- 全驻留 0.45 档实测峰值（对照）：GPU3 13,181 / GPU6 14,963 MiB（profiling
  瞬时尖峰），其余 ~9.3 GiB；abort 时稳定 ~9.3 GiB。

风险与降级（用户审阅拍板：**六卡 smoke 从 `gpu_memory_utilization` 0.40/0.45
起步**，不先试高值）：
1. 六卡 smoke 初始 `gpu_memory_utilization=0.45`（0.40 为第一观测点）；
   实测显存余量允许时再上调，**上调须记录**（resolved config 冻结时一并固定）；
2. `param_offload=true`/`optimizer_offload=true`（CPU 换显存，速度损失 ~20–40%）；
3. **LoRA 降级**（用户指定兜底）：`lora_rank=32` + `target_modules=all-linear` +
   现有一切 LoRA 配置（param_offload=true），batch 约束不变；仅当全参数在
   6 卡 24GB 上不可行时启用，并在 resolved config 与预注册中明确标注降级。
4. **不做单卡模拟六卡显存**（用户审阅删除）：显存验证只在真六卡 smoke 上进行。

## 5. 数据

- `datasets/searchr1-upstream/train.parquet`（169,615 行 = NQ 79,168 + HotpotQA
  90,447；即官方训练集 `PeterJinGo/nq_hotpotqa_train` b7d80ab）全量作为 prompt 池；
  **`data.shuffle=true` + 固定 `data.seed`（如 1234）+ `trainer.seed` 固定**
  （用户审阅拍板：shuffle 保证跨 epoch 不重复顺序依赖，固定 seed 保证可复现；
  verl `create_rl_sampler` 用 `data.seed` 固定 RandomSampler 生成器）；
  `filter_overlong_prompts=true`。
- 50–300 步 × 66–132 prompts 消费 ≤40k 条，远小于池容量，无重复窗口问题。
- val：dev 集评测不依赖训练 val 路径（中期门禁用独立 eval 入口 +
  `official-confirm256-v1`；训练侧 `val_before_train=false`、`test_freq=-1` 关闭
  训练内评测，避免与评测线混用）。

## 6. 训练入口与 gates（独立 wrapper，不动严格线）

- 新脚本 `scripts/run_p3_grpo_official_exp.sh`（镜像 `run_p3_grpo_fix_exp.sh`
  结构），overrides 差异：
  - `trainer.n_gpus_per_node=6`；`actor_rollout_ref.actor.fsdp_config.param_offload=true`、
    `optimizer_offload=true`、ref `param_offload=true`（全参数 FSDP + 状态 offload，
    2026-08-16 按已验证成功的架构从 false 切换，原 0.45/无 offload 形式已被实测
    拒绝）；
  - `env.rollout.n=5`；`data.train_batch_size=66`（正式档，目标 132）；
    `ppo_mini_batch_size=330`（目标 660）；
  - `data.shuffle=true` + `data.seed=1234` + `trainer.seed=1234`（固定，可复现）；
  - `env.projection=official`（patch 0005）、`actor_rollout_ref.actor.use_invalid_action_penalty=false`；
  - `actor_rollout_ref.model.path=models/Qwen2.5-3B`（本地已有）、
    **不带 lora_rank**（全参数；降级路径再带）；
  - `trainer.experiment_name=p3_grpo_official_3b_fsdp6_loose_n5_s0`；
    **`save_freq=50`**（正式；checkpoint 自然对齐 Step 50/100/300；
    smoke/恢复验证 profile 才覆盖为 `save_freq=1`，是验证工具而非正式配置）；
  - 四个 profile：`smoke`（`gpu_mem=0.40` 起点，`--max-train-steps 1` +
    `save_freq=1`，验证管道/显存/checkpoint 生成）、`official-offload-smoke`
    （官方架构适配：`gpu_mem=0.60` + actor/optimizer/ref offload +
    `max_num_seqs=64`，独立命名，不覆盖前两者记录）、
    `official-offload-resume-smoke`（**resume 验证**：与 offload-smoke 同架构，
    `total_training_steps=2` + `save_freq=1`，从源 global_step_1 恢复后只执行一次
    新更新到 global_step_2，`PROJECT3_RESUME_FROM` 必需且被 pin 到
    .../global_step_1，禁止继续到 Step 3）与 `formal`（正式配置，默认；
    save_freq=50；**当前 total_training_steps=50 仅为段配置，冻结前须按 §10-6
    拆分 300 步总调度长度**）；由环境变量选择，默认 `formal`；
  - 其余（V0、max_model_len 2304、topk 3、timeout 180、lr 等）沿用训练基线，
    官方超参（lr/kl/optimizer 细节）落地前从官方 Search-R1 配置提取核验并写入
    resolved config。
- gate：`CUDA_VISIBLE_DEVICES` 必须是 `1,2,3,4,6,7`（**禁止含 0 或 5**，且卡数=6）；
  patch 0005 已应用；retriever health；veRL commit pin 20bd331b；
  `run_managed.sh` 受管（已有 gpu_ids 列表支持）+ preflight 逐卡空闲检查。
- vendor patch 0005：`0005-search-env-loose-projection.patch`（新文件，含
  单元测试或与 0001-0004 同等待验证）。

## 7. 分段运行编排（Step 0/50/100/300）

| 步 | 动作 | 评测 | 判定 |
|---|---|---|---|
| Step 0 | 基线 | 已有 Base-3B 官方线 dev 结果（20/256，official-confirm256-v1）直接作为基线（受管、SHA 在案） | — |
| Step 50 | 训练正常退出（checkpoint global_step_50）→ **停训** → GPU1 评测 | dev 集，官方宽松线 eval 入口 | **开发门禁**（用户审阅拍板：不作统计显著性判定）：训练健康（正常退出、显存回基线）、行为变化（搜索协议遵守率、检索次数）、dev EM 趋势（相对 Step 0 方向性），只用于决定继续/诊断 |
| Step 100 | `trainer.resume_mode=resume_path` 续训至 100 → 停训评测 | 同上 | **开发门禁**：趋势一致性（Step 50/100 方向一致或 Step 100 更强）→ 继续至 300；任一阶段异常 → 停训诊断 |
| Step 300 | 主门禁，最终 checkpoint | **final-confirm512 盲测**（新建，排除全部已用集） | **唯一的确认性检验**（用户审阅拍板）：预注册配对检验，PASS/INCONCLUSIVE/FAIL-TO-OBSERVE 三档；dev 集数字不作终审 |

- resume 依赖 verl 原生 `resume_from_path`（严格线已用）；
  六卡 resume 在"六卡 1 步+恢复"步骤先行验证。
- 停训→评测→续训全程由 run_managed 受管；GPU1 在训练期间被训练占用时评测
  只在停训窗口进行（不并发）。

## 8. Retriever 并发方案（先全局限流，后压测；用户审阅拍板）

- 压力面：6 卡 × 每卡 envs。训练 330/660 samples/step（66/132 prompts × 5 group）
  时并发检索可达数百环境——已超过 256（已 wedged 观察）。**在六卡 smoke 之前**
  先做两件事：
  a) **服务端全局限流**（`create_app` 内实现）：全局并发检索上限
     `max_concurrent_queries`，超限请求在事件循环层排队（asyncio Semaphore，
     **惰性创建**——创建时绑定事件循环，否则绑定到不存在的循环会永久挂起，
     2026-08-15 实测修复）；/health 不受限；受管启动参数化。
  b) **压测**（CPU-only，无 GPU）：真实 retriever + 并发客户端
     （`scripts/stress_p25_retriever.py`），timeout=180（= 训练 env 超时）。
- **瓶颈诊断（2026-08-15 实测）**：单次检索 4.0s 与 OMP 线程数无关（8/24/48/96
  均 ~4.0s）——**内存带宽硬墙**（IndexFlatIP 每查询读 64.5GB），吞吐上限
  ~2.5 req/s。原 `search()` 全局锁会把并发全部串行化（0.23 req/s）；已拆锁：
  锁只包 encode（0.02s），faiss search（只读线程安全）放开并发。
- **压测矩阵（timeout=180）**：

  | 配置 | threads | limit | C=32 p99 | C=64 p99 | C=330 p99/max | 判定 |
  |---|---|---|---|---|---|---|
  | A | 24 | 32 | 17.2s | 34.6s | 177.3s / 179.9s | C=330 超时（FAIL） |
  | B | 8 | 64 | — | — | 144.0s / 146.0s | 0 超时（OK） |
  | C | 4 | 128 | — | — | 147.2s / 149.9s | 0 超时（OK） |

  **选定 B（threads=8, max_concurrent_queries=64）**：330 检索突发 p99 144s < 180s，
  吞吐 2.5 req/s；A 在 330 突发超限（线程过订阅 7920），C 无增益。
  服务已按 B 重启（2026-08-15，health 报告 max_concurrent_queries=64）。
- 训练侧语义不变（env 每次检索都是一次独立 HTTP 请求，排队在服务端完成，
  仅并发控制）；评测侧 `--max-envs-per-batch 32` 分块机制保持不变。
- 残余风险：一次 330 检索波 ~145s，180s 超时 margin ~35s；episode 第二步检索
  只发生在未终局 env 上（数量更小、且与模型重生成间隔错峰）。六卡 smoke 实测
  若仍有超时：a) 服务端再调参（OMP/limit）；b) 训练侧并发限流作为二次手段；
  c) 再评估 batch（用户约束 batch≥64 是下限，不降）。

## 9. 执行序列与每步验收（用户给定顺序）

| # | 步骤 | 验收 |
|---|---|---|
| 1 | 本设计（提交供审阅） | 用户批准设计（已批准步骤 2；本表按审阅意见更新） |
| 2 | 官方训练语义实现（patch 0005 + wrapper + retriever 全局限流/压测） | CPU 逻辑测试 + 压测报告 + 严格线不受影响（现有测试全绿） |
| 3 | **六卡 smoke（显存验证）** | 用户批准后执行；GPU 1,2,3,4,6,7 跑 1 步；**不做单卡模拟**（用户审阅删除）。2026-08-16 实测三段：**0.40 与 0.45 失败**于 vLLM `initialize_cache`（§4.1 画像，激活峰值 5.53 GiB 超预算，KV 0 blocks；均按用户指示停止、未自动调整）；**official-offload-smoke（0.60 + 全 offload + max_num_seqs=64）成功**（§4.1，一次通过，exit 0，checkpoint global_step_1 完整） |
| 4 | 六卡 1 步 + 恢复 | 用户批准（2026-08-16）：`official-offload-resume-smoke`，从源 global_step_1（p3-official-offload-smoke…a/checkpoints/global_step_1，清单+SHA 已记录）resume 至 global_step_2，只执行一次新更新；源只读、新 run 全新目录；通过标准见 PROGRESS_SYNC（日志标记/rank 四态恢复/游标 66→132/无 OOM/清理） |
| 5 | Retriever 并发压测 | §8 判据（全局限流已实现，压测选档） |
| 6 | 冻结 resolved config | config 快照 + SHA 记录（含超参来源核验） |
| 7 | 第二阶段预注册 | 提交（先于任何 Step 50+ 评测）；Step 50/100 为**开发门禁**（不设统计判据）、final-confirm512 为**唯一确认性检验**（预注册配对三档判定） |
| 8 | Step 0–50 训练 | 受管 6 卡；Step 50 停训 → GPU1 dev 评测 → 开发门禁判定 → （通过则继续 100/300） |

## 10. 开放项（在落地步骤中逐一关闭）

1. 官方 Search-R1 训练超参（lr、KL 系数/方式、warmup、optimizer）：从官方
   verl 配置/论文提取，写入 resolved config（§9-6）；
2. 全参数 6 卡显存实测（§4 预算是否成立；只经六卡 smoke，单卡模拟已删除）；
3. retriever 全局限流档位实测（§8 压测选档）；
4. final-confirm512 构建（domain `searchr1-p3-final-confirm-v1`，排除
   dev32/confirm256/official-confirm256-v1/上游 train，512 题配额放大）；
5. Step 0 基线的复用 vs 重跑（预注册中固定：复用受管 Base 结果，run id/SHA 在案）；
6. **LR schedule 分段问题（2026-08-16 记录，阻塞正式 Step 0–50 冻结，不阻塞
   本次 Step 1→2 resume 工程验证）**：正式目标总 300 步，warmup ratio 0.285
   ⇒ warmup ≈ 85 步。若 Step 0–50 段以 `total_training_steps=50` 创建 scheduler、
   resume 时再改为 100/300，前 50 步 warmup 曲线会偏离官方配置（fork 在
   `fsdp_workers.py:371-383` 按 `num_warmup_steps = int(ratio × total_steps)` 建
   scheduler，`ray_trainer.py:622-638` 把 `trainer.total_training_steps` 注入
   `actor.optim.total_training_steps`）。冻结 resolved config 前必须把**总调度
   长度 300**（scheduler/DataLoader epoch 语义）与**本段停止点 50/100/300**
   （段间停训评测）拆成两个独立配置概念（如 scheduler 恒用 300，段停止由单独
   的 stop-at-step 机制控制），再冻结。另：0.47 GiB activation 为组合配置观测，
   如需单变量归因须另行实验（§4.1）；formal 默认已切换为已验证架构
   （0.60/64/offload=true）。

## 11. 声明边界

- 本设计是官方宽松语义线的训练入口设计；严格 fork 线（LoRA + 投影 + 惩罚）
  全部现有产物不动；
- 训练成功与否由预注册门禁判定，本设计不预设结果；
- final-confirm512 盲测是最终 Checkpoint 的唯一验收，dev 集数字不作终审。
