# P3 Search-aware v2 多 seed 配对训练预注册（2026-08-24）

## 1. 目标

验证 Search-aware v2 相对 outcome-only clean GRPO 的效果是否能在第二个训练 seed
上复现。此阶段不改 reward、Prompt、projection、模型、数据、batch 或训练步数；唯一实验变量
是算法线（clean GRPO vs aware-v2 GRPO），随机性重复使用新的 trainer/data seed 2026。

## 2. 资源、时长与存储

- 模型：Qwen2.5-3B-Instruct，fresh Step0，全参数 FSDP。
- GPU：既有且已验证的物理 `1,2,3,4,6,7`，6×24 GiB；GPU0永久禁用，GPU5不启用。
- Retriever：CPU Wiki-18 E5，21,015,324 vectors，`127.0.0.1:18080`。
- 训练：每条线 10 steps，train batch 66，env rollout n=5，约 4 小时/条线。
- 先运行 aware-v2 seed 2026 的 1-step engineering smoke，预计约 25 分钟。
- 既有 10-step run 约 72 GiB；两条新主 run 预计约 144 GiB，另加 smoke 和评测。
  启动时要求数据盘至少 150 GiB 空闲；预注册时实际空闲约 1.7 TiB。
- CPU：预注册时 MemAvailable 约 925 GiB、swap 使用约 2 MiB，满足 offload 门禁。

## 3. 冻结配置

- trainer/data seed：`2026/2026`；env seed 继续固定为 0，与 clean 基线协议一致。
- total_training_steps=10，save_freq=5，warmup ratio=0.285。
- lr=1e-6，KL coefficient=0.001，GRPO gamma=1.0，ppo_epochs=1。
- train_batch_size=66，group n=5，mini_batch=330，max_steps=4，history=4。
- FSDP param/optimizer/ref offload 均开启，gpu_memory_utilization=0.60，max_num_seqs=64。
- 同一上游 `20bd331b`、相同数据和真实 Retriever。
- clean：outcome-only GRPO；aware：固定 v2 reward/trajectory-return 实现，不改系数。

## 4. 顺序与 Run ID

按顺序执行，前一步完成强制验收后才能晋级：

1. aware seed smoke，1 step：
   `p3-aware-v2-seed2026-smoke1-fsdp6-20260824a`
2. clean GRPO seed 2026，10 steps：
   `p3-clean-grpo10-seed2026-fsdp6-20260824a`
3. aware-v2 GRPO seed 2026，10 steps：
   `p3-aware-v2-grpo10-seed2026-fsdp6-20260824a`
4. 分别 merge gs10，并在 official-confirm256-v1 上执行 GPU1-only greedy 评测。

不并行启动两条 6 卡训练，不覆盖已有 run。当前用户指令授权适当的 aware 尝试；本预注册
不授权 20 步以上、GPU5、全量数据或扩大 GPU 拓扑。

## 5. Smoke 晋级门禁

- exit 0；checkpoint、rollout audit 完整且无 `.partial`。
- 330 trajectories，padding/duplicate identity 为 0；reward/return/advantage sum 一致。
- 无 OOM、NaN/Inf、Xid、NCCL collective divergence、worker loss 或 Retriever timeout。
- metadata 物理 GPU 只能是 `1,2,3,4,6,7`；结束后进程、Ray、端口和逐卡显存回基线。
- resolved config 除 seed 和 experiment/run identity 外与 reference ten-step 配置一致。

任一门禁失败则保留证据并停止，不自动降低 batch、改 reward 或换 GPU。

## 6. 主比较与声明边界

主指标：confirm256 EM 的逐题 clean-vs-aware 配对差异；同时报告搜索率、search-to-answer、
searched-and-correct、max-steps exhausted、invalid 和真实冗余搜索。精确双侧 McNemar p 值
全部报告。

第二个 seed pair 只提供重复性证据，不能单独建立论文级显著性。结合既有 seed 1234 结果，
如果 aware 的方向不稳定或仍不优于 clean GRPO，则结论为“机制成立、搜索行为改善，但准确率
改进未稳定复现”，随后才考虑针对停止/作答收敛做一次单变量 reward 或策略调整。
