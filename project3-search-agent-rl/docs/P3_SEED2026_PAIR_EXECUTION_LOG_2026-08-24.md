# P3 Search-aware v2 seed2026 配对训练执行记录（2026-08-24）

## 记录与提交约定

每个训练、合并和评测阶段均在阶段结束后追加：配置与 Run ID、物理 GPU 映射、起止时间、
退出与强制验收、核心指标、曲线/日志路径及 SHA256、异常与声明边界；随后提交并推送 Git。
Checkpoint、原始 rollout、audit、Ray 日志和训练曲线等大文件保留在
`/media/imc/data/project3-search-agent-rl/`，Git 只提交可审阅的代码、计划、摘要、路径和哈希，
不复制大体积实验产物。

## Clean GRPO seed2026：10步完成

- Run：`p3-clean-grpo10-seed2026-fsdp6-20260824a`
- 时间：2026-08-24 11:59:58–15:40:02（Asia/Shanghai），总计 3:40:04。
- 物理 GPU：`1,2,3,4,6,7`；GPU0 禁用，GPU5 未使用。
- 结果：10/10 steps，`exit_code=0`；`global_step_5`、`global_step_10` 完整；gs10
  各含 6 个 model、optimizer、extra-state FSDP 分片和 `data.pt`；无 partial/incomplete 文件。
- 10步训练批次均值：`critic/score/mean=0.2437`、`episode/reward/mean=0.2812`、
  `episode/success_rate=0.2812`、`episode/tool_call_count/mean=1.2013`、
  `episode/length/mean=2.2013`、`episode/valid_action_ratio=0.8211`。
- 第10步 success/reward 为 0.191，第9步为 0.345，存在明显 batch 波动；这些是 on-policy
  训练批次指标，不能替代 confirm256 held-out 结果。
- 退出末尾一条 Worker SIGTERM 出现在 gs10 指标与所有分片保存之后；主进程 exit 0、训练进度
  100%、目标 GPU 全部回到 18 MiB 且无残留，因此记录为关闭阶段 Ray 清理噪声，不作为训练失败。

曲线目录：
`/media/imc/data/project3-search-agent-rl/runs/p3-clean-grpo10-seed2026-fsdp6-20260824a/training_curves/`

| 文件 | SHA256 |
|---|---|
| `index.html` | `5a0a7d17a3398207ec6bcf37a9c50491cee1bbb45ad03fe3b85601e67c6c3318` |
| `metrics.csv` | `b5512cb2521db0015c32e0fcd232c142478edc8d223df2ff669816594d6cb5a8` |
| `summary.json` | `12e59a07f7cf5c76bb4af311acf7270fca01461c7a5a70711d85920f48c06bef` |
| `training_overview.svg` | `4c982e4c671abd077f3a3207956be6027c9c4959df19d9dff24cf85325c1d4da` |
| `training_system.svg` | `9bb0a505f1743a267e6e9a061c9c1a018be1264a105b1686315257ca70cdb5e8` |

## Aware-v2 GRPO seed2026：10步完成

- Run：`p3-aware-v2-grpo10-seed2026-fsdp6-20260824a`
- 时间：2026-08-24 15:47:19–19:35:39（Asia/Shanghai），总计 3:48:20；训练循环
  10/10 用时 3:45:30。
- tmux：`p3-p3-aware-v2-grpo10-seed2026-fsdp6-20260824a`。
- Retriever tmux：`p3-aware-v2-seed2026-retriever-20260824`；CPU-only，Wiki-18
  IndexFlatIP，768维，21,015,324 vectors，`max_concurrent_queries=64`。
- 配置：Qwen2.5-3B-Instruct fresh Step0，全参数 FSDP；seed `2026/2026`；10 steps；
  train batch 66、rollout n=5、mini batch 330；max steps/history 4/4；save freq 5；
  lr `1e-6`、KL `0.001`、warmup ratio `0.285`；v2 step reward 与 trajectory return 开启。
- 物理 GPU：仅 `1,2,3,4,6,7`；6 个 Ray Worker 已一一落卡；GPU0 只有桌面进程，
  GPU5 未使用。
- 结果：10/10 steps，`exit_code=0`；`global_step_5`、`global_step_10` 完整；gs10
  各含 6 个 model、optimizer、extra-state FSDP 分片和 `data.pt`；latest iteration 为 10；
  无 partial/incomplete 文件。
- 10 个 rollout 与 10 个 audit 均完整。每步严格为 330 trajectories；全部步骤
  `duplicate_identity_count=0`、`padding_records=0`，曲线生成器未报告 audit failure、重复
  metric step 或非有限数。
- 10步训练批次均值：`critic/score/mean=0.1670`、`episode/reward/mean=0.2909`、
  `episode/success_rate=0.2897`（NQ 0.2636、HotpotQA 0.3133）、
  `episode/tool_call_count/mean=1.2738`、`episode/length/mean=2.2738`、
  `response_length/clip_ratio=0.0774`、`actor/kl_loss=0.0110`、
  `actor/grad_norm=1.1484`。
- 第9步 success/reward 为 0.373/0.370，第10步为 0.233/0.230；仍有明显训练批次波动，
  不据此声明准确率提升。
- 搜索轨迹率从第1步 0.806 波动到第10步 0.894；第10步 useful-search rate 0.391、
  true-redundant rate 0.119、invalid-search rate 0.009、reached-step4 rate 0.130。
  这些是在线训练行为，不替代 fixed confirm256 配对评测。
- 无 Traceback、OOM、CUDA/NCCL/Xid、Ray Worker failure、NaN/Inf 或 Retriever timeout；
  训练末尾明确记录 TaskRunner、register center 和 6 个 Ray Worker graceful stop。清理日志确认
  物理 GPU `1,2,3,4,6,7` 均无 compute process，显存回到 18 MiB；GPU0仍只有桌面进程。
- 监控缺口：本轮 `peak_memory_nvidia_smi.json` 未生成。训练日志仍保留 veRL allocator
  观测，但不能将其冒充逐物理卡峰值；后续需单独修复 sampler 的 SIGTERM/final-write 逻辑。

曲线目录：
`/media/imc/data/project3-search-agent-rl/runs/p3-aware-v2-grpo10-seed2026-fsdp6-20260824a/training_curves/`

| 文件 | SHA256 |
|---|---|
| `index.html` | `c70ebc970fc807619216a97f5485242c6293501783f47d37d884286c57b6a510` |
| `metrics.csv` | `6397e425bdd9b397bcbf930a3db8481890c0bc4aa12d8cc318e9cfec1875ed6b` |
| `search_behavior.csv` | `cd377fd2a5760ac36e54fe9ba9c58f97e8e7fa98ce15d3fc7208f7939b7a1e43` |
| `search_behavior.svg` | `ede8684ab5b7edc0414f2daef96e8ff802570bcab4280ad51a63c7914b4b7971` |
| `summary.json` | `af62dc56d36b7cc39d92035f7f065d35e9705dca4e40d1379ad6294b2cb8538c` |
| `training_overview.svg` | `7292d8ac09789c95d4cffc5def923a8a08c9e0de93b6e0c6186f8b8632f6d095` |
| `training_system.svg` | `07a9c36afea217310414f521fef990b6b3cebf6f2279bad751594edf686b410a` |

关键证据 SHA256：`stdout.log=6d90b4d25393d2143125bc001c4eb8b45aaf5752106bf36d659acb94b5802ea8`，
`stderr.log=ca066214ef6742c05dd210903fea1c81b105dd4fd7618219966e36065f4d078c`，
`metadata.env=c43b7215675f61b7baf305ddfe5a86d2442ac374edb035cc0707594f4996424f`，
`cleanup.log=0de105b0d489d519939a89a7a7c8792299a8b1392f423f09b4070058e9c69967`。

当前工程训练闭环完成，但质量结论仍待：merge clean/aware gs10，并按预注册协议在同一
official-confirm256-v1 上执行逐题配对评测。
