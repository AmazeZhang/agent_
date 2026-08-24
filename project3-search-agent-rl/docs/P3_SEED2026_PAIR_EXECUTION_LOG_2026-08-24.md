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

## Aware-v2 GRPO seed2026：运行中快照

- Run：`p3-aware-v2-grpo10-seed2026-fsdp6-20260824a`
- 启动：2026-08-24 15:47:19（Asia/Shanghai）。
- tmux：`p3-p3-aware-v2-grpo10-seed2026-fsdp6-20260824a`。
- Retriever tmux：`p3-aware-v2-seed2026-retriever-20260824`；CPU-only，Wiki-18
  IndexFlatIP，768维，21,015,324 vectors，`max_concurrent_queries=64`。
- 配置：Qwen2.5-3B-Instruct fresh Step0，全参数 FSDP；seed `2026/2026`；10 steps；
  train batch 66、rollout n=5、mini batch 330；max steps/history 4/4；save freq 5；
  lr `1e-6`、KL `0.001`、warmup ratio `0.285`；v2 step reward 与 trajectory return 开启。
- 物理 GPU：仅 `1,2,3,4,6,7`；6 个 Ray Worker 已一一落卡；GPU0 只有桌面进程，
  GPU5 未使用。
- 2026-08-24 17:13:54 快照：训练进度 3/10；已原子保存 `1..3.jsonl` 与
  `1..3.audit.jsonl`。三步 audit 均为 330 trajectories；搜索轨迹数分别为
  266、273、258；未发现 Traceback、OOM、CUDA/NCCL/Xid、Ray Worker failure 或检索超时。

此节仅是运行中工程状态，不是最终验收或质量结论。训练完成后必须追加 exit code、10步曲线、
v2 search-behavior 曲线、checkpoint/audit/资源清理验收及 SHA256；随后再 merge gs10 并执行
预注册的 confirm256 clean-vs-aware 逐题配对评测。
