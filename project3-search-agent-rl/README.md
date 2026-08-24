# Search Agent RL：Search-R1 工程复现与 Search-aware 改进

> 项目最终状态：已在 veRL/verl-agent 上完成 Qwen2.5-3B-Instruct 的 Search-R1
> GRPO、GiGPO 工程复现，并完成 Search-aware v2 的两组种子对照实验。本项目定位是
> **秋招工程项目：可靠复现、问题诊断、适度改进与诚实评测**，不宣称论文级 SOTA。

## 最终阅读入口

- **学习与面试讲解主文档：**
  [`docs/FINAL_PROJECT_GUIDE_2026-08-24.md`](docs/FINAL_PROJECT_GUIDE_2026-08-24.md)
- seed2026 Clean/Aware 完整执行记录：
  [`docs/P3_SEED2026_PAIR_EXECUTION_LOG_2026-08-24.md`](docs/P3_SEED2026_PAIR_EXECUTION_LOG_2026-08-24.md)
- 训练曲线与四轮上限计划：
  [`docs/P3_TRAINING_CURVES_AND_TURN_CAP_PLAN_2026-08-24.md`](docs/P3_TRAINING_CURVES_AND_TURN_CAP_PLAN_2026-08-24.md)
- 可复算统计：
  [`gates/p3_seed2026_pair_stats_20260824.json`](gates/p3_seed2026_pair_stats_20260824.json)

## 项目结论

固定 `official-confirm256-v1`、greedy 解码和真实 Wiki-18 Retriever，对 Clean GRPO 与
Aware-v2 做逐题配对：

| seed | Clean GRPO | Aware-v2 | Aware−Clean | McNemar p |
|---|---:|---:|---:|---:|
| 1234 | 74/256 | 73/256 | −1 | 1.0 |
| 2026 | 77/256 | 78/256 | +1 | 1.0 |

两组种子的 EM 方向相反，**不能声称准确率稳定提升**。但 Aware-v2 的行为改善方向稳定；
seed2026 中有效搜索率 `71.1%→91.4%`、search→answer `80.8%→97.0%`、四步耗尽
`35→16`、真实冗余率 `19.9%→13.1%`。反事实检索还表明，Aware-v2 搜索后答对主要依赖
真实证据，而不是仅仅学会调用工具。

![两组种子与 seed2026 行为对比](docs/assets/p3_final_eval_comparison.svg)

## 技术方案

- 模型：`Qwen2.5-3B-Instruct`，全参数 FSDP；
- 框架：veRL/verl-agent，vLLM rollout，Ray 编排；
- 搜索：CPU Wiki-18 Retriever，E5 + `IndexFlatIP`，21,015,324 vectors，top-k=3；
- 训练：6 张稳定计算卡 `1,2,3,4,6,7`，物理 GPU0 永久禁用、GPU5 默认排除；
- 采样：每步 66 个 prompt，每题 5 条 rollout，共 330 条 trajectory，10 optimizer steps；
- 评测：独立 confirm256，temperature=0，逐题配对 McNemar；
- 改进：在不改变 clean prompt/projection/终止协议的前提下，为 GRPO 加入搜索证据奖励、
  无效/冗余/答案泄漏惩罚和 trajectory-return 信用分配。

GRPO 与 GiGPO 是先完成的两条**独立对照线**；最终 Aware-v2 使用的是 GRPO，不是把二者
同时叠加。256 题是 held-out 评测集，不是“只用 256 题训练”。

## 仓库结构

```text
vendor/verl-agent/            固定的上游实现
patches/                      可审阅的上游补丁
searchr1_repro/               奖励、审计与复算逻辑
scripts/                      受管训练、评测、合并和曲线脚本
gates/                        小型统计与审计结果
docs/                         预注册、执行记录、结果与最终讲解
```

大模型、Checkpoint、Retriever 索引、原始 rollout、日志和详细训练曲线保存在
`/media/imc/data/project3-search-agent-rl/`，不会提交到 Git。

## 安全门禁

任何启动、停止、恢复、下载、清理或扩容前，必须完整阅读 [`AGENTS.md`](AGENTS.md) 和
[`docs/EXPERIMENT_SAFETY.md`](docs/EXPERIMENT_SAFETY.md)。GPU 训练只能在只读预检后，
通过新 Run ID、命名 tmux 与 `scripts/run_managed.sh` 启动；禁止物理 GPU0、默认排除 GPU5，
禁止全局 `pkill`、`killall` 或 `ray stop --force`。

当前实验已经收尾，**无需继续训练**。`max_steps=4` 仍是已知限制，应作为后续工作说明，
不能表述为已经修复。
