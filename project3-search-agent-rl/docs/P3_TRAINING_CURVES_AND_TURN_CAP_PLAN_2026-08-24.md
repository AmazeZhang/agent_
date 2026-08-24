# P3 训练曲线落盘与四轮上限后续计划（2026-08-24）

## 1. 训练曲线产物链

新增 `scripts/generate_training_curves.py`，仅使用 Python 标准库，从受管 Run 的
`stdout.log` 与 `rollouts/*.audit.jsonl` 生成：

- `training_curves/metrics.csv`：verl 每个 optimizer step 的完整宽表；
- `training_curves/search_behavior.csv`：v2 audit 的逐步搜索行为聚合；
- `training_curves/training_overview.svg`：reward/success、policy/KL、entropy、
  gradient、tool call、episode/sequence length；
- `training_curves/training_system.svg`：rollout/log-prob/reference/update_actor 耗时、
  throughput、clip ratio 与内存视图；
- `training_curves/search_behavior.svg`：搜索率、evidence-hit、invalid、true-redundant、
  positive advantage、触及 Step4 比例、搜索轮数与 reward components；
- `training_curves/summary.json`：源日志 SHA256、step 完整性、NaN/Inf、重复 step、
  audit 解析失败和生成约束；
- `training_curves/index.html`：本地浏览入口。

`scripts/run_managed.sh` 在精确 Run 清理完成后自动调用生成器：

- 强制 `CUDA_VISIBLE_DEVICES=''`，不使用 GPU0 或任何训练 GPU；
- 评测/诊断 Run 无 `training/global_step` 时跳过；
- 输出目录在同一 Run 内一次性原子创建，若已存在则拒绝覆盖；
- 生成失败写 `curve_generation.log`/`metadata.env`，绝不改写训练命令的原退出码；
- 不依赖 matplotlib、W&B、TensorBoard 或联网安装。

当前 seed-2026 clean Run 在脚本挂钩落地前已经启动，因此退出验收时手动回填一次；之后启动的
aware-v2 seed-2026 Run 会自动生成。

### 历史回放验证

以下已完成 Run 已新增派生曲线（不修改原日志、rollout 或 checkpoint）：

1. `p3-grpo-gigpo-10step-grpo-20260820a`：10/10 metric steps；
2. `p3-grpo-gigpo-10step-gigpo-20260820a`：10/10 metric steps；
3. `p3-search-aware-clean-v2-grpo10-fsdp6-b66-n5-s0-20260823a`：
   10/10 metric steps + 10/10 audit steps。

回放 step10 `critic/score/mean` 分别为 `0.254 / 0.230 / 0.177`，与原报告一致；
解析无重复 step、无非有限指标、无 audit failure；全部 SVG 可被 XML parser 读取。

## 2. `max_steps=4` 的实际风险

当前 SearchEnv 在收到动作后先 `turns += 1`，再判断 `turns >= max_turns`，并在 done 后
直接返回、不执行工具。因此 `max_steps=4` 实际提供：最多三次 Retriever 调用 + 第四个模型
动作作为理想的 answer slot。第四步若仍为 search，文本虽可被 projection 保留，但不会发出
Retriever 请求。

这既是成本边界，也是可观测的强制终止偏差。当前不能在 clean/aware seed 配对中途修改，
否则算法线与搜索预算同时变化，失去归因。

## 3. 后续修复顺序（当前 pair 完成后另行预注册）

### A. 先补可观测性，不改变策略语义

给训练与评测 trace 增加 typed termination reason：

- `answer_submitted`；
- `max_steps_exhausted`；
- `eos_or_empty`；
- `invalid_action_terminal`（若适用）。

同时分别记录 projected search、实际 Retriever 请求和“终止步未执行 search”，禁止用
`env_step==4` 直接冒充 exhaustion。

### B. 再选择一个单变量协议修正

首选候选：保持最多三次真实检索，但在检索预算耗尽后提供一个明确、独立的 final-answer slot；
备选候选：只把 `max_steps` 从 4 提到 6。二者不能同轮一起改。

### C. 先 evaluation-only，再决定是否训练

在相同 checkpoint、confirm256、greedy、topk=3、Retriever 与 Prompt 下做配对：

- answer compliance / `max_steps_exhausted`；
- EM 与逐题 McNemar；
- 每题实际搜索调用、evidence-hit、true-redundant；
- search-to-answer、search-to-correct；
- token、wall time 与 Retriever 请求增量。

只有 exhaustion 明显下降且 EM 不退化，才考虑同 seed 的小步数训练消融。若只是增加冗余搜索
或计算成本，则保留 `max_steps=4`，把停止/作答收敛作为 reward 或策略改进，而不是扩大预算。
