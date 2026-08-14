# P3下一阶段：Step 0 / Step 2 / Step 5 Held-out评测执行计划

## 当前判断

截至2026-08-14，上一阶段已完成但尚未占用GPU：

- HF侧纯评测脚本、受管wrapper和10项专项测试已完成；相关CPU套件记录为33 passed；
- heldout-32已从上游test确定性构建，排除上游train、smoke train和smoke test问题，泄漏计数为0；
- heldout parquet SHA256为
  `1f8caca3255928baeac2aafb1b1c25445533426664a9d85c5519a4d6fab6d62f`；
- Step 2和Step 5的PEFT LoRA目录均存在；
- 宿主机只读检查显示物理GPU1为18MiB、数据盘约3.0TiB可用，无评测、Ray或Retriever进程；
- 受限执行环境中的`nvidia-smi`可能看不到驱动，必须以宿主机预检结果为准，不能据此启动或
  擅自修复驱动；
- 尚无Base/Step 2/Step 5的smoke或heldout评测结果。

因此下一步不是继续训练，也不是扩展多卡，而是执行已准备好的六个纯评测Run。

## 执行目标

在完全相同的HF贪心解码、真实Wiki-18 Retriever和环境/评分逻辑下比较：

1. Base模型（Step 0）；
2. Attempt G的Step 2 LoRA；
3. Attempt H的Step 5 LoRA。

先用smoke-16验证三条模型加载路径，再在heldout-32产生第一轮小样本正式对比证据。全阶段
不创建Optimizer/Scheduler、不调用Backward、不启动Ray，也不修改Checkpoint。

## 执行顺序

### Gate 0：启动前复核

- 完整阅读`AGENTS.md`和`docs/EXPERIMENT_SAFETY.md`；
- 重新执行宿主机`nvidia-smi`和`bash scripts/preflight.sh 1`；
- 确认物理GPU1没有Compute Process，物理GPU0未进入白名单；
- 确认端口18080无未知监听者，六个新Run ID及其tmux会话均不存在；
- 核对veRL commit、三个模型/Adapter SHA、数据manifest和磁盘空间；
- 若任一身份、端口、GPU或路径不清楚，停止并询问用户。

### Gate 1：真实CPU Retriever

使用全新、精确命名的tmux会话启动CPU Wiki-18 Retriever，显式
`CUDA_VISIBLE_DEVICES=''`，只监听`127.0.0.1:18080`。必须等`/health`返回
`status=ready`和`vectors=21015324`后再继续。不得复用状态不明的旧tmux会话。

### Gate 2：smoke-16管线门禁

严格串行运行，任一Run失败即停止，不继续后续Run：

| 顺序 | 建议Run ID | 模型 |
|---|---|---|
| 1 | `p3-eval-smoke-base-s0-20260814a` | Base / Step 0 |
| 2 | `p3-eval-smoke-step2-s0-20260814b` | Step 2 LoRA |
| 3 | `p3-eval-smoke-step5-s0-20260814c` | Step 5 LoRA |

每个Run都必须通过`start_tmux_run.sh → run_managed.sh → run_p3_eval_heldout.sh`，只映射物理
GPU1。逐Run核对exit code、16条episode、results/episodes原子文件、adapter身份、数据hash、
leakage=0、Retriever health、GPU回基线和无残留PID。smoke结果只作管线门禁，不能用于质量声明。

### Gate 3：heldout-32同条件对比

只有三个smoke Run全部合格后，才严格串行运行：

| 顺序 | 建议Run ID | 模型 |
|---|---|---|
| 4 | `p3-eval-heldout32-base-s0-20260814d` | Base / Step 0 |
| 5 | `p3-eval-heldout32-step2-s0-20260814e` | Step 2 LoRA |
| 6 | `p3-eval-heldout32-step5-s0-20260814f` | Step 5 LoRA |

三个Run必须使用相同数据SHA、seed、prompt、max steps、history、top-k、token上限、Retriever和
HF greedy backend。串行顺序可能影响OS缓存和耗时，因此耗时不作为模型效果指标。

### Gate 4：精确停止与资源验收

六个Run结束后，只向本阶段精确Retriever tmux会话发送Ctrl-C。确认Uvicorn完整关闭、端口18080
无监听、Retriever PID消失；再次逐卡检查GPU。禁止`pkill`、`killall`、全局`ray stop`或清理
旧Run/tmux证据。

## 汇总与判定

生成单独完成报告，至少包含：

- 三模型总体及分源EM/success、answer compliance、搜索次数、invalid query/action、检索错误；
- 每个模型的二项比例Wilson区间；
- 同一32题上的Base↔Step 2、Base↔Step 5、Step 2↔Step 5逐题配对变化，并使用配对检验
  （exact McNemar或配对bootstrap），不能把三组当独立样本；
- 所有Run/数据/Adapter/results/episodes SHA256和资源清理证据；
- HF greedy与训练vLLM rollout的backend差异；
- 失败案例按答案格式、无效动作、检索失败、检索到但答错、未搜索分类。

判定路线：

```text
Step 5有一致正向信号
  -> 用veRL/vLLM原生val-only复核
  -> 更大held-out + 多seed/baseline
  -> 再决定20步或多卡训练

Step 5无提升或退化
  -> 排查reward、prompt、rollout.n/group size、学习率与动作格式
  -> 形成最小修正实验
  -> 不直接追加20步
```

heldout-32只是第一轮小样本证据。即使Step 5胜出，也不能据此声称完整Search-R1复现、收敛或
泛化；必须经过原生backend复核、更大评测集和多seed。

## 时间与资源预估

- Retriever冷启动预计仍是主要等待项；
- 六个HF纯评测Run预计合计约15–30分钟，首次真实执行后以实际耗时修正；
- 只使用物理GPU1，显存预计显著低于训练；无Checkpoint增长，结果文件体积很小；
- 若单Run超过预估但日志仍前进，不重复启动；通过stdout日志和精确tmux查看状态。

## 本阶段完成定义

只有六个Run及退出验收全部完成、对比报告和原始hash归档、审计声明边界明确后，才能把
“heldout-32初步对比完成”标记为完成。任何一个Run失败都保留证据并单独记录，不用新结果覆盖。
