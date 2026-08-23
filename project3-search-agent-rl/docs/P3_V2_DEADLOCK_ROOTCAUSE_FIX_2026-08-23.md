# P3 Search-aware clean v2：NCCL 死锁根因定位与修复（2026-08-23）

## 0. 结论先行

**20260823b/c 两次"确定性"崩溃 = NCCL 集体通信发散死锁，根因是我在
`dp_actor.update_policy` 中加入的 per-rank `continue` 跳过守卫。已修复为
collective-uniform 的零损失前向+反向路径，新 smoke 20260823d 11/11 PASS
（exit 0），已按授权自动进入 5 步行为训练。**

- 死锁签名：4 个 rank 卡在 `ALLREDUCE`（mini-batch 边界的 grad-norm），
  2 个 rank 卡在 `_ALLGATHER_BASE`（下一个微批的前向 embedding unshard）——
  同一条通信器上两种集体操作互相等待，NCCL 经典死锁，30 分钟 watchdog
  SIGABRT。
- 根因机制：`_balance_batch`（Karmarkar-Karp 均衡分区）把 1-token 填充行
  分到**不同** rank 的分区里 → 各 rank 跳过的微批数量不同 → FSDP
  前向/反向次数不同 → 集体序列发散。
- 夜间 run 20260822a 不受影响的原因：804 行批次恰好可被 6 整除，
  `adjust_batch` 从未填充，没有任何 rank 跳过任何微批。
- 修复：填充微批不再 `continue`，而是每个 rank 执行相同的前向+反向，
  损失用 `log_prob.masked_fill(全 True, 0.0).sum()` —— 对任意 log_prob
  内容（含 -inf）都严格等于 0.0，且保持 autograd 图连接（FSDP
  post-backward hook 均匀触发，reduce-scatter 均匀发生）。
- 验证链：CPU 套件（D1..D13，144 pass，10 个 v1 线 pre-existing 失败与本
  变更无关）、v2 历史回放门禁（all hard gates passed）、REBUILD_IDENTICAL
  （pristine 20bd331b + v2-0001..0007 == 工作区）、smoke 20260823d
  11/11 PASS、graceful shutdown、GPU 回基线。

## 1. 背景：两个 smoke 的崩溃序列

| Run | 结果 | 时间线 |
|---|---|---|
| 20260822a（夜） | **成功**（首个工程 smoke，804 行 / 可整除批次） | rollout 4 min；训练块 20:27 |
| 20260823a | 崩溃：all-zero 填充行 → unpad 空序列 → HF reshape 错误 | 已修复为 1-token 填充 |
| 20260823b | **NCCL watchdog SIGABRT**（首次挂起） | rollout 11:27→11:33:18；hang onset ~11:53:20；SIGABRT 12:23:20 |
| 20260823c | **同签名确定性复现**（排除偶发） | rollout 结束 13:06:41；hang onset ~13:26:51；SIGABRT 13:56:54 |

b 与 c 的 watchdog 证据逐字段一致：同一 seq 26282 完成、26283/26284
enqueued、同一 op 尺寸（NumelIn=51861163 / NumelOut=311166978 的嵌入
all-gather、grad-norm all-reduce），onset 都在训练块 ~20 分钟处（mini-1
边界），SIGABRT 都在 onset 后 30:00:03。确定性 → 不是传输层偶发。

## 2. 根因：per-rank 跳过 → 集体发散 → 死锁

### 2.1 证据（run c 每 rank 的卡住 op）

```
[rank3] OpType=ALLREDUCE      | last completed 26282 | enqueued 26283
[rank0] OpType=ALLREDUCE      | last completed 26282 | enqueued 26283
[rank2] OpType=ALLREDUCE      | last completed 26282 | enqueued 26283
[rank4] OpType=ALLREDUCE      | last completed 26282 | enqueued 26283
[rank1] OpType=_ALLGATHER_BASE| last completed 26282 | enqueued 26284
[rank5] OpType=_ALLGATHER_BASE| last completed 26282 | enqueued 26284
```

4 个 rank 已结束 mini-1 的微批循环，进入 `_optimizer_step` 的 grad-norm
ALLREDUCE；2 个 rank 仍在循环内，发起下一个真实微批前向的嵌入
ALLGATHER。**AllReduce 等 AllGather 的加入者、AllGather 等 AllReduce 的
加入者 → 同一通信器上两种集体互相等待 → 死锁。**

（b 的 6 个 .err 都显示 `_ALLGATHER_BASE`，是因为取样时死锁尚在演进、
全部 rank 都还在循环内；c 的取证更完整，同时抓到两侧。）

### 2.2 机制链

1. 填充行（`_build_padding_rows`）携带 1 个 attended PROMPT token、
   response 区全零 → `response_mask` 全零。
2. 我的守卫 `if response_mask.sum() == 0: continue` 跳过该微批。
3. `_balance_batch` 用 `get_seqlen_balanced_partitions`（Karmarkar-Karp，
   k=world_size=6，equal_size）把整批分成每 rank 135 行的均衡分区 ——
   1-token 填充行落在**不同** rank 的分区里。
4. 各 rank 因此跳过**不同数量**的微批 → 前向/反向次数不同 → FSDP 的
   all-gather / reduce-scatter 集体序列分叉。
5. 分叉在 mini-batch 边界第一次爆发：先结束的 rank 进 ALLREDUCE，
   未结束的 rank 还在 ALLGATHER → 死锁（见 2.1 的 op 类型分布）。
6. 挂起期间 GPU 100%、worker ~93% CPU（NCCL 内核自旋 + 未死锁 rank 的
   计算）—— 早期被误判为"计算慢"而非"挂起"。

### 2.3 为什么夜跑没死锁

20260822a 的批次恰好 804 行（audit record_index 0..803 连续、全真实
attention 228-2200 token），804 % 6 == 0 → `adjust_batch` 不填充 →
没有 rank 跳过任何微批 → 集体序列天然一致。a/b/c 的批次都不可整除
（a: 全零填充 → unpad 崩溃；b/c: 1-token 填充 → 死锁），只有填充路径
会触发本 bug。

## 3. 修复：collective-uniform 零损失前向+反向

`vendor/verl-agent-v2/verl/workers/actor/dp_actor.py`（v2-0007 补丁重生成，
REBUILD_IDENTICAL 通过）：

```python
if response_mask.sum() == 0:
    # NEVER skip with `continue` ... (per-rank collective divergence,
    # deterministic 20260823b/c hang; see doc)
    _, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=False)
    loss = log_prob.masked_fill(torch.ones_like(log_prob, dtype=torch.bool), 0.0).sum() / self.gradient_accumulation
    loss.backward()
    continue
```

- 每个 rank 对填充微批执行**相同**的 1 次前向 + 1 次反向（1-token 前向
  路径在 b/c 的 log_prob/ref 阶段已证明安全）。
- 损失通过 `masked_fill` 与模型保持图连接（FSDP post-backward hook 均匀
  触发、reduce-scatter 均匀发生），但**对任意 log_prob 内容都严格等于
  0.0**（含 -inf 位置也被全 True mask 换成 0）→ 梯度贡献恰好为零。
- 填充行不进入任何指标（continue 仍在指标收集之前）。

整个训练管线的其他阶段（compute_log_prob / compute_ref_log_prob /
compute_reward / compute_advantage）经 grep 确认没有同类 per-rank 跳过
（b/c 能通过这些阶段并在 update_policy 的 mini-1 边界死锁，也佐证了这
一点）—— 本守卫是唯一的发散源。

## 4. 验证链

### 4.1 CPU 门禁

- `tests/test_p3_v2_duplicate_fix.py` 新增 D13：
  `test_D13_padding_micro_batch_zero_loss_backward_collective_uniform` ——
  真实 `update_policy` 循环（mock 前向、计数）+ 6 行批次（4 真实 + 2 填充）：
  - 每个微批恰好 1 次前向（无跳过）：6/6
  - 填充微批的 backward 真实执行（`log_prob.grad` 非 None）且梯度全零
  - 真实微批走正常 policy-loss 路径、指标有限、`_optimizer_step` 仍执行
- D11 语义注释同步更新（"skips them" → collective-uniform 零损失路径）。
- 全套 CPU 套件：144 passed（含 D1..D13）；10 个失败均为 v1 线测试在
  v2 树上的 pre-existing 不兼容（v1 语义在 v2-0005/0006 中被有意改变），
  移除本会话任何改动后失败集完全相同，与本变更无关。
- v2 历史回放门禁：`p3_v2_reward_replay.py` —— **all hard gates passed**
  （98,059 episodes，n_sum_mismatch=0）。

### 4.2 REBUILD_IDENTICAL

`pristine 20bd331b + patches/v2/v2-0001..0007`（v2-0007 已按增量语义
重生成：对 0001..0006 应用后的状态做 diff，而非对 pristine 整体 diff）
与工作区逐文件一致，`diff -qr` 无差异。

### 4.3 新 smoke 20260823d —— 11/11 PASS

| 项 | 值 |
|---|---|
| Run ID | `p3-search-aware-clean-v2-eng-smoke-fsdp6-b66-n5-s0-20260823d` |
| GPU | 1,2,3,4,6,7（GPU0/5 禁用） |
| 结果 | exit 0；训练块 **100% 1/1 [24:26]**（b/c 死于 ~20 分钟处的同一阶段）；graceful shutdown（register centers / 6 worker actors / TaskRunner 全部停止） |
| 审计 | 800 records；`rollouts/1.audit.jsonl` + `1.jsonl`；`checkpoints/global_step_1`（6/6 model/optim/extra_state，loadable） |
| 检查器 | `check_p3_v2_smoke.py` **all 11 items PASS**，exit 0 |

11 项明细（`check_p3_v2_smoke.json`）：
- S1 real optimizer + global_step=1（6/6 optim rank 文件）
- S2 checkpoint 完整可加载
- S3 255 valid-search 轨迹
- S4 129 条 multi-search 不同 query 新 doc（诚实计数）
- **S5 v2 reward 落地**：330 episodes，`version_not_v2=0`、`schema_bad=0`、
  record/group 分量和 0 mismatch
- **S6 leak 不可重算**：470 个搜索步离线重算，79 个 v2-redundant，
  `verdict_mismatches=[]`（14 个 evidence/answer-leak 步诚实标注不可离线验证）
- **S7 轨迹 advantage 表示**：330 轨迹、66 个 uid 组全部 size 5、
  `adv_mismatches=[]`、unverifiable=0
- S8 167 条 useful-search 轨迹、104 条 positive adv
- S9 800 记录 prompt 区 policy 泄漏 = 0、response mask 不超 active
- S10 无 OOM / NaN / XID / GPU 掉卡 / retriever timeout
- S11 无残留进程、GPU 回基线（18 MiB / 0%）

## 5. 自动续跑：5 步行为实验

按 2026-08-22 九段指令 + 2026-08-23 授权（新 smoke 11/11 PASS 时自动
继续），已启动：

| 项 | 值 |
|---|---|
| Run ID | `p3-search-aware-clean-v2-behavior-fsdp6-b66-n5-s5-20260823d` |
| 门禁 | `PROJECT3_BEHAVIOR_APPROVED=yes`（exit 27 条件）、`total_training_steps=5`（exit 26 条件）、`resume_from` 空（exit 28 条件）、save_freq=1、每步 `rollouts/{n}.audit.jsonl` |
| 起点 | fresh Step0（Qwen2.5-3B-Instruct，与 clean 线同一基线），无 resume |
| 配置 | train_batch_size=66、rollout.n=5、mini_batch=330、ppo_epochs=1、lr=1e-6、kl=0.001、warmup=0.285、FSDP 三 offload、gpu_mem=0.60、max_num_seqs=64 |
| 预计时长 | ~2.5-3 小时（5 步 × ~30-40 分钟） |

完成后：Step5 merge + verify + eval（greedy official-confirm256-v1 于 GPU1、
dev64 sampling 诊断），最终报告 + PROGRESS_SYNC + 提交推送。

## 6. 声明边界

- 本修复只改了填充微批的处理方式；reward 系数、batch 大小、模型、
  Prompt、projection、GPU 拓扑、final-confirm512 均未触碰（授权范围）。
- b/c 的确定性死锁机制（分区 → 跳过 → 发散）由 2.1 的 op 类型分布 +
  2.3 的可整除对照直接支持；D13 在 CPU 上复现了"每个微批一次前向+反向"
  的不变量。
- 挂起期间 GPU 100% 的观测被解释为 NCCL 自旋 + 未死锁 rank 的计算，
  不是"慢计算"—— 新 smoke 24:26 的完整训练块与 20 分钟死锁窗口的对比
  是最终行为证据。
