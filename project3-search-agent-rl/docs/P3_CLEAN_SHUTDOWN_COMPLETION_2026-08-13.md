# P3 Actor干净退出门禁完成报告（2026-08-13）

## 结论

Attempt G已通过**Actor/训练Worker级**干净退出门禁：veRL内部RegisterCenter、GPU融合Worker和
CPU TaskRunner均按确定顺序主动退出并被GCS观察为`DEAD`；训练与Ray日志不存在
`SYSTEM_ERROR`、`RAY_WORKER_FAILURE`、unexpected worker或Segmentation fault。

整体实验完整性仍为`WARN`：Raylet、GCS和Dashboard的正常集群关闭使用预期SIGTERM；而且本实验
仅为8题、seed 0、单次Global Step 1→2恢复更新，无held-out validation。不能宣称完整Search-R1
复现、质量提升、收敛或泛化。

## Attempt G

- Run ID：`p3-grpo-shutdown-gate-qwen15b-s0-20260813g`
- 时间：2026-08-13 19:57:42至20:00:16
- tmux：`p3-p3-grpo-shutdown-gate-qwen15b-s0-20260813g`
- 顶层退出码：0
- 物理GPU：仅GPU1；GPU0保持GNOME桌面基线约387MiB
- 数据：8条Search-R1 smoke训练问题，16条trajectory，21条action
- 恢复：Global Step 1→2
- Retriever：真实Wiki-18 IndexFlatIP，21,015,324向量和语料行

## 训练指标

```text
training/global_step       2
actor/grad_norm            0.275
actor/entropy_loss         1.273
actor/kl_loss              0.001
critic/score mean/max/min  0.062 / 1.000 / -0.100
episode/reward mean        0.125
valid_action_ratio         0.667
step time                  87.623 s
throughput                 96.596 token/s
```

这些指标证明一次真实参数更新链路，不证明质量提升。

## Rollout、Mask和Retriever证据

- `rollouts/2.jsonl`：21条，38,549字节；
- `rollouts/2.audit.jsonl`：21条，522,081字节；
- 两者均无`.partial`，通过独占创建、fsync和原子rename落盘；
- Prompt policy-loss token总数：0；
- Policy-loss token总数：2,629；
- Retriever状态：3次`success`、2次`invalid_query`；
- 3次成功检索共9个互异Wiki文档ID；
- 3次请求均在attempt 1成功。

## Checkpoint证据

`checkpoints/global_step_2`包含：

- 模型状态：7,256,716,618字节；
- Optimizer状态：295,770,858字节；
- LoRA Adapter：147,770,464字节；
- Extra State、DataLoader State、Tokenizer和配置文件；
- `latest_checkpointed_iteration.txt=2`。

## 退出链与根因演进

### Attempt D

没有显式关闭Worker，Driver结束时Ray用SIGTERM回收GPU Worker，产生`SYSTEM_ERROR`。

### Attempt E

GPU Worker改为`INTENDED_USER_EXIT`，但CPU TaskRunner仍被Driver的`ray.shutdown()`回收。

### Attempt F

GPU Worker和TaskRunner均主动退出；进一步全量扫描发现Worker拥有的
`WorkerGroupRegisterCenter`因引用计数归零被Ray回收，core日志仍含SIGTERM/SYSTEM_ERROR。

### Attempt G

最终退出顺序：

```text
WorkerGroupRegisterCenter
  → GPU WorkerDict（Actor/Reference/Rollout共置）
  → CPU TaskRunner
  → Driver ray.shutdown()
  → Raylet/GCS/Dashboard EXPECTED_TERMINATION
```

日志明确记录：

```text
Gracefully stopped 1 WorkerGroup register center
Gracefully stopped 1 physical Ray worker actor
Gracefully stopped TaskRunner actor
```

三类Actor均为`INTENDED_USER_EXIT`、exit code 0并进入DEAD。全量禁止关键字扫描结果：

```text
SYSTEM_ERROR       0
RAY_WORKER_FAILURE 0
unexpected worker  0
Segmentation fault 0
Actor SIGTERM      0
```

Ray基础设施仍有正常关闭SIGTERM，日志带`EXPECTED_TERMINATION`。它不属于训练Actor失败，后续门禁
必须分开统计，不能再笼统写“所有Ray日志无SIGTERM”。

## 资源清理

- Attempt G结束后GPU1回到18MiB；
- `nvidia-smi`计算进程只剩GPU0的GNOME远程桌面，未出现项目Python；
- Attempt G训练、TaskRunner、Worker和Ray PID均消失；
- 本轮CPU Retriever通过只向`project3-p3e-retriever`发送Ctrl-C停止；
- Retriever pane退出0，PID 480665/480667消失，127.0.0.1:18080无监听；
- 未使用`pkill`、`ray stop`、`tmux kill-server`或GPU0。

## SHA256

```text
metadata.env       9dafe99776b48255b292b5cb02456b4b28c7d3c1566f4b449efe55c89d20c7aa
Hydra config       c19e26c1b62ae9fe4ca1a1070b76e8c2c7006130715aceec10d19801cbe8ed9c
2.jsonl            2026e57d7f08c7d8ade0e901babde47325224bdb9dd8db47919451b0154cf740
2.audit.jsonl      520356299a81f2683d07ba6c1355c8bdde8dec6bee8fd37882092fefd2cb65e8
LoRA adapter       6cf057bcf3f76ef89c947e5d3e3049acaabe4d7e3fadcccba0ee316c3866453d
stdout.log         954ebf4188140c98020fabb4b4306b1aabe611c270c1184d20e252d3ae8dc9e9
stderr.log         25e5a92a99767a800e786fe07c85cfbb41ab420f88cb3f05f44fe9a9d790aca5
patch 0003        1d18a57a7ab0108a063d7972db98c4ba8d5db8b97163865bb5442bf48fd33560
```

## 审计与下一步

独立`experiment-audit`结论为WARN（same-family provisional）：训练与Actor退出声明有证据，范围声明
必须受限。下一阶段可以讨论5步晋级，但必须保持：新Run ID、GPU1、真实Wiki-18、原子证据、
Checkpoint可恢复、Actor级退出门禁；在held-out evaluation和baseline前不得宣称质量提升。
