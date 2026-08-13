# P3-B执行日志

## Attempt A：Ray Socket路径门禁失败

- Run ID：`p3-grpo-1step-qwen15b-s0-20260813a`
- 时间：2026-08-13 17:30（Asia/Shanghai）
- 退出码：1
- 运行到：Retriever health通过，`ray.init()`创建Plasma Socket之前
- 模型加载：未发生
- GPU显存：未占用
- Checkpoint：未产生

失败原因：

```text
OSError: AF_UNIX path length cannot exceed 107 bytes
```

旧`run_managed.sh`把`RAY_TMPDIR`设为完整Run目录下的`ray/`。Ray继续追加带时间戳的session
目录和`sockets/plasma_store`，描述性Run ID使最终路径超过Unix Socket上限。

安全验收：

- 失败Run目录保留，未覆盖或删除；
- `cleanup.log`报告物理GPU1无计算进程；
- 未发现`raylet/plasma_store/gcs_server`残留；
- Run目录仅104K；
- CPU Retriever按计划继续运行，等待修正后的新Run。

修复：

1. `run_managed.sh`使用`mktemp -d /tmp/p3r.XXXXXX`创建短且唯一的Ray live目录；
2. live目录路径写入`metadata.env`；
3. 受管进程结束后用`mv`归档到该Run的`ray/`目录；
4. 不删除失败证据，不使用全局`ray stop`或模糊`pkill`；
5. 修复后使用新Run ID后缀`b`，不重用Attempt A。

该修复不改变veRL算法、模型、数据、GPU映射或训练超参数，只修正进程运行目录长度。

## Attempt B：vLLM KV Cache门禁失败

- Run ID：`p3-grpo-1step-qwen15b-s0-20260813b`
- 时间：2026-08-13 17:34（Asia/Shanghai）
- 退出码：1
- 已通过：Ray初始化、数据8/16行、配置校验、Qwen模型加载、LoRA注入、FSDP和NCCL初始化
- 失败点：vLLM初始化KV Cache，尚未生成Rollout或执行Backward

错误：

```text
ValueError: No available memory for the cache blocks.
Try increasing `gpu_memory_utilization` when initializing the engine.
```

安全验收：GPU1恢复18MiB、无计算进程；Ray live目录已归档到Run目录，大小约1.1MiB；
未产生Checkpoint，Retriever继续健康运行。

原因与修正：

1. 规划值0.35低于Actor模型驻留后的vLLM最低KV Cache预算；
2. 固定上游的Search及其他训练示例统一使用0.60，因此Attempt C改为0.60；
3. 当前veRL文档将vLLM V1描述为显式opt-in，但环境默认进入了V1路径；为贴近该提交的
   Hybrid Engine基线，Attempt C显式设置`VLLM_USE_V1=0`；
4. Token上限2304、Eager模式、Cache Engine释放、Actor/Optimizer/Reference Offload均保持；
5. 新Run使用后缀`c`，不复用Attempt B。

## Attempt C：单步更新完成，退出段有警告

- Run ID：`p3-grpo-1step-qwen15b-s0-20260813c`
- 时间：2026-08-13 17:37:48至17:40:28（160秒）
- 顶层退出码：0
- 训练：1个Global Step、1次Actor Update、Checkpoint保存完成

关键指标：

```text
grad_norm                    0.300
pg_loss                     -0.001
ppo_kl                       0.001
entropy                      1.256
reward mean/max/min          0.125 / 1.0 / 0.0
advantage max/min            1.155 / -1.155
valid_action_ratio           0.667
tool_call_count mean         0.312
step time                    100.164 s
peak allocated/reserved GPU  20.835 / 23.014 GiB
throughput                   85.370 token/s
```

真实性证据：

- 21条Action记录来自8个问题、Group 2的16条轨迹；
- Score分布为2个`1`、12个`0`、7个`-0.1`；
- LoRA Adapter有392个张量，LoRA-B的20,643,840个元素全部非零；
- Scheduler为`last_epoch=1`、`_step_count=2`、最后LR为`3e-6`；
- 保存模型状态7.26GB、Optimizer 295MB、LoRA Adapter 148MB和Extra/Data State。

主要SHA256：

```text
model state  6a5454c8464fa09917ca8fe20a5c9156391303f28c07289883880b4ebcc340fe
optimizer    a32b50a1269650786996183a6af9d8cea3f838b8e993ce0f25d5c2e184c56223
lora adapter d84d48d73223e2235646e118cce30427989e2b56bc079cf3834d7330230c2186
rollouts     42a24abc5b29fbd729b6c6301c30da1a2905abd515445b9773d481ba77edaa5e
```

退出问题：Checkpoint与指标完成后，Ray Worker收到SIGTERM并在关闭阶段Segmentation fault。
训练顶层仍正常返回0，GPU1、Ray进程和Retriever均已清理。该问题不否定已保存更新，但不能
表述为干净退出，恢复实验前必须定位或规避。

独立实验完整性审计为`WARN`，见`EXPERIMENT_AUDIT.md`：支持“真实单步更新完成”，不支持
“完整复现”或“质量提升”；Checkpoint恢复和Token级Mask证据仍待执行。

## Attempt D：Checkpoint恢复到Global Step 2完成，退出仍WARN

- Run ID：`p3-grpo-resume-step2-qwen15b-s0-20260813d`
- 时间：2026-08-13 18:38:59至18:41:37
- 顶层退出码：0
- 恢复：明确加载Step 1模型、Optimizer、Extra State和DataLoader State
- 更新：`training/global_step=2`、`grad_norm=0.283`，保存Global Step 2
- 连续性：Scheduler推进1→2，392/392个Adapter张量继续变化
- Mask：21/21记录Prompt Loss Token为0，Policy Token总数2629
- Retriever：3次成功Top-3共9个真实Wiki ID，2次空查询被标记为`invalid_query`
- 退出：TaskEventBuffer Segfault消失，但Ray仍把SIGTERM关闭Worker记为`SYSTEM_ERROR`
- 清理：GPU1、Ray、训练Python、Retriever端口和服务均已释放

详细结果见`docs/P3_CHECKPOINT_RESUME_COMPLETION_2026-08-13.md`。本阶段证明工程恢复链路，
不证明完整复现或质量提升。
