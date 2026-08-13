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
