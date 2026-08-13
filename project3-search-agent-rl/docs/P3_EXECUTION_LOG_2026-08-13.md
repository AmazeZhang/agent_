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
