# P3 Ray干净退出与证据文件保护修复（2026-08-13）

## 1. 本阶段目标

修复Checkpoint恢复Step 2之后仍存在的两个工程问题：

1. veRL训练正常完成后，Ray把GPU Worker的SIGTERM记录为`SYSTEM_ERROR`；
2. veRL原始rollout文件使用写模式打开，同一个Run/Step路径被复用时会静默覆盖实验记录。

本文件记录代码修复与CPU/Ray预验证。物理GPU1复验的结果将在运行结束后追加，当前不提前声明
“GPU训练已干净退出”。

## 2. 原因定位

### 2.1 Ray Worker被动终止

原始调用链在`trainer.fit()`返回后没有显式关闭Ray WorkerGroup。Driver随后退出，Ray只能用
SIGTERM回收仍存活的融合Actor；因此即使训练指标和Checkpoint均已写完，Ray事件仍会显示
`Worker exit type: SYSTEM_ERROR`和unexpected worker failure。

这不是GPU0误用。本轮此前的训练被资源门禁固定在物理GPU1，问题发生在正常训练结束后的
生命周期收尾阶段。

### 2.2 rollout证据可被覆盖

veRL的`_dump_generations()`直接用`open(filename, "w")`写普通rollout JSONL。若Run ID和
Global Step被复用，旧文件会被截断。结构化audit文件此前已禁止覆盖，但普通rollout尚未得到
同等保护。

## 3. 实现内容

### 3.1 显式、去重的Ray Actor退出

- 为veRL `Worker`新增`graceful_shutdown()`；先销毁已初始化的Torch distributed process
  group，再调用`ray.actor.exit_actor()`，使Ray将其识别为用户主动退出；
- `RayPPOTrainer.shutdown_workers()`汇总Actor、Critic、Reference和Reward Model的WorkerGroup；
- 按Ray Actor ID去重，避免角色共置时对同一物理Actor重复发退出请求；
- 等待全部退出任务引用完成，并通过Ray GCS actor table确认每个Actor最终进入`DEAD`；
- `TaskRunner.run()`用嵌套`finally`关闭训练/验证环境并关闭Worker；
- 仅当本函数自行初始化Ray时，Driver外层才在`finally`执行`ray.shutdown()`。

代码没有使用全局`pkill`或`ray stop`，不会终止其他用户/其他实验的进程。

### 3.2 JSONL原子写入和禁止覆盖

- 普通rollout与结构化audit统一经过`dump_jsonl_records()`；
- 目标存在时抛出`FileExistsError`，拒绝静默覆盖；
- 先以独占模式写`*.partial`，执行flush和`fsync`，再原子重命名；
- 遗留partial存在时也停止，要求人工核验，不把半文件当成完整证据。

### 3.3 可重现补丁

新增`patches/0003-graceful-ray-shutdown-and-atomic-rollout.patch`，并加入：

- `scripts/apply_project_patches.sh`的顺序补丁列表；
- `scripts/run_p3_grpo_one_step.sh`的必需补丁门禁。

veRL仍固定在`20bd331bdbc9026a5668e11362178e10ab7400c8`，不直接提交vendor脏状态。

## 4. 已完成验证

### 4.1 单元测试

```text
python -m unittest discover -s tests -p 'test_p3_training_*.py' -v
Ran 7 tests in 0.006s
OK
```

覆盖Actor去重、超时Actor定位、空WorkerGroup、退出确认但未达DEAD、Prompt Loss Mask、结构化
audit原子写和普通JSONL禁止覆盖。

### 4.2 真实Ray CPU Actor探针

使用项目实际Ray 2.43环境启动CPU-only Actor，故意把同一Actor放入两个WorkerGroup：

```text
{'shutdown_actor_ids': ['bf9a096aa854c76653c2e14701000000']}
```

探针确认只发出一次退出请求，并通过GCS确认`DEAD`后返回。调试中还发现：

- `ray.experimental.state.api.get_actor()`依赖Dashboard HTTP，在本机返回502，不适合作为收尾门禁；
- Ray 2.43的GCS actor table状态字段为`State`而非`state`。

最终实现使用不依赖Dashboard的GCS actor table，并兼容两种字段名。

### 4.3 补丁重放

在`/tmp/p3patch-shutdown.pqx5Ar`从固定veRL HEAD导出干净树，依次应用0001、0002、0003，
三个修改目标与工作树逐字节diff一致；`git diff --check`和vendor diff检查均通过。

## 5. GPU复验门禁

后续使用新的Run ID从Global Step 1恢复到Step 2，仅使用物理GPU1。通过条件为：

1. 顶层退出码为0，Global Step 2 Checkpoint完整；
2. 日志出现`Gracefully stopped 1 physical Ray worker actor`；
3. Ray日志中该Worker为`INTENDED_USER_EXIT`，且无`SYSTEM_ERROR`、unexpected worker failure、
   SIGTERM或Segmentation fault；
4. 普通`2.jsonl`和`2.audit.jsonl`均存在且不是partial；
5. 运行结束后GPU1无本项目计算进程、Retriever端口和本Run Ray进程均释放；
6. GPU0在整个实验中不进入可见设备列表。

任何一项失败都保留为WARN并记录原始日志，不删除、改写或美化失败证据。

## 6. Attempt E物理GPU1复验结果：WARN

- Run ID：`p3-grpo-shutdown-gate-qwen15b-s0-20260813e`
- 时间：2026-08-13 19:43:42至19:46:27
- tmux顶层退出码：0
- 设备：仅物理GPU1；GPU0保持GNOME桌面基线约387MiB
- 恢复与训练：Global Step 1恢复到Step 2，`grad_norm=0.288`，单步88.716秒
- 检索：3次真实请求均首次成功，各返回Top-3
- 证据：`2.jsonl` 38,549字节，`2.audit.jsonl` 522,081字节，无partial
- Checkpoint：模型约7.26GB、Optimizer约296MB、LoRA Adapter约148MB及Extra/Data State均存在
- 清理：tmux退出0，GPU1回到18MiB，训练/Ray进程均释放

GPU融合Actor `fa957ef6d25e3680b052d6da01000000`确实在Ray原始日志中以
`INTENDED_USER_EXIT`和exit code 0退出，并进入`DEAD`。但是CPU编排Actor `TaskRunner`
在其`run()`返回后仍然存活；Driver随即调用`ray.shutdown()`，Ray用SIGTERM回收TaskRunner，
仍产生一条`RAY_WORKER_FAILURE / SYSTEM_ERROR`。因此Attempt E不能通过整体干净退出门禁。

关键区别是：这次`SYSTEM_ERROR`不再来自占用GPU显存的Worker，而来自CPU TaskRunner生命周期。
它不影响已完成的Step 2和Checkpoint，但仍必须如实保留为退出WARN。

## 7. Attempt F前的第二层修复

在Driver中增加显式TaskRunner退出：

1. `ray.get(runner.run.remote(config))`完成；
2. 调用`runner.graceful_shutdown.remote()`；
3. 等待引用完成并通过GCS确认TaskRunner为`DEAD`；
4. 最后才调用`ray.shutdown()`关闭Driver和本地Ray集群。

`TaskRunner.graceful_shutdown()`同样使用`ray.actor.exit_actor()`。更新后的0003补丁再次通过7项
单元测试、py_compile、逆向检查和`/tmp/p3patch-taskrunner.47Fkmc`干净树顺序重放。

当前状态更新为“GPU Worker退出已通过，TaskRunner退出修复完成但等待Attempt F物理GPU1复验”。
