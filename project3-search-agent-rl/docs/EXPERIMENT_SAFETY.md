# 实验安全与进程生命周期

## 硬件边界

- 物理GPU 0永久保留给Linux图形界面，所有项目三脚本硬拒绝GPU 0；
- 物理GPU 5有历史不稳定记录，默认拒绝，仅允许有人值守时显式覆盖；
- 默认稳定集合为物理GPU `1,2,3,4,6,7`；
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`与`CUDA_VISIBLE_DEVICES`必须在Python、Ray和vLLM启动前设置；
- 启动前目标卡存在任何Compute Process时拒绝运行，不抢占、不结束未知进程。

## 存储边界

模型、数据、Retriever索引、日志和Checkpoint必须写入`PROJECT3_DATA_ROOT`。运行脚本要求该目录
已经存在、可写且默认至少保留150 GiB。标签为`data`的额外NVMe设备是`/dev/nvme0n1`，已挂载在
`/media/imc/data`；项目使用其下的`project3-search-agent-rl/`专用目录，不触碰盘上其他数据。
脚本不会自动格式化或重新挂载磁盘。

## 启动与停止

所有GPU任务通过独立会话启动：

```bash
export PROJECT3_DATA_ROOT=/media/imc/data
bash scripts/preflight.sh 1
bash scripts/run_managed.sh s0-inference-001 1 -- <command> <args...>
```

每个Run写入独立目录并保存命令、物理GPU、会话ID、标准输出、错误输出和退出状态。已有Run目录
不会被覆盖。同一项目的GPU锁可避免两个受管实验误用同一张卡。

正常终止优先在前台按`Ctrl-C`。需要从另一个终端停止时：

```bash
export PROJECT3_DATA_ROOT=/media/imc/data
bash scripts/stop_managed.sh s0-inference-001
```

停止脚本只向该Run记录的进程组发送`TERM`；等待30秒后仍未退出时，才向同一进程组发送
`KILL`。禁止使用`pkill python`、`killall`、按进程名批量结束或无范围的`ray stop --force`。

脚本不会自动结束进程组之外的显卡进程。退出后如果显存仍被占用，它只报告PID并暂停下一轮，
由人工核验所有者、命令行和父子关系后处理。这一限制用于避免误杀桌面服务或其他用户任务。

## 实验晋级

按`Inference → 1 Step → 5 Step → 20 Step`逐级运行。每一级都要确认进程完全退出、显存恢复、
无OOM/NaN、Retriever错误可区分、Checkpoint可恢复，再进入下一级。100–300 Step实验必须在
20 Step吞吐与资源外推完成后单独批准。
