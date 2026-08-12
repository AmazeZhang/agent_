# Search-R1复现环境基线

## 宿主机

- Ubuntu 22.04.4 LTS，Linux 6.5，glibc 2.35；
- 8×NVIDIA GeForce RTX 4090 D，Compute Capability 8.9，每张24564 MiB；
- NVIDIA Driver 595.45.04；`nvidia-smi`显示CUDA 13.2，含义是驱动最高兼容能力；
- 本地Toolkit：CUDA 12.4.131为默认，同时存在CUDA 12.6；
- GPU 0运行Xorg与GNOME Remote Desktop，永久不用于计算；GPU 5默认不使用；
- GPU 0–3位于NUMA 0，GPU 4–7位于NUMA 1；同组为PIX，跨组为SYS，无NVLink；
- 96逻辑CPU、约1 TiB RAM；`data`盘挂载于`/media/imc/data`，剩余约3.2 TiB；
- tmux 3.2a。

## 首轮复现栈

| 组件 | 固定版本 | 依据 |
|---|---:|---|
| Python | 3.10 | 上游Search Retriever说明及cu124 FlashAttention wheel |
| PyTorch | 2.6.0+cu124 | Search-R1加入verl-agent时的官方安装说明 |
| vLLM | 0.8.5.post1 | 上游长期使用的cu124 CI镜像/安装脚本 |
| FlashAttention | 2.7.4.post1 | 上游cu124镜像固定版本 |
| Transformers | 4.51.1 | 上游requirements固定版本 |
| Ray | 2.43.0 | vLLM 0.8.5要求的最低版本；default extra不强制升级OpenTelemetry |
| TensorDict | 0.8.3 | 当前固定verl-agent提交声明`>=0.8.0`；仍属于Torch 2.6同期版本 |

完整直接依赖见`configs/requirements-searchr1-repro.txt`，实际安装后以自动生成的
`requirements-searchr1-repro.lock.txt`为第三方包证据。editable的veRL与SkyRL Gym不写入该
文件，其源码由Git submodule SHA `20bd331bdbc9026a5668e11362178e10ab7400c8`锁定，并由
`create_repro_env.sh`显式安装。

Ray 2.48虽然位于verl声明的范围内，但其`default` extra要求OpenTelemetry不低于1.30，和
vLLM 0.8.5要求的1.26.x冲突；Ray 2.41又低于vLLM 0.8.5要求的2.43。wheel元数据审计确认
Ray 2.43的`default` extra不强制升级OpenTelemetry，因此固定为兼容交集2.43.0。不能只依据
verl的宽泛版本区间选择最新版。

vLLM还对OpenAI、OpenCV和CuPy只给出了较宽的下界。为避免2026年的新API代际被解析进
2025年的Search-R1栈，额外固定OpenAI 1.76.0、OpenCV Headless 4.11.0.86和
CuPy CUDA12x 13.3.0。最终`pip check`结果为`No broken requirements found`。

## 为什么首轮不使用vLLM 0.11

当前verl-agent README的通用安装段后来升级到了vLLM 0.11，而Search-R1加入仓库时使用的是
Torch 2.6/cu124与vLLM 0.8.5。vLLM 0.11通常配套Torch 2.8和更新的CUDA wheel，会同时改变
Torch、CUDA运行时、TensorDict和内核行为。首轮复现选择上游实际验证过的Search-R1代际，减少
变量；升级vLLM时创建新的版本化环境，不原地升级基线环境。

## 已有环境处理

`paretotool-searchr1`包含Python 3.9、Torch 2.4/cu121、vLLM 0.6.3、Ray 2.51.2和TensorDict
0.5；它不满足当前verl-agent的Ray上限，不能作为项目三复现环境。`paretotool-retriever`为CPU
Retriever历史环境。二者只读保留，不修改、不卸载、不混用。

## 环境与缓存路径

```text
/media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124
/media/imc/data/project3-search-agent-rl/cache/uv
/media/imc/data/project3-search-agent-rl/cache/huggingface
/media/imc/data/project3-search-agent-rl/cache/torch
```

## 2026-08-12验证结果

- CPU隔离导入：Torch、Transformers、Ray、TensorDict、FlashAttention、vLLM、veRL、
  Search环境、Reward和TrajectoryCollector均通过；
- veRL训练入口：`python -m verl.trainer.main_ppo`模块导入通过；
- GPU最小测试：受管脚本仅映射物理GPU1，Torch FP16 MatMul和FlashAttention小张量内核通过；
- GPU测试记录：`/media/imc/data/project3-search-agent-rl/runs/env-smoke-gpu1-20260812`；
- 测试退出后GPU1无计算进程，GPU0的GNOME Remote Desktop保持不变；
- 这只证明软件栈与基础CUDA内核可用，不等于模型推理或Search-R1已经复现。

依赖输入SHA256为`dd03ff2a705b4b0446dce48358e83fc8846a26a2979855b2456dd7f76d36c530`，
第三方安装锁SHA256为`758ef9afec5a0796ecec4a9c072bcd72f2c973719ab756206959fe35bd177a65`。
每次有意修改依赖后必须重新生成并同步到阶段报告。

## 多卡拓扑约束

- 单卡开发优先物理GPU1；
- 2–3卡开发优先从`1,2,3`选取，避免跨NUMA；
- 4卡首选`1,2,3,4`会跨NUMA，其中物理GPU4位于NUMA 1，必须先做NCCL预检；
- 6卡稳定集合为`1,2,3,4,6,7`，跨NUMA且无NVLink，不能按NVLink服务器估算吞吐；
- 不先验设置`NCCL_P2P_DISABLE`等绕过变量，只有NCCL测试给出证据后才调整并记录。

## tmux运行约定

训练使用`start_tmux_run.sh`，底层仍经过GPU门禁与独立进程组：

```bash
export PROJECT3_DATA_ROOT=/media/imc/data
bash scripts/start_tmux_run.sh <run-id> 1 -- <training-command>
```

用户查看命令：

```bash
tmux list-sessions
tmux attach -t p3-<run-id>
# 从tmux退出但不中止：Ctrl-b，然后按d
tmux capture-pane -pt p3-<run-id>:0 -S -100
```

tmux窗口退出后保持可查看；确认日志已经落盘后，可以手工执行
`tmux kill-session -t p3-<run-id>`清理窗口。该命令只清理已结束的tmux会话，不用于停止训练。
