# 环境配置状态

记录日期：2026-08-06

> 2026-08-10 增量提示：本文主体是初始环境快照。项目二后来新增
> `.venvs/phase1-openrlhf`，并下载 Qwen2.5-Coder-7B-Instruct 至数据盘；fused CE 与
> 单卡真实一步 SFT smoke 已通过；GPU5 保持故障，但已用容器 UUID 白名单隔离，物理
> GPU2/4/6/7 的 NCCL 与四卡 ZeRO-3 正确性 smoke 也已通过。正式 238 条 SFT 尚未完成，
> 当前训练和 GPU 状态见 `docs/PROJECT_STATUS_2026-08-10.md`。不要依据本页早期描述判断
> Phase 1 训练是否已经完成。

## 主机状态

- OS：Ubuntu 22.04.4 LTS
- GPU：8 × NVIDIA GeForce RTX 4090 D（24GB）
- GPU 0：明确禁用
- CUDA Toolkit：12.4
- NVIDIA Driver：595.45.04
- tmux：已安装
- Docker CLI/daemon：29.3.0，当前用户已加入 `docker` 组
- 初始可用磁盘：约 146GB；完成可行性验证后约 137GB 可用（根分区使用率 92%）
- 数据盘：`/dev/nvme0n1`，挂载于 `/media/imc/data`，容量约 3.6TB、可用约 3.4TB
- Docker 根目录：`/var/lib/docker`，仍位于系统盘；批量拉取 SWE-bench 官方镜像前需要另行决定是否迁移，不能直接扩到几十个镜像。

## 存储策略

- 项目代码、配置、脚本和报告保留在 `/home/imc/yzy/agent`。
- 用户已于 2026-08-06 授权模型、数据集、训练 checkpoint 和大体积缓存使用 `/media/imc/data`。
- 后续统一使用 `/media/imc/data/yzy/agent/` 作为本项目的大文件命名空间；实际创建子目录时再按 `models/`、`datasets/`、`checkpoints/`、`caches/` 分开。
- 无论位于系统盘还是数据盘，任何删除、清理和破坏性覆盖操作都必须先停止并询问用户。

## 源码版本

| 项目 | 仓库 | Commit |
|---|---|---|
| 项目一 | microsoft/agent-lightning | `f0a77cfad71e6222a3edb7dfc7a0f611bd231364` |
| 项目一 | microsoft/AgentRx | `f228165bfec60a801fd5fedd9d8ffe0f9de0c69d` |
| 项目一 | sierra-research/tau2-bench | `a51cedb30a292896e7aff9a67c9f6dcbc4a29cd3` |
| 项目二 | SWE-agent/SWE-agent | `3ea751c087f32b16e039a2233dd6eefecef325d5` |
| 项目二 | SWE-bench/SWE-smith | `9b74ac08118a85c39c356802f7961893af73e07f` |
| 项目二 | rllm-org/rllm | `1d1109a655e291b3001d8526d7c9ecc5b9328226` |

所有仓库均使用 `--depth 1` 浅克隆。

## Python 与虚拟环境

uv 0.12.2、CPython 3.11.15 和 CPython 3.12.13 均安装在工作区内：

```text
.tools/bin/uv
.tools/uv-python/
.cache/uv/
```

| 环境 | Python | 用途 | 状态 |
|---|---:|---|---|
| `project1-harness-evolution/.venvs/agent-lightning` | 3.11 | Agent Lightning | 已安装，CPU Smoke Test 通过 |
| `project1-harness-evolution/.venvs/agentrx` | 3.11 | AgentRx | 已安装，CPU Smoke Test 通过 |
| `project1-harness-evolution/.venvs/tau2` | 3.12 | tau2-bench | 已安装，CPU Smoke Test 通过 |
| `project2-coding-agent-rl/.venvs/swe-tools` | 3.11 | SWE-agent + SWE-smith generate | 已安装，CPU Smoke Test 通过 |
| `project2-coding-agent-rl/.venvs/rllm-base` | 3.11 | rLLM 基础依赖 | 已安装，CPU Smoke Test 通过 |
| `project2-coding-agent-rl/.venvs/phase1-openrlhf` | 3.11 | Phase 1 7B OpenRLHF SFT | 已安装；fused 分支和单卡一步 smoke 通过；正式 ZeRO-3 SFT 未完成 |

Agent Lightning 与 AgentRx 不能放进同一个环境：前者的 AgentOps 依赖 `packaging<25`，后者依赖 `packaging>=25`。Agent Lightning 还需使用其 `uv.lock` 对应的 `fastapi==0.121.2`，否则新版 FastAPI 与当前 LiteLLM 不兼容。

## GPU Smoke Test

- 启动方式：tmux session `agent-p2-gpu-smoke-20260806`
- 物理 GPU：1（GPU 0 未使用）
- 启动前状态：18MiB used、24066MiB free、0% utilization
- 测试内容：PyTorch CUDA 初始化、1024×1024 矩阵乘法及同步
- 结果：`CUDA_SMOKE_OK`
- 结束后状态：18MiB used、24066MiB free、0% utilization
- 日志：`project2-coding-agent-rl/runs/smoke/gpu-1.log`

安装时首次创建的 `project1-harness-evolution/.venvs/agent-tools` 是依赖解析失败后留下的未使用环境。根据“删除前先确认”的约束，目前保留，不做清理。

## Docker Smoke Test

- `docker info`：通过，storage driver 为 `overlayfs`
- `docker run hello-world`：通过
- 项目二 `swe-tools` 环境的 Python Docker SDK：`client.ping() == True`
- SWE-agent 官方安装检查 `sweagent --help`：通过
- 测试镜像：`hello-world:latest`
- 测试容器：`agent-setup-hello-20260806`，状态为正常退出

根据“删除前先确认”的约束，测试镜像和已退出容器均保留。

项目二还基于本机已有镜像构建了 `agent/swe-rex-py311:20260806`，加入 Python 3.11 和固定版本 `swe-rex==1.4.0`，用于绕过公共 Python 镜像拉取超时。镜像约 7.55GB（与基础镜像共享层）。

## DeepSeek 与 τ²-bench Smoke Test

- API Base：`https://api.deepseek.com`（OpenAI 兼容）
- 模型：`deepseek-v4-flash`
- `/models` 验证：通过
- Chat Completions 验证：通过，返回 `API_OK`
- τ²-bench domain：`mock`
- 任务：`create_task_1`，1 trial，seed 300
- 结果：Reward 1.0、DB Match 1.0、Write Action 1/1
- 时长：约 8 秒
- 结果目录：`project1-harness-evolution/vendor/tau2-bench/data/simulations/p1-smoke-deepseek-v4-flash-20260806/`

项目级适配器已注册官方价格并修正 NL assertion judge 的模型路由；早期 smoke 结果中的成本为 0，后续结果已能记录成本。

## 基本可行性结论

结论：两个项目的核心技术链路均已跑通，可以进入小规模正式实验设计，但目前不代表训练完成或论文结论成立。完整证据和边界见 [`FEASIBILITY_REPORT.md`](FEASIBILITY_REPORT.md)。

- 项目一：DeepSeek 在真实 `tau2-bench` retail 任务上连续完成 3 个任务，均为 Reward 1.0；其中一条轨迹已转换并通过 AgentRx IR 阶段，得到 25 个有效步骤。
- 项目二：SWE-agent 通过 DeepSeek 的函数调用自主检查仓库、复现错误、生成一行修复并提交补丁；轨迹含 12 次模型调用。候选补丁随后在断网 CPU 容器中独立应用，3/3 单元测试通过，退出码 0。
- 本轮 API/CPU/Docker 验证未使用 GPU。此前唯一 CUDA smoke 遵守约束，在 tmux 中使用物理 GPU 1，未使用 GPU 0。
- 独立复验容器 `agent-p2-patch-eval-20260806` 和相关镜像均保留，未执行清理。
- 真实仓库探针期间中止的 `agentswe-rex-py31120260806-5121956a-a049-4c74-9994-c8970dce0cb6` 容器已在获得用户许可后停止，并由 Docker `--rm` 自动删除；其他评测容器、镜像和结果未清理。

## 下一阶段的大文件门禁

以下大文件优先放到已授权的数据盘命名空间；执行前仍检查容量和目标路径：

- 下载 Hugging Face 模型或数据集；
- 安装 `rllm[verl]`、vLLM、FlashAttention 等完整训练栈；
- 拉取 SWE-smith 环境镜像；
- 构建批量 Repository Docker 镜像。
