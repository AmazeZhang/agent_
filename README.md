# Agent 算法双项目工作区

本目录按《Agent 算法岗位双项目实施手册》拆分为两个独立项目：

- `project1-harness-evolution/`：Trace、失败诊断、自进化与可靠性评测。
- `project2-coding-agent-rl/`：Coding Agent 数据、轨迹、SFT 与 Agentic RL。

环境和当前阻塞项见 [SETUP_STATUS.md](SETUP_STATUS.md)。

核心可行性见 [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md)，五任务小规模验证和训练门槛判断见 [PILOT_REPORT.md](PILOT_REPORT.md)。

带时间戳的当前完整进度见
[docs/PROGRESS_SUMMARY_2026-08-07.md](docs/PROGRESS_SUMMARY_2026-08-07.md)。

本轮正式开发范围与完成验收标准见
[docs/DEVELOPMENT_SCOPE.md](docs/DEVELOPMENT_SCOPE.md)。

## 克隆源码

上游开源依赖以 Git submodule 固定版本，首次克隆请使用：

```bash
git clone --recurse-submodules https://github.com/AmazeZhang/agent_.git
cd agent_
git apply --directory=project2-coding-agent-rl/vendor/SWE-agent \
  project2-coding-agent-rl/patches/sweagent-shallow-reset.patch
```

虚拟环境、密钥、模型、数据集和运行目录不会上传；它们需要按
[SETUP_STATUS.md](SETUP_STATUS.md) 在本机或数据盘重新配置。

## 操作约束

- 所有 GPU 任务必须由 tmux 启动。
- 禁止使用物理 GPU 0。
- 每次启动 GPU 任务前必须重新执行 `nvidia-smi` 检查。
- 不执行删除、覆盖、清理缓存或重建环境等操作，除非先获得用户确认。
- 项目代码、配置和报告继续保存在 `/home/imc/yzy/agent`。
- 用户已授权将模型、数据集、训练 checkpoint 和大体积缓存存放在数据盘 `/media/imc/data`；后续使用独立的 `yzy/agent/` 命名空间，避免与其他数据混放。
- 数据盘上的文件同样遵守“删除前先确认”，不擅自清理或覆盖。

## 快速检查

```bash
bash shared/scripts/cpu_smoke.sh
bash shared/scripts/check_gpu.sh 1
bash shared/scripts/docker_smoke.sh
```

项目二 CUDA Smoke Test：

```bash
bash project2-coding-agent-rl/scripts/start_gpu_smoke_tmux.sh 1
```
