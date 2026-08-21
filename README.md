# Agent 算法项目工作区

本目录当前包含五个独立项目：

- `project1-harness-evolution/`：Trace、失败诊断、自进化与可靠性评测。
- `project2-coding-agent-rl/`：Coding Agent 数据、轨迹、SFT 与 Agentic RL。
- `project3-search-agent-rl/`：Search Agent 的检索环境、轨迹级/步骤级信用分配与Agentic RL。
- `project4-opensearch-vl-rl/`：OpenSearch-VL 多模态搜索 Agent 的源码审计、学习路线与小规模复现规划。
- `project5-mini-chartqa/`：基于 VTool-R1/veRL 的 MiniChartQA 视觉工具强化学习实验原型。

项目四、项目五目前处于审计/原型阶段，不应表述为已经完成训练复现。第三方源码与个人文档、配置修改在各项目 README 中分别标注。

环境和当前阻塞项见 [SETUP_STATUS.md](SETUP_STATUS.md)。

核心可行性见 [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md)，五任务小规模验证和训练门槛判断见 [PILOT_REPORT.md](PILOT_REPORT.md)。

当前双项目接手状态见
[docs/PROJECT_STATUS_2026-08-10.md](docs/PROJECT_STATUS_2026-08-10.md)。该文档同时记录
项目一 r3 最终结论、项目二 Phase 1a/1b 状态、未提交工作区与当前阻塞。
项目二本轮 fused CE 与真实一步 SFT 的专项证据见
[PROJECT2_PHASE1B_SMOKE_REPORT_20260810.md](project2-coding-agent-rl/PROJECT2_PHASE1B_SMOKE_REPORT_20260810.md)。

旧进度总结保留为历史快照：
[2026-08-08](docs/PROGRESS_SUMMARY_2026-08-08.md)、
[2026-08-07](docs/PROGRESS_SUMMARY_2026-08-07.md)。

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

项目三固定使用`project3-search-agent-rl/vendor/verl-agent`，上游已经包含Search-R1环境及
GRPO、GiGPO等训练链路，避免同时维护第二套旧版verl源码。

虚拟环境、密钥、模型、数据集和运行目录不会上传；它们需要按
[SETUP_STATUS.md](SETUP_STATUS.md) 在本机或数据盘重新配置。

## 操作约束

- 所有 GPU 任务必须由 tmux 启动。
- 禁止使用物理 GPU 0。
- 物理 GPU 5 有历史掉卡记录，默认不使用；项目三稳定卡集合为 `1,2,3,4,6,7`。
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
