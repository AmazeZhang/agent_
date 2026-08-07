# 双项目基本可行性报告

日期：2026-08-06

## 总结

两个项目均已达到“核心链路技术可行”的门槛，可以继续进入小规模正式实验。当前结论只覆盖环境、模型调用、工具执行、轨迹落盘和结果复验，不等同于算法收益、训练效果或论文主张已经成立。

## 项目一：Trace 驱动 Agent 自进化

已完成：

1. Agent Lightning、AgentRx、tau2-bench 分别安装在隔离环境并通过导入/CPU smoke。
2. DeepSeek OpenAI 兼容接口、Chat Completions 和工具调用验证通过。
3. tau2 mock 任务 Reward 1.0。
4. tau2 retail 真实任务 0、1、2 均得到最终 Reward 1.0；任务 2 使用项目级适配器完成 DeepSeek NL judge 和成本记录。
5. 任务 2 轨迹已转换为 AgentRx wrapper 输入，并成功生成含 25 个有效步骤的 IR。

主要证据：

- `project1-harness-evolution/vendor/tau2-bench/data/simulations/p1-feasibility-retail3-deepseek-v4-flash-20260806/results.json`
- `project1-harness-evolution/vendor/tau2-bench/data/simulations/p1-feasibility-retail-task2-adapter-20260806/results.json`
- `project1-harness-evolution/diagnosis/runs/retail-task2-ir/trajectory_ir.json`

剩余工作：实现 AgentRx 的 DeepSeek endpoint adapter，之后才能验证完整 invariant 提取、judge 和闭环改进；再扩大任务数量并设计对照组。

## 项目二：Coding Agentic RL

已完成：

1. SWE-agent、SWE-smith generate 组件和 rLLM 基础环境通过导入/CPU smoke；Docker daemon 与 SDK 可用。
2. DeepSeek 函数调用接口通过独立最小测试。
3. SWE-agent 在受控 buggy-calculator 仓库上完成端到端修复：读取代码、创建复现、运行测试、编辑代码、复测并提交补丁。
4. 轨迹状态为 `submitted`，共 12 次 API 调用，14,245 输入 token、389 输出 token，记录成本约 `$0.0008292592`。
5. 补丁经 `git apply --check`，并在断网 CPU-only Docker 容器中独立应用与执行测试，结果为 3/3 通过、退出码 0。

主要证据：

- `project2-coding-agent-rl/runs/sweagent-feasibility-deepseek-v4-flash-20260806-retry2/0ed001/0ed001.traj`
- `project2-coding-agent-rl/runs/sweagent-feasibility-deepseek-v4-flash-20260806-retry2/0ed001/0ed001.patch`
- 保留的复验容器：`agent-p2-patch-eval-20260806`，状态 exited，exit code 0。

剩余工作：用 5–10 个代表性仓库任务做小规模 pilot，建立成功率、成本、轨迹质量基线；确认磁盘和 GPU 资源后，再安装 `rllm[verl]` 并进入 SFT/GRPO。当前不应宣称任何训练提升。

## 风险与约束

- 根分区约 137GB 可用、使用率 92%；完整训练栈、数据集和批量仓库镜像都应单独过磁盘门禁。
- 公共 Python Docker 镜像拉取曾遇到镜像源 TLS/超时；当前用本地已有基础镜像构建的 Python 3.11 + SWE-ReX 镜像绕过。
- AgentRx 完整 DeepSeek 接入尚未完成。
- 后续所有 GPU 工作仍须先检查 GPU 状态、禁止 GPU 0，并在 tmux 中运行。
- 未执行任何人工删除或清理；退出容器、失败环境和镜像均保留，后续如需释放空间须先确认。
