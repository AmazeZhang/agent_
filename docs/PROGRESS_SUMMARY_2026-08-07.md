# 双项目当前进度总结

- 文档创建时间：2026-08-07 16:53:27 CST（UTC+08:00，北京时间）
- 状态数据截止时间：2026-08-07 16:53:27 CST
- 工作区：`/home/imc/yzy/agent`
- 远程仓库：`https://github.com/AmazeZhang/agent_`
- 总结前基线提交：`01ed23b9d64ef3a618c9ca1ba7ec6e28b25c5df7`
- 正式开发边界：[`DEVELOPMENT_SCOPE.md`](DEVELOPMENT_SCOPE.md)

## 一、总体结论

两个项目的基础环境、DeepSeek API 调用、工具执行和结果落盘链路均已跑通，但“工程链路可行”不等于“算法效果已经成立”。

| 项目 | 当前阶段 | 已确认结论 | 尚不能宣称 |
|---|---|---|---|
| 项目一：Trace 驱动 Agent 自进化 | 工程链路完成，诊断能力评测中 | tau2 轨迹采集、AgentRx 六阶段诊断和多配置对照均可运行 | 诊断准确率可靠提升、自进化闭环有效 |
| 项目二：Coding Agentic RL | 严格真实任务 Pilot 进行中 | SWE-agent 生成补丁、隔离复验、完整性检查链路可行 | 20 任务成功率、SFT/GRPO 训练收益 |

## 二、代码仓库与环境

### 2.1 Git 仓库

- 2026-08-07 已初始化根仓库并推送 GitHub `main` 分支。
- 初始提交：`01ed23b Initial import of agent research projects`。
- AgentRx、Agent Lightning、tau2-bench、SWE-agent、SWE-smith、rLLM 以 Git submodule 固定版本。
- SWE-agent 的定向浅拉取修复保存为 `project2-coding-agent-rl/patches/sweagent-shallow-reset.patch`。
- DeepSeek Key、虚拟环境、缓存、模型、数据集和运行目录均未进入 Git。
- 本地 SWE-agent submodule 当前显示 modified，这是已经应用的浅拉取修复；可复现补丁已提交到主仓库。

### 2.2 运行环境

- 主机：Ubuntu 22.04.4 LTS。
- GPU：8 × NVIDIA GeForce RTX 4090 D（24GB）；GPU 0 明确禁用。
- 已配置 5 个隔离 Python 环境，Agent Lightning、AgentRx、tau2、SWE-agent、SWE-smith 和 rLLM 基础导入均通过。
- Docker daemon、Python SDK 和 SWE-ReX 镜像可用。
- 2026-08-07 提交前 CPU smoke test 再次通过，本次未使用 GPU。
- 大模型、数据集、checkpoint 和大缓存统一放在 `/media/imc/data/yzy/agent/`。

## 三、项目一：Trace 驱动 Agent 自进化

### 3.1 已完成

1. 跑通 DeepSeek 与 tau2-bench 的 OpenAI 兼容调用及工具调用。
2. 完成 tau2 retail 轨迹采集：自然任务共 20 条，均为 Reward 1.0、DB match 通过。
3. 完成 tau2 到 AgentRx IR 的轨迹转换。
4. 通过项目级 DeepSeek 适配器跑通 AgentRx 的六阶段流程：
   `ir → static → dynamic → check → judge → report`。
5. 使用 AgentRx 上游 7 条真实失败轨迹建立严格失败诊断基准。
6. 完成零样本、标签 few-shot、带真实轨迹片段 few-shot 三种配置对照。

### 3.2 当前结果

| 配置 | 类别正确 | 精确步骤正确 | 备注 |
|---|---:|---:|---|
| 零样本 | 1/7 | 1/7 | 步骤 ±2 为 4/7 |
| 标签 few-shot | 2/7 | 3/7 | 共 73,240 tokens |
| 轨迹片段 few-shot v2 | 1/7 | 2/7 | 共 105,588 tokens |

三种配置没有表现出跨指标稳定提升。为避免在同一组 7 条测试轨迹上继续调参造成过拟合，当前已停止继续针对该测试集优化提示词。

### 3.3 当前边界与剩余工作

- 工程链路已经可用，但诊断准确率尚不足以支持“可靠自进化”结论。
- 下一阶段需要建立互不重叠的开发集和留出测试集。
- 需要把诊断结果真正反馈到 Agent 策略，并通过对照实验验证任务成功率是否提升。
- 在有独立测试集以前，不继续根据现有 7 条失败轨迹调提示词。

## 四、项目二：Coding Agentic RL

### 4.1 已完成的基础链路

1. 跑通 DeepSeek + SWE-agent 的代码检查、复现、编辑、测试和补丁提交。
2. 5 个受控小仓库任务全部独立复验成功，共 15/15 测试通过。
3. 完成一条真实 `pydicom__pydicom-1458` 探针；候选补丁未通过完整评测，诚实记录为 reward 0。
4. 从 SWE-smith 官方 train split 固定 20 条候选，覆盖 10 个轻量 Python 仓库。
5. 建立单提交净化仓库和 `HEAD^` 不可访问的强制完整性门禁。
6. 发现并纠正第 4–6 次运行的父历史泄漏；相关轨迹保留作证据，但永久排除出指标和训练数据。

### 4.2 SWE-smith Pilot 当前可信状态

正式汇总文件目前记录：

- 计划任务：20 条。
- 已登记运行：7 次。
- 严格可信且已完成独立评测：4 条，覆盖 3 个仓库。
- reward 1：2 条。
- reward 0：2 条。
- 无效运行：3 条。
- 暂时的严格成功率：2/4（50%）；样本过小，不能作为稳定性能估计。

可信结果如下：

| 任务 | 完整测试 | Reward | 状态 |
|---|---|---:|---|
| OAuthlib 任务 1 | 673 passed, 2 skipped | 1 | 可信 |
| OAuthlib 任务 2 | 673 passed, 2 skipped | 1 | 可信 |
| Pygments VimLexer | 5114 passed, 2 failed, 16 skipped | 0 | 可信；问题描述存在不完整标记 |
| Funcy curry/compose 净化重跑 | 201 passed, 2 failed | 0 | 可信 |

### 4.3 第 8 次运行的即时状态

- 运行完成时间：2026-08-07 12:26:10 CST。
- 任务：Funcy lookuper 净化重跑。
- SWE-agent 状态：`submitted (exit_cost)`。
- API 调用：41 次。
- 输入 token：599,272。
- 输出 token：68,979。
- 记录成本：`$0.02418304`。
- 已生成非空补丁，只修改 `funcy/calc.py`。
- 模型执行了 `git log --oneline -10`，但净化仓库只返回一个 grafted 提交，未暴露父历史或正确答案。
- 该补丁尚未在独立、含隐藏测试的评测 checkout 上运行，因此当前不能计入可信 20 任务指标。

### 4.4 项目二剩余工作

1. 对第 8 次 Funcy lookuper 补丁执行独立完整测试并登记结果。
2. 使用净化单提交仓库重跑此前无效的 Pygments Groff 任务。
3. 为其余 14 条任务制作并验证净化快照。
4. 完成全部 20 条任务的 rollout、隐藏测试、完整性判定和统一汇总。
5. 自动化“净化 → rollout → 独立评测 → 完整性检查 → 报告”流程。
6. 分别导出可信成功、可信失败和无效轨迹；无效轨迹不得进入训练集。
7. 达到数据质量门槛后，再在数据盘安装 `rllm[verl]` 和本地基座模型。
8. 最后执行小规模 LoRA SFT/GRPO smoke，验证训练、保存 checkpoint 和回归评测闭环。

## 五、项目完成标准

### 5.1 项目一达到阶段完备

- 独立开发集和留出测试集建立完成。
- 失败诊断在留出集上优于明确基线，而非只在 7 条样本上调优。
- 诊断反馈能够形成一次可复现的 Agent 改进闭环。
- 改进后任务成功率、成本和副作用均有对照结果。

### 5.2 项目二达到 Pilot 完备

- 20 条 SWE-smith 任务全部完成无泄漏 rollout 和独立隐藏测试。
- 每条任务都有补丁、轨迹、测试结果、成本和完整性状态。
- 被污染或无法复验的轨迹全部排除，不参与成功率和训练。
- 汇总报告能够从原始结果自动重建。

### 5.3 项目二达到训练闭环完备

- 可信训练集和验证集冻结并带版本号。
- 完成至少一次小规模 SFT 或 GRPO 训练 smoke。
- checkpoint、训练日志和评测结果均可复现。
- 与未训练基线进行同协议对照，确认是否存在真实收益。

## 六、操作约束

- 所有 GPU 作业必须在 tmux 中启动。
- 每次使用 GPU 前重新检查状态，禁止使用 GPU 0。
- 未经用户确认，不删除文件、容器、镜像、缓存、数据或环境。
- GPU/训练大文件优先使用 `/media/imc/data/yzy/agent/`。
- DeepSeek Key 仅保存在本地 `.secrets/deepseek.env`，不得写入 Git、日志或报告。

## 七、关键证据位置

- 总体可行性：`FEASIBILITY_REPORT.md`
- 小规模 Pilot：`PILOT_REPORT.md`
- 环境状态：`SETUP_STATUS.md`
- 项目一对照汇总：`project1-harness-evolution/diagnosis/benchmarks/upstream-tau-failure-comparison.json`
- 项目二正式汇总：`/media/imc/data/yzy/agent/project2/swesmith-pilot20/evaluations/summary.json`
- 项目二第 8 次运行：`/media/imc/data/yzy/agent/project2/swesmith-pilot20/runs/deepseek-v4-flash-run8-sanitized/`

后续每次达到明确里程碑，应更新本文件的状态截止时间，并保留历史版本，避免覆盖过去阶段的结论。
