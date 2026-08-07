# SPEC 00 — 项目一开发总体规格

- 版本：v0.1（草稿，待评审）
- 日期：2026-08-07
- 依据：`项目一Agent自进化开源项目调研报告.md`（2026-08-07）+ `docs/DEVELOPMENT_SCOPE.md` + `docs/PROGRESS_SUMMARY_2026-08-07.md`
- 状态：等待用户评审后进入开发

## 1. 目标

按调研报告第 1 节定义，把项目一从"诊断半环"补全为完整闭环：

```text
Agent 执行
→ 发现失败
→ 诊断根因（AgentRx，已完成）
→ 生成候选修改（新增：APO → GEPA）
→ 在验证集重新运行（新增）
→ 检查收益与回归（新增）
→ 接受或拒绝候选（新增：回归门控）
→ 更新 Harness 版本（新增：资源版本化）
```

**注意**：本项目只做 Harness/Prompt 层进化，不做权重训练（权重更新属于项目二）。

## 2. 技术选型（来自调研报告）

| 组件 | 角色 | 阶段 |
|---|---|---|
| Agent Lightning（vendor 已含 v0.2+，内置 APO） | 核心运行框架：rollout、trace、resource、Trainer | M0 起 |
| tau2-bench（vendor 已有） | 任务环境与评测（DB 校验 + NL judge） | 已有 |
| AgentRx（vendor 已有） | 失败诊断：六阶段 + 10 类分类 + 证据 | 已有 |
| GEPA（外部，待安装 `pip install gepa`） | 第二阶段高级优化器，读诊断文本（Actionable Side Information） | M4 |
| DeepSeek API（OpenAI 兼容） | 模型本体 `deepseek-v4-flash`，APO/GEPA 的优化 LLM 也可用 DeepSeek | 全程 |

关键依据：Agent Lightning vendored 版本已含 `agentlightning/algorithm/apo/`（`apo.py` + `prompts/`），官方示例 `examples/apo/room_selector_apo.py` 演示内置 APO 用法：`Trainer(algorithm=APO(...), n_runners, initial_resources, adapter=TraceToMessages())` + `trainer.fit(agent, train_dataset, val_dataset)`。API 形态与调研报告 2.1 节一致。

## 3. 里程碑划分

| 里程碑 | 内容 | 验收标准 | 预计资源 |
|---|---|---|---|
| **M0** | Agent Lightning 接入验证：跑通官方 minimal + APO 示例（换 DeepSeek endpoint） | 官方 APO 示例在本机完成一轮训练，日志落盘 | CPU + API，无 GPU |
| **M1** | tau2 适配 + 基线：定义 `tau2 任务 → Agent Lightning dataset` 适配器、agent 函数、reward；采集 40–80 条自然任务，按"执行 → 失败 → 诊断 → 轨迹落盘"流程建立基线 | 基线任务成功率、成本、诊断样本数可复现；轨迹与诊断结果结构化落盘 | CPU + API（每条 ~几美分） |
| **M2** | 数据与度量：开发/验证/留出划分；闭环指标定义（成功率/成本/副作用/诊断准确率）；可优化 Harness 资源定义（prompt 模板、工具约束、执行策略）；非法候选过滤规则 | 划分脚本与指标脚本可用；留出集从未进入优化 | 无 GPU |
| **M3** | APO 最小闭环：AgentRx 诊断输出 → APO 文本批评/反馈适配器；候选生成 → 验证集重跑 → 收益与回归检查 → 接受/拒绝 → 资源版本更新；与无诊断基线（纯 APO）对照 | 至少一轮完整闭环；留出集未污染；对照数字落盘 | CPU + API |
| **M4** | GEPA 接入：AgentRx 诊断 → Actionable Side Information 适配；多资源联合优化；与 M3 的 APO 结果做消融 | GEPA 一轮优化完成；消融表产出 | CPU + API |
| **M5** | 最终交付：消融矩阵、成本/回归分析、报告、面试叙事 | 与 `DEVELOPMENT_SCOPE.md` 2.3 验收项逐条对照 | 无 GPU |

## 4. 操作约束（硬性）

1. **GPU**：禁止使用物理 GPU 0（已被占用 387MiB，疑似显示服务）。每次使用 GPU 前先 `nvidia-smi` 诊断。APO/GEPA 主路径 **CPU + API 即可**，GPU 仅用于可选的大规模并行 rollout；不满足时不阻塞开发。
2. **tmux**：所有长时间运行的命令（采集、训练、批量评测）必须在 tmux 中启动。
3. **密钥**：DeepSeek key 仅存 `.secrets/deepseek.env`（工作区根），不进 Git、日志、报告。
4. **数据**：大文件（轨迹、缓存、checkpoint）放 `/media/imc/data/yzy/agent/`。
5. **不删除**：未经用户确认不删除文件、数据、环境、容器、镜像、缓存。
6. **环境**：使用 `project1-harness-evolution/.venvs/` 下的 agent-lightning / agentrx / tau2 / agent-tools 四个隔离环境，不混用。
7. **诚实原则**：失败、无效实验、无提升结果如实落盘，不选择性报告。

## 5. 目录规划（新增部分）

```text
project1-harness-evolution/
├── spec/                    # 本目录：spec 文档
├── optimizers/              # M3/M4：APO/GEPA 接入、反馈适配器、候选过滤
├── evaluation/              # M2/M3/M5：指标、回归门控、版本记录、消融脚本
├── resources/               # M2：可优化 Harness 资源定义（prompt 模板、工具约束、执行策略）
├── data/                    # M1/M2：任务数据集划分、诊断反馈样本（小文件，大文件走数据盘）
└── runs/loop-*/             # 每轮闭环的原始结果（与 diagnosis/runs 并列）
```

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 20 条任务太少，闭环无统计意义 | M1 扩量到 40–80 条（API 成本可控）；划分 60/20/20 或 70/15/15 |
| Agent Lightning 与 DeepSeek 兼容问题（function calling 差异） | M0 用官方示例 + DeepSeek endpoint 先行验证；报告已知 APO 用 OpenAI-compatible API |
| 诊断准确率低（类别 1/7）导致反馈质量差 | 闭环主指标是任务成功率，诊断只是反馈信号；消融对比"有诊断 vs 无诊断"验证诊断价值 |
| APO 生成非法 prompt | M2 定义合法性校验（格式、长度、禁止修改测试文件等约束），M3 接入过滤 |
| 成本不可控 | 每实例成本上限（沿用 $0.05/instance 模式），APO/GEPA 搜索轮数设上限 |
| 留出集污染 | 划分后 hash 锁定；脚本断言留出集 ID 不进入任何优化调用 |

## 7. 里程碑间依赖

M0 → M1 → M2 → M3 → M4 → M5，顺序执行；M2 的指标定义可在 M1 并行起草，但划分脚本必须在 M1 采集完成前定稿（防泄漏）。
