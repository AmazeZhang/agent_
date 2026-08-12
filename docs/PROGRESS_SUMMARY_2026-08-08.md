# 双项目当前进度总结

> **历史快照提示**：本文截止于 2026-08-08 早期，后续同日完成了项目一 r3 和
> 项目二 Phase 1a，且 2026-08-09 已进入 Phase 1b SFT 显存攻关。当前接手入口为
> `PROJECT_STATUS_2026-08-10.md`；本文仅用于追溯当时状态。

- 文档创建时间：2026-08-08 09:30:00 CST（UTC+08:00，北京时间）
- 状态数据截止时间：2026-08-08（项目一 M5 收尾，四臂 round 2 全部完成）
- 工作区：`/home/imc/yzy/agent`
- 远程仓库：`https://github.com/AmazeZhang/agent_`
- 上一版进度：[`PROGRESS_SUMMARY_2026-08-07.md`](PROGRESS_SUMMARY_2026-08-07.md)（保留历史）
- 正式开发边界：[`DEVELOPMENT_SCOPE.md`](DEVELOPMENT_SCOPE.md)

## 一、总体结论

项目一（Trace 驱动 Agent 自进化）达到阶段完备：**闭环工程完整跑通并全链可追溯，
但方法本轮未产生收益**（四臂候选全部 gate 拒绝，版本停留 v0）。项目二状态与
2026-08-07 版一致（见上一版 §四）。

| 项目 | 当前阶段 | 已确认结论 | 尚不能宣称 |
|---|---|---|---|
| 项目一：Trace 驱动 Agent 自进化 | **M1–M5 完成**（四臂消融 round 2） | 自进化闭环工程可行、四臂各完成一轮、诊断臂方向性优势 | 方法产生真实收益（本轮全部 reject） |
| 项目二：Coding Agentic RL | 严格真实任务 Pilot 进行中 | SWE-agent 补丁、隔离复验、完整性检查链路可行 | 20 任务成功率、SFT/GRPO 训练收益 |

## 二、项目一里程碑进度（M1–M5）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M1 | tau2 retail 40 任务基线（0.900 成功率，$0.058/任务）+ dev/val/holdout 划分（24/8/8，hash 3694ebbc） | ✅ |
| M2 | 轨迹采集 + 格式转换 + AgentRx 失败诊断（10 类失败类别）+ 候选语义过滤（白名单对齐真实 seed） | ✅ |
| M3 | APO 最小闭环（diagnosis/plain 双臂）+ AgentOps 插桩缺陷适配层修复（3 处，patch 保存） | ✅ |
| M4 | GEPA 进化闭环（diagnosis/plain 双臂）+ 线程安全指令注入 | ✅ |
| M5 | 四臂消融 + 回归门控 + 验收清单 + 消融报告 | ✅（收尾完成） |

## 三、项目一 round 2 结果（2026-08-08，三处修复后）

| 臂 | 内部 val 分 | val 独立重跑 | gate | 版本 |
|---|---|---:|---:|---|---|
| baseline | 0.900 (40 任务) | — | — | v0 |
| apo-plain | 0.875 | 0.750 (6/8) | reject（-15pp） | v0 |
| apo-diagnosis | 1.000 | 0.875 (7/8) | reject（-2.5pp） | v0 |
| gepa-plain | 0.875 | 0.750 (6/8) | reject（-15pp） | v0 |
| gepa-diagnosis | 0.875 | 0.875 (7/8) | reject（-2.5pp） | v0 |

- 诊断臂两对均高于对应 plain 臂（+1/8 任务），方向一致但 8 任务样本不显著
- GEPA 两臂各生成 1 个真实新候选（2087/1920 字符），val 全量评测未超 seed → best 回退
- 本轮修复：CandidateFilter 白名单对齐真实 seed / GEPA 反射双层包装 / skip_perfect_score / runner 过滤失败落盘
- 诚实结论：闭环工程可行但方法尚未产生收益（DEVELOPMENT_SCOPE 2.3 模板）

## 四、项目一验收状态

- DEVELOPMENT_SCOPE 2.2 交付：7/8 达成；2.3 完成判定：4/4 达成；SPEC 04 验收：4/5 达成
- 未达成项（如实标注）：一键复现脚本 `scripts/run_loop.sh` 未创建（现有 `scripts/restart_all_arms.sh`
  为四臂并行启动，已验证）；方法收益未达成（gate 全部拒绝，属诚实结果而非交付缺口）
- 详见 `project1-harness-evolution/reports/acceptance_checklist.md`（2026-08-08 定稿）

## 五、项目二

状态与 2026-08-07 版完全一致，未在本轮触碰（避免交叉污染）。见
[`PROGRESS_SUMMARY_2026-08-07.md`](PROGRESS_SUMMARY_2026-08-07.md) §四。

## 六、操作约束（不变）

- 所有 GPU 作业必须在 tmux 中启动；每次使用 GPU 前重新检查状态，禁止使用 GPU 0。
- 未经用户确认，不删除文件、容器、镜像、缓存、数据或环境。
- 大文件优先使用 `/media/imc/data/yzy/agent/`。
- DeepSeek Key 仅保存在本地 `.secrets/deepseek.env`，不得写入 Git、日志或报告。

## 七、关键证据位置

- 项目一四臂结果：`project1-harness-evolution/runs/loop-*/round2.json`
- 项目一消融报告：`project1-harness-evolution/reports/ablation_2026-08-08.md`
- 项目一验收清单：`project1-harness-evolution/reports/acceptance_checklist.md`
- 版本变更历史：`project1-harness-evolution/resources/versions/CHANGELOG.md`
- 项目二正式汇总：`/media/imc/data/yzy/agent/project2/swesmith-pilot20/evaluations/summary.json`

后续每次达到明确里程碑，应更新本文件的状态截止时间，并保留历史版本，避免覆盖过去阶段的结论。
