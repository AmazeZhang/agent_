# Project2 阶段 1a 交付报告（2026-08-08）

> 阶段 1a（数据管道）核心交付完成：数据全量落盘、任务池构建、环境构建、
> gold 复核、SFT 数据提取。关键发现：**SWE-smith 语义与 SWE-bench 相反**。

## 1. 交付清单

| 资产 | 位置 | 规模 |
|---|---|---|
| SWE-smith tasks | /media/.../datasets/swe-smith-tasks/ | 11 parquet，52k 任务 |
| SWE-smith trajectories | /media/.../datasets/swe-smith-trajectories/ | 8 parquet，26k 轨迹 |
| 任务池（最终） | phase1/task_pool/ | **eval 10 + train 148** |
| repos（clone） | phase1/repos/ | 67 仓库，164/164 任务 commit 可解析 |
| 任务环境 | phase1/work/（worktree）+ phase1/eval-venvs/ | 10 eval 任务 venv 就绪 |
| gold 复核结果 | phase1/stats/gold_review*.jsonl | 16 任务复核，10 OK |
| SFT 数据 | phase1/sft_data/sft_train.jsonl | 287 轨迹（OpenRLHF multiturn） |

## 2. 关键发现 1：SWE-smith 语义反转（重要）

**SWE-smith 数据集的 `patch` 字段是"引入 bug"的 diff，不是修复**（与 SWE-bench 相反）：
- agent 的**初始代码状态** = checkout commit + `git apply patch`（broken）
- 正确修复 = `git apply -R patch`（反向变换）
- 数据集基于仓库"干净 commit"（>80% 测试通过），通过代码变换（combine_file /
  func_pm_* / lm_rewrite / PR 反向）注入 bug

**实测验证**（cloudpickle 任务）：clean 状态 f2p 6/6 通过（修复可达）→
apply patch 后 6/6 全挂（bug 真实）。已写入 spec 3.5 节。

→ **阶段 2b 评测环境必须 checkout commit + apply bug patch 作为初始状态**，
   否则评测的是无 bug 代码，结果无效。

## 3. 关键发现 2：数据质量审计

| 发现 | 影响 | 处置 |
|---|---|---|
| trajectories **patch 列 71% 错位**（pool 内 220/311，呈循环移位；如 cloudpickle 实例的 patch 是 voluptuous 的） | SFT 若误用 patch 列则污染 | SFT 只用 messages 列（抽查与 instance 匹配 ✅）；gold patch 一律从 tasks parquet 取 |
| `pr_*` 类型任务**无 hidden tests**：注入 bug 不被现有 f2p 覆盖（4/4 实测 FAIL） | 不适合做 eval（broken 状态 f2p 也过，无法区分模型好坏） | eval 集全部用生成测试类任务；pr_* 保留训练用 |
| 部分 repo 在 Python 3.11 有环境性测试失败（safety/radon/glom 的 cli 测试） | 基线不净 → 判定 FAIL | 剔除出 eval；若作训练任务轨迹仍有效 |

## 4. Gold 复核方法（G1 门禁，gold_review.py）

三条件（每任务，平均 ~2-5 分钟）：
1. **clean 状态** f2p 全过 → 修复可达、测试有效
2. **apply bug patch 后**（broken）f2p 有失败 → bug 真实
3. **p2p delta=0**（broken vs clean）→ 不破坏既有行为

流程：worktree @ commit（blob:none 共享对象库）→ 任务 venv + `pip install -e .`
（失败则 PYTHONPATH fallback）→ pytest；测试依赖从 ModuleNotFoundError
自动补装（≤4 轮）。注意 pytest `-q` 失败摘要解析 `FAILED <node> - <reason>`，
`--tb=short` 才能暴露 ModuleNotFoundError。

**复核结果**（16 任务 / 三轮）：
- 第一轮 eval 10：4 OK（cloudpickle/patsy/pyparsing/iniconfig），
  6 FAIL（4 个 pr_* bug 不覆盖 + 2 个环境不兼容）
- 第二轮候选 6：5 OK（markdownify/tomli/h11/PySnooper/apispec），
  1 FAIL（glom cli 测试环境问题）
- 第三轮替补 1：1 OK（textfsm）

**最终 eval 集 10 任务**（10 个不同 repo，f2p 178 个测试点 / p2p 2951 个）：
cloudpickle, patsy, pyparsing, iniconfig, markdownify, tomli, h11,
PySnooper, apispec, textfsm —— 难度覆盖 rr 0.25-0.75，与 phase0 5 任务
合并为 15 个 eval 任务（2b 评测）。

## 5. SFT 数据（1a 附带交付）

- 287 条 resolved 轨迹（覆盖 148 train 任务中的 130+），平均 43.7 消息
- 协议规范化：system → phase0 SYSTEM_TEMPLATE（逐字节一致）、
  SWE-agent ` ```fence` → `<execute><command>`、submit → exit ✅
- messages 与 instance 匹配已验证（cloudpickle/PySnooper 抽查）✅
- 下一步：1b 用 OpenRLHF 跑 SFT（新 venv .venvs/phase1-openrlhf，不动 rllm-base）

## 6. 下一步

1. **1b**：OpenRLHF venv + SFT 训练（Qwen2.5-Coder-7B-Instruct，max_len 32K，
   lr 5e-5）+ 快速验证（2-3 个 eval 任务）
2. **2a**：rllm GRPO 闭环（官方 4B 配方缩到 7B/24GB，context 24-32K）
3. **2b**：三臂评测（zero-shot / SFT / SFT+RL）× 15 任务，环境 = checkout +
   apply bug patch
