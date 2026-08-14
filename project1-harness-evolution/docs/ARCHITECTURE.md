# 项目一架构文档：Trace 驱动 Agent 自进化闭环

> 更新：2026-08-14（r3 四臂完成后的架构定稿）
> 目的：讲清楚"目前流程是什么、harness 怎么接入、怎么修改、版本怎么管控"
> 实验结论与协议演进见 `reports/ablation_2026-08-08.md`；接手状态见工作区根
> `docs/PROJECT_STATUS_2026-08-10.md`

## 0. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│  scripts/（编排层，bash + python 入口）                          │
│  run_loop.sh ── run_tau2_baseline ─ partition ─ 诊断 ─ 四臂 ─ 汇总│
└──────┬──────────────────────────────────────────────────────────┘
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  optimizers/（适配层：harness 与执行域的桥）                     │
│  run_apo_loop.py   run_gepa.py        ← 每臂闭环 runner          │
│  tau2_rollout.py   gepa_adapter.py    ← 候选 → tau2 仿真 → reward │
│  candidate_filter.py  diagnosis_to_feedback.py                   │
└──────┬──────────────────────────────────────────────────────────┘
       ▼
┌────────────────────────────┬─────────────────────────────────────┐
│  vendor/agent-lightning    │  vendor/tau2-bench                  │
│  （优化 harness）          │  （执行域/评测环境）                │
│  Trainer · APO · GEPA      │  LLMAgent · UserSimulator · DB 校验 │
│  LightningStore(4747/4748) │  AGENT_INSTRUCTION 注入点           │
└────────────────────────────┴─────────────────────────────────────┘
       ▲                                  ▲
       └────── DeepSeek API（.secrets/deepseek.env，不入 git）──────┘
```

数据流向（一轮完整闭环）：

```
M1 基线采集 ──▶ M2 数据划分 ──▶ M3 失败诊断 ──▶ M4 优化 ──▶ M5 门控
retail40     dev/val/holdout   AgentRx         APO×2/GEPA×2  独立重跑×3
0.900        24/8/8（hash）     task27 根因     →候选过滤     多数票→gate→版本
```

## 1. 目前流程（M1–M5，每步的输入/处理/输出）

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| **M1 基线采集** | tau2 retail 40 任务 + v0 seed 提示 | `scripts/run_tau2_baseline.py`（seed 301，串行仿真） | `/media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json`（0.900） |
| **M2 数据划分** | task_manifest.json | `data/partition.py`：sha256 确定性哈希 → dev 60% / val 20% / holdout 20% | `data/datasets/partition_manifest.json`（24/8/8，hash 锁定） |
| **M3 失败诊断** | 基线失败轨迹 | `scripts/diagnose_baseline_failures.py` + AgentRx（DeepSeek 适配 CLI） | `data/diagnostics/summary.json`（task 27 = 类别 1 操作顺序错误：return+exchange 并行） |
| **M4 优化** | dev 集 + v0 资源 + 诊断反馈 | 四臂：APO/APO+诊断、GEPA/GEPA+诊断；候选过滤；best 在 val 独立重跑 ×3 | `runs/loop-{apo,gepa}-{plain,diagnosis}/round3.json` |
| **M5 门控** | 候选 val 多数票 vs 基线 val 多数票 | `evaluation/gate.py`：持平/回退/成本超限 → 拒绝；严格提升 → 接受 | 版本更新 `resources/versions/v{N+1}` 或拒绝记录入 CHANGELOG |

**r3 关键协议（相对 r1/r2 的三处修正，全部有实测触发）**：
1. **同尺度对照**：gate 基线 = 基线 v0 在 val8 的实测多数票（`runs/baseline_val_rerun.json`，0.875），而非 40 任务整体 0.900
2. **降噪重跑**：best 候选独立重跑 ×3，按任务多数票计成功率（LLM 单次评测噪声被 task 27 [0,0,0] 实证）
3. **持平=拒绝**：`c_rate <= b_rate` 一律拒绝（r2 曾把持平误判 accept，产生 seed 复制伪版本 v1/v2，已清理并修正语义）

## 2. Harness 怎么接入（三层解耦）

### 2.1 优化 harness：vendor/agent-lightning（Agent Lightning）

- **APO**（`agentlightning/algorithm/apo/apo.py`）：beam 搜索 + 文本梯度，`run_initial_validation` 后执行 `beam_rounds` 循环；`_history_best_score` 严格提升才替换 best
- **GEPA**（gepa 0.1.4 包）：`optimize(seed, trainset, valset, adapter, reflection_lm, ...)`，内部同样 strict_improvement 选 best
- **LightningStore**：`ClientServer` 策略 `managed_store=True` 时主进程拉起 store server（端口 4747，可用 `AGL_SERVER_PORT` 隔离多臂）；候选提示按 `resource_name` 注册/读取
- 接入方式：**不 fork 框架**，`run_apo_loop.py`/`run_gepa.py` 直接 import harness API，把 tau2 仿真包装成 harness 的 rollout/evaluate 回调

### 2.2 执行域：vendor/tau2-bench（tau2 retail）

- `tau2_rollout.py` 把**候选 system_prompt 注入执行域**：临时替换
  `tau2.agent.llm_agent.AGENT_INSTRUCTION`（UserSimulator 独立类不受影响），
  agent 与用户模拟器都走 DeepSeek API（litellm + openai 协议）
- 任务执行后由 **tau2 DB 校验**给 reward（0.0/1.0）——评测标准来自执行域，不来自优化器
- 每次 rollout 追加 JSONL 诚实记录（任务 id、候选 hash、reward、成本、终止原因）

### 2.3 适配层：optimizers/（本项目的桥）

| 文件 | 职责 |
|---|---|
| `run_apo_loop.py` | APO 臂闭环：Trainer/APO 装配 → 候选过滤 → val 独立重跑 ×N → gate → 版本 |
| `run_gepa.py` | GEPA 臂闭环：adapter + reflection_lm 装配 → 进化 → 过滤 → 重跑 → gate |
| `tau2_rollout.py` | 单任务仿真包装（seed 301、AGENT_INSTRUCTION 注入、reward、诚实日志） |
| `gepa_adapter.py` | GEPA 的 evaluate/reflective_dataset/DeepSeekLM；r3 起注入真实任务诉求 + 身份保真约束 |
| `candidate_filter.py` | 候选提示白名单过滤（防 seed 之外格式漂移；r2 曾因白名单与真实 seed 不符误杀 seed） |
| `diagnosis_to_feedback.py` | AgentRx 诊断 → 优化器可消费的反馈文本（diagnosis 臂专用） |

## 3. 怎么修改的（两类修改，两种策略）

### 3.1 vendor 子模块工作区修复 → patches/ 集中管理

vendor 指向上游官方仓库（microsoft/agent-lightning 等），**工作区修复不推送到上游**，
统一以 patch 文件保存在 `patches/`，保证实验可复现：

| patch | 修复对象 | 问题 | 修复方式 |
|---|---|---|---|
| `agentlightning-m3-agentops-role-fix.patch` | agent-lightning `adapter/messages.py` | AgentOps 0.4.21 插桩缺陷导致 APO 训练在 val 评测后崩溃（KeyError role/id/name） | 适配层按消息内容推断 role，call 字段 `.get()` 兜底 |

应用方式：`cd vendor/agent-lightning && git apply ../../patches/*.patch`

### 3.2 自有代码（optimizers/ evaluation/ scripts/ data/）→ 正常 git 提交

- 全部提交在本仓库 main 分支（AmazeZhang 身份），r1→r2→r3 协议演进保留完整历史
- 关键修改点（每处都有实验触发）：
  - gate 持平拒绝语义（r3，实测伪版本事件触发）
  - 独立重跑 ×3 多数票协议（r3，task 27 单次评测噪声触发）
  - GEPA 反思注入真实任务上下文 + 身份保真约束（r3，r2 候选"仿真操作员"退化触发）
  - 基线 val 同尺度对照（r3，r2 门控尺度不匹配触发）
  - APO 402 崩溃自动重试（r3 运维，DeepSeek 瞬时 402 触发）

## 4. 版本怎么管控（resources/versions/ + CHANGELOG + gate）

```
resources/versions/
├── CHANGELOG.md     ← 每行一条 JSON：{version, round, decision, reason, metrics}
└── v0/              ← 基线资源（system_prompt.txt / action_strategy.txt / tool_policy.txt）
```

版本决策流（M5）：

```
best 候选独立重跑 ×3 多数票 ─▶ evaluation/gate.py
                                  ├─ c_rate > b_rate 且成本不超限 → 接受
                                  │     → save_resources(v{N+1}) + CHANGELOG accept
                                  └─ 持平 / 回退 / 成本超限 → 拒绝
                                        → CHANGELOG reject（version: null）
```

- **版本号递增**由 `resources/loader.latest_version()` 决定，gate 接受才写新版本
- **每条 CHANGELOG 记录都是证据链**：reason 含具体数字（如
  `未产生收益（0.875 ≤ 基线 0.875，持平/无提升）`），metrics 与 round 记录可复核
- **诚实性铁律**：gate 判定与指标全部由 `evaluation/metrics.py` 计算，禁止手工改数；
  持平/回退绝不产生新版本（r3 清理过 2 个伪版本，教训已固化进自检用例）
- 当前版本库状态：**仅 v0**，r1/r2/r3 共 10 条 CHANGELOG（6 条拒绝历史 + 4 条 r3 拒绝）

## 5. r3 最终状态（2026-08-14 核对）

- 四臂 r3 全部完成：GEPA×2 持平拒绝；APO×2 内部 val 1.000 假信号被独立重跑揭穿
  （task 27 [0,0,0] 稳定失败），best 逐字节回退 seed（307 字符，已比对 v0）→ 拒绝
- **自进化工程闭环完整跑通**（采集→划分→诊断→优化→评测→决策→版本管控），
  但**方法尚未产生真实收益**——四臂均无提升，诚实结论不产生伪版本
- 一键复现：`bash scripts/run_loop.sh`（部分环节如诊断需人工确认，见 README）
- 详细数据：`reports/ablation_2026-08-08.md` r3 追加章节
