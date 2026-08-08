# 项目二 Coding Agentic RL 完善实施 Spec

- 制定时间：2026-08-08
- 依据：`DEVELOPMENT_SCOPE.md` 第三节（项目二验收标准）与 `PROGRESS_SUMMARY_2026-08-07.md` 第四节
- 本 spec 只覆盖项目二剩余工作；项目一另行处理。

---

## 一、当前状态快照（核查于 2026-08-08）

### 1.1 已完成

| 项 | 状态 |
|---|---|
| 20 条任务选样 | 已固定：10 仓库 × 2（oauthlib / pygments / funcy / bottlepy / stackprinter / boltons / typeguard / flake8 / tenacity / iniconfig） |
| 仓库克隆 | 6/10 仓库已克隆（oauthlib×2、pygments×2、funcy×2），其余 7 仓库未克隆 |
| 净化快照 | 3 个（funcy-curry、funcy-lookuper、pygments-groff），单提交、`HEAD^` 不可访问 |
| rollout 运行 | 8 次（run1–run8），其中 run4–6 因父历史泄漏被判无效（保留作证据） |
| 严格可信结果 | 4 条：OAuthlib×2 reward 1，Pygments VimLexer reward 0，Funcy curry 净化重跑 reward 0 |
| 待办在途 | funcy-lookuper（run8）已提交补丁，**尚未**独立评测登记；pygments-groff 净化快照已就绪，**尚未**重跑 |
| 环境 | `rllm-base` 已装 rllm / torch / transformers；**缺** peft、vllm、verl；无基座模型（hf-cache 仅有 SWE-smith 数据集） |
| GPU | 1–7 空闲（各 ~24GB 可用）；GPU 0 禁用 |

### 1.2 缺口总览（对应验收标准）

1. 20 条严格 Pilot 未完成：剩 16 条未跑（14 条新任务 + funcy-lookuper 评测 + pygments-groff 重跑）。
2. 无自动化流水线："净化 → rollout → 独立评测 → 完整性判定 → 汇总"目前为分散手工步骤，`summary.json` 无法从原始结果一键重建。
3. 无训练数据导出：可信成功/失败轨迹、无效隔离清单及追溯性尚未脚本化（pilot5 有 `export_verified_rollouts.py` 先例，swesmith 无）。
4. 训练栈未安装：`rllm[verl]`（vendor rllm 对应 verl==0.8.0）、vllm、peft 均缺；无本地基座模型。
5. 训练闭环未做过：无 LoRA SFT 真实运行、无 GRPO smoke、无 checkpoint 保存/加载、无训练前后同协议评测。

---

## 二、目标与验收（与 DEVELOPMENT_SCOPE §3 对齐）

| 编号 | 验收项 | 本 spec 对应 WP |
|---|---|---|
| A1 | 20 条严格 Pilot 全部完成并独立评测 | WP1、WP2 |
| A2 | 数据清洗与完整性门禁自动运行 | WP2、WP3 |
| A3 | SFT 与 GRPO 各完成一次真实最小训练（非仅配置/导入测试） | WP4、WP5、WP6 |
| A4 | checkpoint 可保存、加载并用于评测 | WP5、WP6 |
| A5 | 有训练前后同协议结果（无论是否提升，如实报告） | WP7 |

约束（全局）：GPU 作业一律 tmux 启动；每次启动前跑 `shared/scripts/check_gpu.sh`；禁止物理 GPU 0；模型/数据/checkpoint 放 `/media/imc/data/yzy/agent/`；DeepSeek Key 不进 Git/日志/报告；不删除既有文件、容器、镜像、缓存。

---

## 三、工作包分解

### WP0 收尾两条在途任务（不依赖 GPU，先行）

目标：把可信结果从 4 条推进到 6 条，且跑通"已提交补丁 → 独立评测 → 登记"的完整手工流程，为 WP2 自动化提供参照。

- **T0.1 funcy-lookuper（run8）评测**：将 run8 补丁应用到 `eval-repos/funcy-lookuper-3y0j7te5`（含隐藏测试的 checkout），恢复测试后跑完整 pytest，登记 full_test_results 与 reward；同时做完整性复核（净化仓库无父历史、模型未接触 gold patch）。
- **T0.2 pygments-groff 净化重跑**：用已就绪的净化快照重跑 rollout（DeepSeek + SWE-agent，沿用 run8 配置模板：40 calls、per-instance 上限 $0.05），随后执行与 T0.1 相同的独立评测与登记。
- 交付：两条记录进入正式汇总；若 pygments-groff 出现与 VimLexer 类似"问题描述未覆盖 bug patch"情况，按既有规则加 `problem_statement_underdescribes_bug_patch` 标记。
- 验收：`evaluations/summary.json` 可信计数变为 6，且每条都有完整测试证据路径。

### WP1 剩余 14 条任务的数据制作与 rollout（不依赖 GPU，DeepSeek API）

目标：完成全部 20 条严格任务。

- **T1.1 克隆 7 仓库**：bottlepy、stackprinter、boltons、typeguard、flake8、tenacity、iniconfig（`repos/` 下已有 6 仓库为参照）。
- **T1.2 净化快照**：每仓库按官方 `base_commit` 制作单提交、`HEAD^` 不可访问的快照到 `sanitized-repos/`，用现有 `materialize_local_swesmith_instance.py` 的 `HEAD^` 门禁强制校验；7 仓库 × 2 任务 = 14 个净化快照。
- **T1.3 rollout**：每条任务一次可信 rollout（DeepSeek-v4-flash + SWE-agent，run8 配置为模板：40 calls、`per_instance_cost_limit=0.05`、temperature 0）。运行时长较长，即使无 GPU 也放入 tmux 会话并写日志。
- **T1.4 独立评测**：每条补丁在含隐藏测试的 eval checkout 上完整测试（FAIL_TO_PASS + 全套件），记录 passed/failed/skipped、reward、token、成本、完整性状态。
- 成本预算：14 条 × ~$0.01–0.05 ≈ $0.3–0.8，全部走 DeepSeek 现有关键词。
- 交付：20/20 任务具备"轨迹 + 补丁 + 隐藏测试结果 + reward + token + 成本 + 完整性状态"完整记录。
- 验收：A1。

### WP2 自动化流水线脚本（核心工程交付）

目标：把 WP0/WP1 的手工步骤固化为一条命令可复现的流水线，`summary.json` 可由原始结果重建。

新增脚本（放 `scripts/`，参照既有脚本风格）：

| 脚本 | 职责 |
|---|---|
| `pipeline_sanitize.sh` | 克隆 → 净化单提交快照 → 完整性门禁（拒绝 `HEAD^` 可访问） |
| `pipeline_rollout.sh` | 生成 run_batch 配置（DeepSeek 注册表 + 40 calls 上限）→ SWE-agent `run_batch` → 轨迹/补丁/日志落盘 |
| `pipeline_evaluate.sh` | 补丁 → eval checkout（恢复隐藏测试）→ 完整 pytest → 解析 full_test_results |
| `pipeline_integrity.sh` | 校验单提交、`HEAD^` 不可访问、模型未读 gold patch、未改测试文件 |
| `pipeline_summary.py` | 扫描 `runs/` + `candidate-evals/` → 重建 `evaluations/summary.json`（含可信/无效判定、reward 统计） |
| `run_pilot20.sh` | 一键编排上述五步，支持 `--tasks` 白名单与断点续跑 |

- 完整性判定规则固化：父历史可访问 / 模型接触 gold patch / 无法复验 → `invalid`，永久排除出指标与训练数据（与已有 `integrity_correction` 语义一致）。
- 验收：A2；在清空 `evaluations/` 后重建 `summary.json` 与现有记录逐项一致。

### WP3 训练数据导出

目标：冻结可信训练集与验证集，满足"每条样本可追溯到原始任务、补丁、独立测试证据；无效轨迹不得进入训练"。

- **T3.1 可信成功轨迹导出**（SFT 正样本）：每条 = instance_id + problem_statement + 完整消息轨迹（动作序列）+ 最终补丁 + 独立测试证据路径 + token/成本。
- **T3.2 可信失败轨迹导出**（GRPO 负样本 / 过程奖励参照）：同样字段 + 测试失败详情。
- **T3.3 无效轨迹隔离清单**：run4–6 及任何后续 invalid 运行的轨迹单独归档，写入清单，脚本层面禁止其进入训练目录。
- 产出到 `/media/imc/data/yzy/agent/project2/training-data/`，带 manifest（schema 版本、导出时间、来源 run 映射）。
- 验收：成功/失败/无效三类文件齐全，每个样本可反向定位到 run 目录与测试证据。

### WP4 训练环境（GPU 作业开始）

目标：在数据盘安装 `rllm[verl]` 完整训练栈并验证；下载本地基座模型。

- **T4.1 依赖安装**：数据盘新建 venv（如 `/media/imc/data/yzy/agent/project2/training/.venvs/train`），安装 `rllm[verl]`（vendor submodule 版本，对应 verl==0.8.0、vllm）+ peft；以 rllm 自带 trainer 为优先（vendor 已含 `rllm/trainer/sft`、`rllm/trainer/agent_trainer.py`、`examples/harbor_swe` 的 SWE agentic RL 先例），必要时降级方案为 transformers+peft（SFT）与 verl 原生命令。
- **T4.2 基座模型**：按用户确认结果下载（见 §五 决策点 1），放 `/media/imc/data/yzy/agent/models/`，配 hf-cache 环境变量指向数据盘。
- **T4.3 冒烟验证**：在 GPU 1–7 中选一空闲卡（先 `check_gpu.sh`，禁 GPU 0），tmux 内跑一次最小前向/rollout 冒烟（沿用 `start_gpu_smoke_tmux.sh` 入口模式），确认 vLLM 推理与训练链路可用。
- 验收：训练栈导入、单卡 3B 级模型前向/推理冒烟通过；资源使用与安装日志落盘。

### WP5 LoRA SFT smoke（真实训练 #1）

目标：完成一次真实 LoRA SFT，而非仅配置或导入测试。

- 数据：WP3 可信成功轨迹（按决策点 2 的实际规模，预期 ~10–16 条；不足则以"小批 + 多 epoch + 早停观察"方式运行）。
- 配置：LoRA（rank 8–16），目标层 q/k/v/o + mlp；Qwen2.5 系 3B 级基座，单卡 4090D 24GB 足够；混合精度 bf16；日志、tensorboard/console 指标、资源占用记录。
- 必须验证：loss 下降、checkpoint 保存（含 optimizer state）、重新加载 checkpoint 后可继续训练或推理。
- 验收：A3（SFT 部分）、A4。

### WP6 Agentic RL（GRPO）smoke（真实训练 #2）

目标：完成一次小规模 GRPO/Agentic RL 训练，验证 reward 真实进入训练、参数真实更新。

- 模式：优先 rllm `agent_trainer` / harbor_swe 同型流程（vLLM 做策略 rollout + 训练更新），将 SWE 任务环境接入；reward 用 WP1 评测同源的隐藏测试通过信号（1/0）——过程信号可用测试数量差分。
- 数据：WP3 可信轨迹初始化策略后，由训练环境重新 rollout 生成组内样本；规模小（单 batch 组内 n≈4–8），明确这是 smoke 而非正式 RL 实验。
- 必须验证：训练日志中 reward 均值随 step 有记录且非恒值、KL 与策略参数快照有变化、checkpoint 可保存加载、vLLM 与训练进程在同一 tmux 会话内协调。
- 验收：A3（GRPO 部分）、A4；若因数据/资源原因无法收敛，如实记录"系统工程可行、无提升"。

### WP7 训练前后回归评测与报告

- **T7.1 留出集**：从未进入训练/调参的任务作为同协议评测集（训练只使用部分任务数据，或使用独立留出的任务；具体划分按数据规模定，训练集/留出集互不重叠并冻结版本）。
- **T7.2 评测**：同一批任务上分别跑基座模型（未训练）与 SFT/GRPO 后模型，同协议（相同 instance 配置、相同 call 上限、相同评测器），记录成功率、patch 有效数、token/成本、耗时。
- **T7.3 报告**：`reports/` 下产出训练闭环报告：配置、日志、资源使用、checkpoint 信息、前后对照表、失败分析；数字均可追溯到原始文件。
- 验收：A5。

---

## 四、执行顺序与依赖

```
WP0（收尾 2 条，先行）
  → WP1（14 条新任务，复用 WP0 评测流程）
  → WP2（自动化流水线，基于 WP0/WP1 实测流程固化）
  → WP3（数据导出，依赖 WP1 完成 + WP2 汇总）
  → WP4（训练环境，可与 WP2/WP3 并行推进）
  → WP5（SFT）→ WP6（GRPO）→ WP7（回归评测与报告）
```

WP0–WP3 不占用 GPU（rollout 走 DeepSeek API）；WP4–WP7 为 GPU 作业，全部 tmux 启动 + 启动前 `check_gpu.sh`，使用物理 GPU 1–7 中的空闲卡。

---

## 五、决策点（已于 2026-08-08 确认）

1. **基座模型：Qwen2.5-Coder-3B-Instruct**（用户确认）。SFT 蒸馏目标与编码任务匹配；下载到 `/media/imc/data/yzy/agent/models/`。若 GRPO 阶段需要更强的 agentic 先验，可另备 Qwen2.5-3B-Instruct 作对照，但默认以 Coder-3B 为准。
2. **失败任务补 1 次 rollout**（用户确认）：20 条任务各 1 次基础上，reward 0 任务补 1 次重 rollout，为 GRPO 提供组内样本与失败轨迹；预计新增 API 成本约 $0.3。
3. **SFT 数据规模预期**：按当前 50% 严格成功率，20 条完成后预计成功轨迹 ~10–16 条、失败轨迹 ~4–10 条。这是 smoke 量级，报告中如实标注，不宣称论文级结论。
4. **训练 GPU 用量**：单卡（1 张 4090D，24GB）足够 3B LoRA SFT 与小型 GRPO；若 GRPO 需要更大组内 batch，可用 2 卡（需确认 GPU 1–7 同时空闲）。
3. **SFT 数据规模预期**：按当前 50% 严格成功率，20 条完成后预计成功轨迹 ~10–16 条、失败轨迹 ~4–10 条。这是 smoke 量级，会在报告中如实标注，不宣称论文级结论。
4. **训练 GPU 用量**：单卡（1 张 4090D，24GB）足够 3B LoRA SFT 与小型 GRPO；若 GRPO 需要更大组内 batch，可用 2 卡（需确认 GPU 1–7 同时空闲）。

---

## 六、风险与应对

| 风险 | 应对 |
|---|---|
| SWE-smith 任务与模型能力不匹配，成功率低 | 如实记录；失败轨迹本身是 GRPO 负样本；不修改任务或泄露测试 |
| 部分仓库依赖重（stackprinter/typeguard 等）导致评测容器环境失败 | 评测失败与模型失败分开登记（`eval_failure` 与 reward 0 区分），不影响完整性判定 |
| verl==0.8.0 与 torch/vllm 版本矩阵冲突 | 固定 rllm submodule 对应版本；必要时降级：SFT 用 transformers+peft，RL 用 verl 原生命令；安装问题单独记录 |
| 训练数据不足导致 loss 不降/reward 无变化 | 按验收标准如实报告"系统工程可行、无提升"，不伪造提升 |
| 网络下载模型/依赖超时 | 全部走 `/media/imc/data/yzy/agent/` 缓存与镜像，断点续传 |
| rollout 中途断连 | 全部长任务 tmux 启动；run_batch 支持断点续跑 |

---

## 七、交付物清单

1. 20/20 任务完整记录（轨迹、补丁、隐藏测试结果、reward、token、成本、完整性状态）→ 数据盘 `swesmith-pilot20/`。
2. 自动化流水线脚本 6 个 + 一键编排 → `project2-coding-agent-rl/scripts/`。
3. 可重建的 `evaluations/summary.json` + 数据导出（成功/失败/无效三类）→ 数据盘 `training-data/`。
4. 训练环境记录（安装日志、版本矩阵、冒烟日志）。
5. SFT 与 GRPO 各一次真实运行的配置、日志、checkpoint、资源使用。
6. 训练前后同协议对照结果与最终报告 → `reports/`。
7. 本 spec 执行状态回填：每完成一个 WP，更新 `docs/PROGRESS_SUMMARY` 并保留历史版本。
