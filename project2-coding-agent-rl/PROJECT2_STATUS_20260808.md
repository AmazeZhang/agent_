# 项目二（Coding Agentic RL）状况说明 — 2026-08-08

> **历史状态提示（2026-08-10）**：本文记录的是旧 3B smoke/WP0–WP7 闭环，
> 不是当前 Phase 1 的完整状态。之后已完成 7B 新路线的 Phase 1a 数据管道，并进入
> Phase 1b SFT 显存攻关。当前入口见 `../docs/PROJECT_STATUS_2026-08-10.md`，
> Phase 1a 证据见 `PROJECT2_PHASE1A_REPORT_20260808.md`。

> 本文档供下一步决策诊断使用。所有数字可追溯到各节列出的原始文件。
> 完整训练闭环报告见 `reports/WP7_TRAINING_CLOSED_LOOP_20260808.md`。

## 1. 一句话状况

**系统工程链路全部打通、验收 A1–A5 全部达成；但训练无提升——GRPO 四连全零 reward、
WP7 评测 0/15 提交，根因已逐环实证。**

## 2. 完成度（WP0–WP7）

| WP | 内容 | 状态 |
|---|---|---|
| WP0 | run8/run9 独立评测登记（r1/r0） | ✅ |
| WP1 | rollout 队列 run10–23：20 条可信，15 r1 / 5 r0（75%） | ✅ |
| WP2 | 流水线脚本（sanitize / evaluate / integrity / summary） | ✅ 实测 |
| WP3 | training-data 导出：15 成功 + 5 失败 + 3 无效隔离 | ✅ |
| WP4 | 安装栈 + GPU 冒烟（GPU 1–7，避开 GPU 0） | ✅ |
| WP5 | SFT#1（轨迹步骤）与 SFT#2（金补丁 diff）各一次真实最小训练 | ✅ |
| WP6 | GRPO smoke：run 9–12 四种 init×任务池组合，reward 全零（如实登记） | ✅ |
| WP7 | 留出评测 5 任务 × {base, sft, grpo} 同协议，0/15 提交；训练闭环报告 | ✅ |

验收：A1（20 条严格 Pilot）✅；A2（数据清洗门禁）✅；A3（SFT+GRPO 真实最小训练）✅；
A4（checkpoint 保存/加载/用于评测）✅；A5（前后同协议结果如实报告）✅。

## 3. 核心实验数字

### 3.1 数据（WP1/WP3）

- 20 条 rollout，15 严格成功（75%）。留出 5 任务：funcy-curry-compose-3u9hti2d、
  pygments-groff-0jqqr58z、stackprinter-1i9gep13、funcy-lookuper-3y0j7te5、boltons-7nlifqzn。
- 偏差：oauthlib-signature-1bsv3m8l 原在留出集，被 SFT#1 误训污染 → 替换为 funcy-lookuper。

### 3.2 SFT（WP5）

| | SFT#1（轨迹步骤） | SFT#2（金补丁 diff 格式） |
|---|---|---|
| 数据 | 60 行（3 任务工具调用步骤） | 14 行（13 任务 + oauthlib） |
| adapter | sft-gs15：lora_A 与同种子 init 差 0.044（微动） | sft2-gs3：504 张量 / 14.97 M **全部非零**（真实训练） |
| 合并模型 | models/qwen25-coder-3b-sft-merged | models/qwen25-coder-3b-sft2-merged |

### 3.3 GRPO（WP6）

| run | init | 任务池 | 结果 |
|---|---|---|---|
| 9 | 基座 | p2_swe_smoke（4 任务） | 全零（57 个 eval 目录 git apply 全失败） |
| 10 | SFT#1 热启动 | p2_swe_smoke | 全零 |
| 11 | SFT#1 热启动 | SFT seen 任务 | 全零 |
| 12 | SFT#2 热启动 | p2_swe_smoke_v2 | 全零 |

- 全零实证：`critic/rewards/mean:0.0`、`actor/pg_loss:0.0`、`grad_norm:0.0`
  （smoke-data/grpo{,-warm,-sftseen,-sft2}.log）。
- GRPO 无更新时 checkpoint = 初始化值：导出 adapter `grpo-gs1-r12` 的 lora_A 非零（Kaiming
  同种子）、**lora_B 全零**；对比 SFT#2 adapter lora_B 全非零。

### 3.4 WP7 留出评测

5 任务 × 3 变体，SWE-agent 同协议（40 调用上限、temperature 0.0、单 worker、本地 vLLM
LoRA 服务 GPU 3 端口 8011）：**15/15 未提交（exit_cost），reward 全 0**。
traj 抽查：模型全程 bash 探索 + str_replace_editor 编辑（43–78 次），从未调用 submit。

## 4. 失败根因链（逐环实证）

1. **模型产出格式**：基座 3B 无法产出可 apply 的独立 diff（run 9：57 目录全失败）。
2. **SFT#1 数据格式错配**：学的是工具调用（bash/str_replace_editor/submit），GRPO 奖励
   只接受独立 diff，必然失败（run 10/11）。
3. **SFT#2 学格式不学内容**：14 条 × 1 epoch 让 LoRA 记住 diff 骨架，输出编造补丁
   （vLLM 探针实证：hunk 位置 123/134 vs 金补丁 537/798、虚构 index 行；
   `git apply --check` 实测失败）。run 12 仍全零。
4. **评测侧能力**：3B 模型 40 步预算内完不成任务（不提交）——对照组 DeepSeek 同协议 15/20。

推论（对下一步的指导）：三环缺一不可——**模型能力 / 数据格式与奖励对齐 / 数据量**。

## 5. 可复用资产（不随实验结论失效）

- 训练链路：rllm(verl) GRPO/SFT + LoRA，tmux 管理，GPU 1–7。
- checkpoint 工程：FSDP2 DTensor → `to_local()` → PEFT adapter → vLLM `--enable-lora` 服务
  （`scripts/export_lora_adapter.py`、`scripts/merge_adapter.py`、`scripts/serve_wp7.sh`）。
- 评测：`scripts/pipeline_evaluate.sh`（补丁→eval-repo→pytest→FAIL_TO_PASS）、
  `scripts/holdout_evaluate.sh`（WP7 同协议）、5 个留出 eval-repo + venv 已就绪。
- 数据：training-data/（15 成功轨迹+金补丁、5 失败轨迹）、smoke-data/*.parquet。
- 已知坑（省下次时间）：verl auto-resume 陷阱、mini_batch 整除约束、vLLM 0.22.1 无
  Qwen2.5 tool parser（→ thought_action 文本协议）、litellm registry 模型名须与 LoRA 名一致、
  `--lora-modules` 重复键只留最后一个、ninja PATH 前置。

## 6. 偏差记录（完整版见 WP7 报告 §8）

1. oauthlib 留出集污染（SFT#1 builder 无过滤）→ 替换 + builder 加 HOLDOUT。
2. WP7 协议：function calling → thought_action（vLLM parser 不匹配），三变体一致，与
   WP1 DeepSeek 协议不同（跨模型不严格可比）。
3. thought_action 模板 `{{WINDOW}}` 渲染为空（三变体一致，已知瑕疵）。
4. run 9 曾被 verl auto-resume 从 run 8 ckpt 续跑 → 后续用全新 ckpt 目录。

## 7. 下一步决策点（诊断用）

**前提事实**：0/15 提交意味着当前评测协议下 3B 连"提交"都做不到；GRPO 恒零意味着
当前数据/格式下 RL 无梯度可学。任何路线都绕不开这两点。

| 路线 | 动作 | 成本 | 依据 | 风险 |
|---|---|---|---|---|
| A. 换大模型重跑 | 7B/14B（如 Qwen2.5-Coder-7B）+ 复用现有数据/评测，先验证"能提交" | 中（推理/显存） | 根因 1/4 是模型能力；WP1 证明 DeepSeek 级别模型可完成 | 数据量 20 条仍小，SFT 泛化存疑 |
| B. 数据扩容+格式对齐 | 新增 rollout（50–100 条）+ SFT 用与奖励一致的独立 diff 格式 + GRPO 直接监督学习热启动 | 高（rollout 成本） | 根因 2/3 是格式与数据量；SFT#2 证明格式对了能训动 | 留出任务需重建；时间长 |
| C. 验证 3B 上限 | 提高调用上限（40→100）+ 修 thought_action（function calling 或训 submit） | 低 | 根因 4 待澄清：是 40 步不够还是 3B 不会提交 | 大概率仍失败，但能收窄根因 |
| D. 承认 smoke 定位收尾 | 项目二结论封存（链路+负结果已文档化），资源转项目一 | 零 | 规格即 smoke 量级；A5 允许无提升 | 无 |
| E. 奖励信号改造 | GRPO 改多轮工具调用奖励（不只独立 diff） | 中高 | 根因 2：奖励与动作空间错配 | 需重写奖励+workflow，工作量大 |

**建议**：先走 C（1–2 天，澄清 3B 真实上限），再按结果选 A 或 B；若资源有限则 D。
文档化后所有路线都可复用 §5 资产。

## 8. 产物索引

- 代码：`project2-coding-agent-rl/scripts/`（30+ 脚本）、`configs/local-model-registry.json`
- 数据/模型（数据盘 `/media/imc/data/yzy/agent/project2/`）：`swesmith-pilot20/`、
  `training-data/`、`smoke-data/`、`checkpoints/`、`adapters/`、`models/`、
  `holdout-eval/`（15 评测记录 + 15 traj）
- 文档：`docs/PROJECT2_SPEC_20260808.md`（规格）、
  `reports/WP7_TRAINING_CLOSED_LOOP_20260808.md`（闭环报告）
