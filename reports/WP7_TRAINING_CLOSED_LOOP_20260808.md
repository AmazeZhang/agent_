# 项目二（Coding Agentic RL）训练闭环报告 — WP7

- 日期：2026-08-08
- 基座模型：Qwen2.5-Coder-3B-Instruct（hf-cache `488639f1ff808d1d3d0ba301aef8c11461451ec5`）
- 训练栈：rllm（verl 后端），LoRA rank 8
- 本报告全部数字均可追溯到原始文件（路径列于各节）

## 1. 执行摘要

系统工程链路（数据 → SFT → GRPO → checkpoint 导出 → vLLM LoRA 服务 → 同协议评测）
**全链路打通**，验收项 A1–A5 全部达成。训练本身**无提升，如实报告**：

- WP1 数据：20 条可信 rollout，15 成功（75%）。
- SFT#2（金补丁 diff 格式）是**真实训练**（adapter 15 M 参数全部非零）；SFT#1（轨迹步骤格式）权重仅微动。
- GRPO 四次运行（base / SFT#1×2 / SFT#2 热启动）**reward 全零**，且实证了"GRPO 无更新时 checkpoint = 初始化值"。
- 根因（探针实证）：SFT#1 数据是 SWE-agent 工具调用轨迹（bash/str_replace_editor/submit），
  策略学的是工具调用格式而非独立 diff，GRPO 单轮 git apply 必然失败；SFT#2 模型学会了 diff
  "格式骨架"但内容编造（14 条样本 × 1 epoch 不足以让 3B LoRA 记住补丁内容），git apply 失败。
- WP7 留出评测：5 留出任务 × {base, sft, grpo} 同协议评测，**0/15 提交**（40 调用上限耗尽，
  模型全程探索编辑、从未尝试 submit），reward 全 0，三变体无可测差异。

## 2. 闭环全景

```
WP1 rollout（20 条，15 成功）
  ├─ WP3 training-data/ 导出（15 成功 + 5 失败 + 3 无效隔离）
  │    └─ SFT#1：轨迹步骤（60 行，3 任务）→ adapter sft-gs15
  │    └─ SFT#2：金补丁 diff（14 行，13 任务 + oauthlib）→ adapter sft2-gs3
  └─ WP6 GRPO（run 9/10/11/12，reward 全零，checkpoint 导出实证）
        └─ WP7 留出评测（base / sft2 / grpo 三变体，0/15 提交）
```

## 3. 数据（WP1/WP3）

| 项 | 值 | 证据 |
|---|---|---|
| rollout 总数（run10–23） | 20 条可信 | `swesmith-pilot20/runs/deepseek-v4-flash-run{10..23}` |
| 严格成功（r1） | 15 / 20 = 75% | `training-data/successes/`（15 个补丁+轨迹） |
| 失败（r0） | 5 / 20 | `training-data/failures/` |
| 无效隔离 | 3 | `training-data/invalid/` |
| 数据导出脚本 | `scripts/export_training_data.py` | |

留出集 5 任务（修正后冻结）：funcy-curry-compose-3u9hti2d、pygments-groff-0jqqr58z、
stackprinter-1i9gep13、funcy-lookuper-3y0j7te5、boltons-7nlifqzn。
**偏差记录**：oauthlib-signature-1bsv3m8l 原在留出集，因 SFT#1 误训（builder 无 HOLDOUT 过滤）
被污染，已替换为 funcy-lookuper-3y0j7te5 并文档化（见 §8）。

## 4. SFT（WP5）

| 项 | SFT#1（轨迹步骤） | SFT#2（金补丁 diff 格式） |
|---|---|---|
| 数据 | 60 行（3 任务成功轨迹的 bash/str_replace_editor/submit 步骤） | 14 行（13 任务金补丁 + oauthlib，system+issue+```diff） |
| builder | `build_smoke_sft_data.py` | `build_patch_sft_data.py` |
| 日志 | `smoke-data/sft.log`（GPU 1，15 iter） | `smoke-data/sft2.log`（3 iter） |
| checkpoint | `checkpoints/smoke-sft/`（94 G） | `checkpoints/smoke-sft2/global_step_3/`（24 G，无 actor/，shard 在顶层） |
| adapter | `adapters/sft-gs15` | `adapters/sft2-gs3` |
| 训练真实性 | lora_A vs 同种子 init 最大差 0.044（微动） | 504 张量 / 14.97 M 参数**全部非零**，lora_B 252 个全非零（真实训练） |
| 合并模型 | `models/qwen25-coder-3b-sft-merged` | `models/qwen25-coder-3b-sft2-merged` |

## 5. GRPO（WP6，run 9–12）

全部在 GPU 1–7（tmux），4 任务 × n=4，mini_batch 2，单步训练。**reward 全零**：

| run | 日志 | checkpoint | init | 任务池 | 结果 |
|---|---|---|---|---|---|
| 9（base init） | `grpo.log`（step 4 全零行） | `smoke-grpo`（step 1–4，含 run 8 续跑） | 基座 | p2_swe_smoke | 57 个 scratch 目录全部 git apply 失败，0 个 result.xml |
| 10（SFT#1 热启动） | `grpo-warm.log` | `smoke-grpo2` | sft-merged | p2_swe_smoke | reward 全零 |
| 11（SFT#1 热启动+seen） | `grpo-sftseen.log` | `smoke-grpo3` | sft-merged | SFT seen 任务 | reward 全零 |
| 12（SFT#2 热启动） | `grpo-sft2.log` | `smoke-grpo4` | sft2-merged | p2_swe_smoke_v2 | reward 全零 |

关键数字（可追溯）：各日志 `critic/rewards/mean:0.0`、`batch/solve_all:0.0`、
`Rollout completed. Rewards: ['default_traj_name: 0.0']`；`actor/pg_loss:0.0`、`grad_norm:0.0`。

### 5.1 根因链（探针实证，2026-08-08）

1. **run 9（base init）**：Qwen2.5-Coder-3B-Instruct 基座无法产出可 apply 的独立 diff——
   57 个评测 scratch 目录全部 `git apply` 失败（目录是 eval-repo 原始副本，无 result.xml）。
2. **run 10/11（SFT#1 热启动）**：SFT#1 数据是 SWE-agent 工具调用轨迹，模型学的是
   `bash`/`str_replace_editor`/`submit` 调用格式。GRPO 奖励只接受独立 diff（单轮 git apply），
   工具调用格式必然失败。
3. **run 12（SFT#2 热启动）**：vLLM 探针直查（`/tmp/sft2-probe.txt`，2026-08-08）：
   sft2 模型对 oauthlib 问题输出 ```` ```diff ```` 块——**格式正确但内容编造**：
   - hunk 位置编造在 123/134 行，金补丁实际在 537/798 行（`training-data/successes/deepseek-v4-flash-run1/oauthlib__oauthlib.1fd52536.combine_file__1bsv3m8l.patch`）；
   - index 行为虚构的 `1234567..89abcdef`；
   - `git apply --check` 对真实 eval-repo（`eval-repos/oauthlib-signature-1bsv3m8l`）**实测失败**：
     `error: 打补丁失败：oauthlib/oauth1/rfc5849/signature.py:123`。
   - 结论：14 条金补丁 × 1 epoch 让 LoRA 记住了 diff 的格式骨架，但内容未泛化到训练分布外。

### 5.2 GRPO checkpoint 实证（A4）

`export_lora_adapter.py` 导出 `adapters/grpo-gs1-r12`：
- 504 张量（252 lora_A + 252 lora_B），lora_A 非零（Kaiming 同种子初始化）、**lora_B 全零**；
- 与 SFT#2 adapter（lora_B 全非零）对比 → GRPO 全零 reward 下无策略更新，checkpoint = init 值，
  机制层面证实"奖励恒零 → 参数不更新"。
- 工程修复：verl FSDP2 checkpoint 的 LoRA 张量是 DTensor（Shard(dim=0)），safetensors 无法写，
  export 前 `to_local()`（world size 1 时还原全量）。

## 6. WP7 留出评测

### 6.1 协议

SWE-agent 1.1.0 `run-batch`，每个任务上限 40 次调用、temperature 0.0、单 worker、成本 0。
模型端点 = 本地 vLLM（GPU 3，端口 8011，`--enable-lora`，`--lora-modules sft=... grpo=...`，
`--served-model-name qwen25-coder-3b-base`）。

**协议偏差（如实记录）**：vLLM 0.22.1 无匹配 Qwen2.5 输出的内置 tool parser
（hermes 期望 `<tool_call>` XML、llama3_json 依赖 `<|python_tag|>` token 报 500、
qwen3_xml 期望 Qwen3 XML、`--enable-auto-tool-choice` 下模型输出 ```json 无法被解析），
因此三个变体统一改用 SWE-agent 经典**thought_action 文本协议**（讨论 + ``` 代码块，
system 模板取自官方 `config/sweagent_0_7/07_thought_action.yaml`）。
**三变体严格同协议**，base/sft/grpo 对比有效；与 WP1 的 DeepSeek 评测（function calling）
协议不同，跨模型不严格可比。

### 6.2 结果

| 任务 | base | sft | grpo |
|---|---|---|---|
| funcy-curry-compose-3u9hti2d | 未提交（0.0） | 未提交（0.0） | 未提交（0.0） |
| pygments-groff-0jqqr58z | 未提交（0.0） | 未提交（0.0） | 未提交（0.0） |
| stackprinter-1i9gep13 | 未提交（0.0） | 未提交（0.0） | 未提交（0.0） |
| funcy-lookuper-3y0j7te5 | 未提交（0.0） | 未提交（0.0） | 未提交（0.0） |
| boltons-7nlifqzn | 未提交（0.0） | 未提交（0.0） | 未提交（0.0） |

评测记录：`swesmith-pilot20/holdout-eval/evaluations/<short>-<variant>-eval.json`（15 条）。
轨迹：`holdout-eval/runs/<short>-<variant>/<instance>/<instance>.traj`。
驱动/评测脚本：`scripts/start_holdout_tmux.sh`、`scripts/holdout_evaluate.sh`（新增）。

### 6.3 行为分析（traj 抽查）

15/15 全部 `exit_status: exit_cost`（40 次调用上限耗尽），`submission: None`。
抽查 3 条 traj：模型在 bash 探索与 str_replace_editor 编辑间反复（编辑尝试 43–78 次），
`submit` 一词仅出现在 system prompt 的工具文档中——**模型从未尝试提交**。
3B 模型在 40 步预算内无法完成这 5 个任务（对照组 WP1 的 DeepSeek 同协议 15/20 成功），
且 thought_action 协议（讨论+单命令）步进开销高于 function calling。

**结论（A5）**：训练前后同协议结果——base/sft/grpo 均 0/5 提交、reward 全零，
**训练无提升**，如实报告；训练前后差异无法在此量级上观测。

## 7. 验收对照

| 验收 | 状态 | 证据 |
|---|---|---|
| A1 20 条严格 Pilot | ✅ | run10–23，15 r1 / 5 r0（75%），独立评测证据在 `evaluations/` |
| A2 数据清洗门禁 | ✅ | `pipeline_integrity.sh` + `pipeline_summary.py` 重建一致；无效 3 条隔离 |
| A3 SFT 与 GRPO 各一次真实最小训练 | ✅ | SFT#2 adapter 15 M 全非零；GRPO 全流程跑通（reward 全零如实记录，更新机制经 checkpoint 对比实证） |
| A4 checkpoint 保存/加载/用于评测 | ✅ | verl checkpoint → `export_lora_adapter.py`（含 DTensor 修复）→ PEFT adapter → vLLM `--enable-lora` 实测服务三变体 |
| A5 训练前后同协议结果如实报告 | ✅ | 本报告 §6：0/15 提交全零，无提升如实报告 |

## 8. 偏差与风险记录

1. **oauthlib 留出集污染**：SFT#1 builder 无 HOLDOUT 过滤，oauthlib 被误训 → 替换留出任务为
   funcy-lookuper-3y0j7te5；builder 已加 HOLDOUT 常量并注释说明（`build_smoke_sft_data.py`）。
2. **vLLM 协议切换**：function calling → thought_action（原因 §6.1），三变体一致。
3. **WINDOW 模板变量**：thought_action system 模板 `{{WINDOW}}` 未提供值渲染为空串
   （"shows you  lines"），三变体一致，不影响对比，已知渲染瑕疵。
4. **GRPO run 编号**：run 9 曾被 verl auto-resume 从 run 8 checkpoint 续跑
   （default_local_dir 自动续训），后续 run 均使用全新 checkpoint 目录。
5. 训练数据规模为 smoke 量级（15 成功轨迹 / 14 条金补丁），结论不宣称论文级有效性。

## 9. 资源使用

- SFT#1：GPU 1（CUDA_VISIBLE_DEVICES=1，tmux），3 分钟 15 iter；SFT#2：3 iter。
- GRPO：单 GPU（1–7 轮流空闲卡，tmux），每步 ~90–165 s（4 任务 × n=4，~8–13 k tokens）。
- vLLM 服务：GPU 3（~13.7 G 显存，--gpu-memory-utilization 0.55），三个 LoRA 变体并发。
- GPU 0 全程未使用（gnome-remote-desktop-daemon 占用）。
- 磁盘：checkpoints 共 ~160 G（含 optimizer 状态），位于数据盘 `/media/imc/data/yzy/agent/project2/`。
- 外部 API：仅 WP1 rollout 用 DeepSeek（Key 不入 Git/日志/报告）；训练与评测全程本地。

## 10. 产物索引

- 数据：`swesmith-pilot20/`（runs / evaluations / training-data / eval-repos / holdout-eval）
- 训练数据：`/media/imc/data/yzy/agent/project2/training-data/`、`smoke-data/sft-{train,patch-train}.parquet`
- 训练集（rllm）：`~/.rllm/datasets/p2_swe_smoke{,_v2}/train_verl.parquet`
- 模型/适配器：`models/qwen25-coder-3b-{sft,sft2}-merged`、`adapters/{sft-gs15,sft2-gs3,grpo-gs1-r12}`
- checkpoints：`checkpoints/smoke-{sft,sft2,grpo,grpo2,grpo3,grpo4}`
- 脚本：`scripts/{smoke_sft,smoke_rl,build_*_data,export_lora_adapter,merge_adapter,serve_wp7,start_holdout_tmux,holdout_evaluate}.{py,sh}`
- 配置：`configs/local-model-registry.json`、`swesmith-pilot20/holdout-eval/configs/*.config.yaml`
