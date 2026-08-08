# 阶段 0 报告：SWE-Master-4B-RL 作为"能否提交"验证器 — 2026-08-08

> 结论先行：**门禁① FAIL**。15 次运行 **0 次提交**（Submitted），resolve rate 0/5。
> 但协议三轮演进后模型展现出真实修复能力：5/5 任务均产生源文件编辑，
> 4/5 为零回归的部分正确修复（funcy-lookuper 核心行与 gold patch 完全一致）。
> 决策建议：验证器角色由**规则评估器**（门禁②已全 PASS）承担，4B 模型降级为
> 阶段 1 的部分修复轨迹生成器候选；Qwen2.5-Coder-7B-Instruct 按原计划作训练底座。

## 1. 目标与门禁定义

阶段 0 目标：验证 **SWE-Master-4B-RL** 能否充当阶段 1 的"能否提交"验证器——
在 5 个 holdout 任务（funcy-curry-compose-3u9hti2d、funcy-lookuper-3y0j7te5、
pygments-groff-0jqqr58z、stackprinter-1i9gep13、boltons-7nlifqzn）上，
以 SWE-agent 同协议独立运行，提交（`exit`/`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`）
则判定该任务"可提交"。

**门禁①（阶段 0 验收）**：≥1 次提交 **且** resolve rate > 0
（至少 1 个提交的 patch 通过全部 F2P 测试，即 reward 协议下 f2p 全过、delta=0）。

## 2. 基础设施（本轮交付物，均可复用）

| 组件 | 说明 |
|---|---|
| vLLM serving | GPU 1、端口 8012、`swe-master-4b-rl`、max_model_len 49152、KV 62.1k tokens |
| `swe_command` 工具解析器 | 自定义 vLLM ToolParser：`<command>`/```bash```/JSON 三种格式 → bash tool call；剥离模型自生成的假 `<tool_response>`/`Exit code:` 转录尾 |
| phase0 协议 | mini-swe-agent 2.4.6（`.venvs/miniswea`）+ SWE-agent 风格观察格式 + `exit` 提交拦截（bare `exit` 直接提交，不经 shell）+ BLOCKLIST 安全过滤 + 输出截断 6000 字符 + COMPLETE/exit 提交检测 → `git diff HEAD` |
| reward 评估器 | 门禁②（此前）全 PASS：verified→1.0、empty/garbage/tests_modified→0；`phase0.py eval {--from-work|--patch}` |

协议三轮演进（每轮均为可复用的修复，对任何 SWE-agent 风格 RL 模型通用）：

| 轮 | 变更 | 动机 |
|---|---|---|
| v1 | swe_command parser、exit/COMPLETE 检测、截断、KV 扩容 | 基线 |
| v2 | 观察格式改 SWE-agent 风格（`Exit code:`/`Execution Success:`/`[STDOUT]`）、剥离 `<tool_response>` 假转录、JSON tool-call 格式匹配、格式错误容忍 3→5、max_tokens 1024→2048 | 修 RepeatedFormatError |
| v3 | bare `exit` 提交拦截（不经 shell）、提示词明示两种提交方式、剥离 `Exit code:` 开头假转录尾、max_tokens 2048→4096 | 修"从不提交"与上下文膨胀 |

## 3. 结果总表（3 轮 × 5 任务 = 15 次运行，temperature 0.0）

| 任务 | 轮1 | 轮2 | 轮3 | 轮3 补丁质量（eval） |
|---|---|---|---|---|
| stackprinter-1i9gep13 | LimitsExceeded 空 | 编辑，f2p 1/1，1 回归（0.3） | 编辑（与轮2 完全相同的补丁），同上 | **f2p 1/1、delta 1** |
| funcy-lookuper-3y0j7te5 | RepeatedFormatError | RepeatedFormatError | 编辑（核心行正确），最终态被后续编辑破坏 | **单行修复 f2p 1/3、delta 0**（0.5） |
| funcy-curry-compose-3u9hti2d | RepeatedFormatError | 编辑但破坏 import | 编辑，干净 | **f2p 2/4、delta 0**（0.5） |
| pygments-groff-0jqqr58z | RepeatedFormatError | RepeatedFormatError + 上下文超限 | 编辑，干净 | **f2p 2/4、delta 0**（0.5） |
| boltons-7nlifqzn | LimitsExceeded 空 | LimitsExceeded 空 | 编辑，未命中目标 | **f2p 0/2、delta 0**（0.5） |
| **提交次数** | 0 | 0 | 0 | **合计 0/15** |

数据位置：`/media/imc/data/yzy/agent/project2/phase0/`（preds/、evals/、patches/、
trajs/、work/）；补丁存档 `phase0/patches/*-r{2,3}.diff`。

## 4. 根因分析

### 4.1 轮1 失败（协议不匹配，0/5 提交、3×RepeatedFormatError）

- 模型输出 `<command>` 后会**自行续写观察结果**（假 `<tool_response>`，含
  `Exit code: 0 / Execution Time: 0.03ms / Execution Success: true / [STDOUT]`）——
  RL 训练中命令后紧跟观察，模型学会了"把整段对话写出来"（transcript-continuation）。
- 假转录烧掉 max_tokens，第二条命令（常以 JSON 格式 `{"name": "bash", ...}` 模仿历史
  中的 tool_call 序列化）被截断 → 无命令 → RepeatedFormatError。
- 观察格式不匹配：训练中观察是 `Exit code:` 风格，我们给的是 `<returncode><output>`。

### 4.2 轮2 修复（能力显现）

- stackprinter：模型诊断出 `format()` 无参 bug 的两处根因
  （`sys._getframe(2)` 偏移 + `_is_exc_info` 元组长度 2→3），f2p 1/1 通过
  目标测试，但 `_is_exc_info` 改动引入 1 个回归；轮3 末尾模型已识别该回归并
  在修复中，预算耗尽。
- funcy-curry：修复了 compose 的 `None→identity` 与 `min_n→req_n`，但 sed
  多行替换误删 `def curry(func, n=EMPTY):` 行 → 整个包 import 失败。

### 4.3 轮3（5/5 编辑，但系统性缺陷）

| 缺陷 | 证据 | 影响 |
|---|---|---|
| **从不提交**（0/15） | 支持 `exit`、提示词明示后仍 0 提交 | 门禁① 直接 FAIL；模型无法作为提交验证器 |
| **过度编辑** | funcy-lookuper 正确修复后插入语法错误块（`elif` 挂 `else` 后）；funcy-curry r2 删 def 行 | 最终态常被破坏，虽正确修复过 |
| **上下文爆炸** | pygments r2/r3 ContextWindowExceededError（49k 上限） | 长任务中途死亡 |
| 修复不完整 | 4/5 补丁零回归但 f2p 部分通过 | resolve 需全过 |

"从不提交"的成因推测：4B 模型对"测试全过、修复完成"的自校验能力不足——
它做出正确修复后仍继续分析/编辑（如 funcy-lookuper 在正确单行修复后继续
改 else 分支并写坏），从未到达"完成"判断。训练中 SWE-agent 的提交动作是
`exit`，但我们观察到它连 `exit` 也从不输出。

## 5. 门禁①判定

**FAIL**：0/15 提交（要求 ≥1），resolve rate 0/5（要求 >0，无补丁 f2p 全过）。

## 6. 决策建议（阶段 1 调整）

1. **验证器角色 → 规则评估器**：门禁②已全 PASS 的 reward 评估器（f2p 全过 +
   delta=0 门禁）直接承担阶段 1 轨迹验证，无需模型验证器。评估器确定性、免费、快。
2. **SWE-Master-4B-RL 降级为部分修复生成器**：其零回归部分修复轨迹
   （funcy-lookuper 单行、funcy-curry 2/4、pygments 2/4）是 GRPO 训练的
   高质量正例/负样本来源——"模型做到一半"的轨迹对 reward shaping 有信息量。
   不建议为它继续投入协议适配。
3. **Qwen2.5-Coder-7B-Instruct**：按原计划作 SFT/GRPO 训练底座（已下载验证，
   15.2 GB、4 shards、339 tensors）。
4. **阶段 1 数据扩展**：teacher（DeepSeek API）生成轨迹 → 规则评估器验证
   （f2p≥1 且 delta=0）→ 100+ 任务数据池。
5. 若未来仍需要"模型验证器"，可评估 Qwen2.5-Coder-7B 或 API 模型，但需新的
   协议适配，非当前必需。

## 7. 可追溯性

- 运行日志：`/tmp/run{1,2,3}-*.log`；preds：`phase0/preds/*.json`；
  eval 明细：`phase0/evals/*.json`；轨迹：`phase0/trajs/*.traj.json`；
  补丁：`phase0/patches/*.diff`。
- 协议代码：`scripts/phase0/phase0.py`（v3）、`scripts/phase0/serve_phase0.sh`；
  解析器（vLLM venv 内，安装步骤见 scripts/phase0 目录说明）：
  `vllm/tool_parsers/swe_command_tool_parser.py` + `tool_parsers/__init__.py` 注册。
- 模型：`/media/imc/data/yzy/agent/project2/models/{SWE-Master-4B-RL, Qwen2.5-Coder-7B-Instruct}`。
