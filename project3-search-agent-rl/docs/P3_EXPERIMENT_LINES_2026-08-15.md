# P3 实验线拆分：官方宽松语义（Search-R1 复现基线） vs 严格 fork 实验线

**日期**：2026-08-15
**目的**：把"复现 Search-R1（论文口径）"与"本 fork 的严格语义改进"明确拆成两条
独立的实验线，所有训练 run、评测、结果档案按线归档，任何声明都必须先声明
所在线。这是 EXPERIMENT_AUDIT.md Required Action 4（官方宽松语义基线）的
文档化组织。

## 1. 两条线的语义定义（已从代码核验）

| 维度 | 官方宽松语义线（Search-R1 论文口径） | 严格 fork 实验线（本项目） |
|---|---|---|
| 动作解析 | 宽松：`<(search|answer)>(.*?)</\1>`（IGNORECASE），取**第一个**完整块；无闭合标签等一律走 env 分支 | `search_projection`（`agent_system/.../search/projection.py`）：先裁剪到第一个 `</search>`/`</answer>`，再取第一个 `<search>` 块（无则 `<answer>` 块，再无则空串）；混合标签/重复标签 → `valids=0` |
| env 所见动作 | **raw action** 直接进 env；`_parse_action` 只查 `<search>…</search>`；无查询 → tool 异常 → 观察文本为错误信息，模型**重试** | **只看到投影后的动作**（`SearchEnvironmentManager.step` 先投影再执行，env 从不接触 raw action） |
| invalid 处理 | 环境提示重试，**无惩罚** | 投影 `valids=0` 被记录（`action_quality`），训练侧 `apply_invalid_action_penalty` 每 invalid 行 **-0.1** |
| 终止/重试 | `<answer>` 闭合即终局（skyrl `_is_done`） | 相同（投影后的 `<answer>` 块触发） |
| 终局奖励 | skyrl `compute_score(chat_history, gt, format_score=0.1)`：EM 命中 1.0 / 格式正确但答错 0.1 / 无答案 0.0 | 相同（patch 0004 把 env reward 对齐到官方 format_score=0.1）；训练侧额外 -0.1/invalid 行 |
| 评测主指标 | EM（env reward ≥ 1.0） | EM（env reward ≥ 1.0，**不含** format 0.1 与 invalid 惩罚——`run_p3_eval_vllm.py`/`run_p3_eval_heldout.py` 的 reward 就是 env reward） |

**核验证据**：`scripts/analyze_p3_action_reward_diag.py`（377/377 训练行精确重建：
`recorded score == episode_rewards - 0.1 × invalid`）；`projection.py`（本仓库，
严格线）；`third_party/skyrl_gym/envs/search/env.py`（官方 env 核心，未经投影）。

## 2. 每条线所属的现有产物

**严格 fork 线**（当前所有训练与评测）：
- 训练：`p3-grpo-fix-train64-nqh-n4-prompt-fmt-s0-20260814a`（global_step_8 LoRA = train64nqh8）等全部 GRPO run；`run_p3_grpo_fix_exp.sh` + patch 0001–0004
- 评测：dev32（heldout-32）vLLM 评测、confirm-256 预注册配对比较（`P3_CONFIRM256_PREREG_2026-08-15.md`）
- 结果：`analysis/` 下所有对比表

**官方宽松语义线**（论文复现基线，尚未运行）：
- 需要：一个"宽松"评测入口——env 直接接收 raw action、无投影有效性标记、
  无 invalid 惩罚（env 核心就是 skyrl SearchEnv，现状只差评测入口的投影层）
- 用途：对照 Search-R1 论文数字；回答"本 fork 的严格语义是否带来净收益"；
  任何"复现了 Search-R1"的声明只能出自这条线

## 3. 组织约定（从 2026-08-15 起）

1. **运行命名**：run id 增加 `-line-<official|strict>` 段（如
   `p3-eval-...-line-official-...` / 现有 run 默认 strict，无需改名）；
2. **结果归档**：`analysis/` 下按线分目录（`analysis/official-line/`、
   `analysis/strict-line/`），对比表必须标注所在线；
3. **评测脚本**：`run_p3_eval_vllm.py`/`run_p3_eval_heldout.py` 保持严格线
   不变；官方线评测以独立脚本（`run_p3_eval_vllm_official.py`，待建）实现
   宽松语义，**不允许**在严格线脚本里加开关混用；
4. **声明纪律**：写"优于/差于 Base"必须声明线；写"复现 Search-R1"只能引用
   官方线数字。

## 4. 状态与边界

- 本 session 的 confirm-256 预注册比较属于**严格线**；其结论的措辞边界已在
  `P3_CONFIRM256_PREREG_2026-08-15.md` §7 写明（"不声称 Search-R1 复现"）。
- 官方线基线评测（Base / train64nqh8 的宽松语义变体）是后续门禁批准后的
  独立工作项，不阻塞 confirm-256 分析。
