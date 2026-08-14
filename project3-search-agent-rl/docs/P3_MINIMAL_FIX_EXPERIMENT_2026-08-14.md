# P3 最小修正实验：Prompt 指令 + 格式奖励 + GRPO group size

日期：2026-08-14。基于 `P3_CONFIG_DIAGNOSIS_2026-08-14.md` 的根因判定
（prompt 无搜索协议指令 / reward 稀疏无格式分 / GRPO n=1）设计的最小修正实验。
**本实验仍使用 8 行 smoke train，目的是验证"修正后的配置能否让模型学会搜索行为"，**
不扩大数据、不追加 20 步。

## 实验变量（同时启用三项，但每项独立可审计）

| # | 变量 | 改动 | 改动点 | 依据 |
|---|---|---|---|---|
| 1 | Prompt 指令 | 问题文本前加系统指令 + 1 个 few-shot 搜索示例 | `agent_system/.../search/envs.py::_sync_reset`（训练与评测共用同一环境代码，自动同条件） | 根因 #1 |
| 2 | 格式奖励 | `compute_score` 传 `format_score=0.1` | `skyrl_gym/envs/search/env.py::_get_reward` | 根因 #2，Search-R1 原版 0.1 |
| 3 | GRPO group size | `actor_rollout_ref.rollout.n=4` | 新 wrapper override | 根因 #3，组基线恢复 |

Prompt 文案（沿用 Search-R1 风格，中文数据集同样适用）：

```text
You are an AI assistant. If you are not certain about the answer, you should
search the web first by outputting <search>query</search>, read the returned
passages, then give the final answer inside <answer>...</answer>.

Example:
Question: who wrote the song xyz?
<search>who wrote the song xyz</search>
Search results: ...
<answer>author</answer>
```

## 固定变量（与 Attempt H 完全一致）

8 行 smoke train、LoRA rank32/alpha32 all-linear、lr 3e-6、warmup 0、
kl_loss_coef 0.001、temperature 1.0、seed 0、max_steps 2、history_length 2、
topk 3、max_response 256、total_training_steps 5、物理 GPU1、真实 CPU Wiki-18
retriever、veRL commit `20bd331b…` + 补丁 0001–0004。

## 实施步骤（CPU 部分本轮完成，GPU 部分另行批准）

1. 写补丁 `patches/0004-search-prompt-and-format-reward.patch`（两个改动点），
   按 0001–0003 流程 apply 到 vendor/verl-agent；
2. 新增 wrapper `scripts/run_p3_grpo_fix_exp.sh`（复制 one_step，追加
   `actor_rollout_ref.rollout.n=4` override，其余门禁不变）；
3. CPU 测试：env 返回文本包含指令与 few-shot、格式 reward=0.1 生效、
   原 33 项相关套件不回归；
4. GPU 阶段（待批准）：预检 → 启动 retriever → 训练 5 步
   （Run ID `p3-grpo-fix-n4-prompt-fmt-s0-20260814a`）→ 检查训练 reward 曲线
   出现非零值 → 用同一 heldout-32 管道评测 Base / old Step 5 / new Step 5′
   （Run ID `p3-eval-heldout32-fix-base-s0-20260814b` 等三组）→ 对比报告。

## 预注册判据（训练 5 步后核对，全部满足才进入下一步）

| 指标 | 判据 |
|---|---|
| 训练期 reward | 出现非零值（格式奖励至少让格式合规样本有信号），step5 均值 > 0 |
| 搜索调用率（heldout-32 eval） | new Step5′ 中执行搜索的 episodes 占比 ≥ 50%（base 为 ~6/32） |
| 无效动作率 | new Step5′ < old Step5（18.4%） |
| EM 方向 | new Step5′ ≥ 2/32 且与 base 的不一致对方向为正；任何"明确提升"必须先 vLLM 原生复核 |

**失败判据**：训练 reward 仍全 0 → 检查补丁/配置并停止；行为指标无改善 →
按预注册顺序调 lr（1e-5）、rollout.n（8）、epochs（10）、max_response（512）
形成第二轮最小实验；仍无信号才考虑扩大数据规模。任何情况下不追加 20 步。

## 边界声明

- 8 行 smoke 上只验证"行为是否被学会"，不作为质量声明；行为指标通过后才谈
  扩大数据/步数；
- HF greedy 评测与训练 vLLM rollout 的 backend 差异依旧适用，提升需原生复核；
- 补丁 0004 同时改变训练与评测的 prompt 呈现，三组对比仍同条件（同一环境代码）。
