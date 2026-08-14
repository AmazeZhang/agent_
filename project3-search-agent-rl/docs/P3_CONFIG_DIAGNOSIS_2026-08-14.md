# P3 训练配置根因排查：为什么 Step 5 没有学会搜索

日期：2026-08-14。范围：只读排查（配置、reward 代码、训练 rollout 审计、评测证据），
未修改任何代码/配置，未占用 GPU。

## 证据链

### 1. Prompt 里没有任何搜索协议指令

`agent_system/environments/env_package/search/envs.py` 的 `_sync_reset`：

```python
obs = kwargs["question"]          # 模型第一轮收到的就是原始问题文本
```

模型（Qwen2.5-1.5B-Instruct）没有系统指令、没有工具格式说明、没有 few-shot 示例，
只收到一行问题。它必须从零"发现" `<search>` 协议。评测中三个模型都呈"直接作答"模式
（heldout-32 未搜索占比 base 28/32、step2 24/32、step5 26/32）与此直接对应：
贪心解码下模型走参数记忆路径，从不检索。

### 2. Reward 只有最终严格 EM，稀疏且无中间信号

`verl/utils/reward_score/search_r1_like_qa_em.py::compute_score`：
- 无 `<answer>` → 返回 0；答案 EM 命中 → 1.0；否则返回 `format_score`（默认 0.0）
- skyrl env 调用 `compute_score(chat_history_str, ground_truth)`，**未传 format_score** →
  格式奖励恒为 0；中间 step 无 reward（`_get_reward` 仅 done 时评分）

后果：训练期 rollout audit（Attempt H epoch 3–5）24/24 条 **reward 全为 0**，
没有任何正信号可以学习。

### 3. GRPO 组内 reward 全零 → advantage 恒为零，只剩 KL

本 fork（verl+env）对 `actor_rollout_ref.rollout.n` 有硬断言（
`verl/trainer/main_ppo.py:173`：`n==1`，"achieve GRPO by env.rollout.n"）；GRPO
组大小实际由 `agent_system/environments/env_manager.py:609` 读取
`config.env.rollout.n` 作为 `group_n`。Attempt H 配置 `env.rollout.n=2` →
组大小 2（**并非 1**）。

`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage`（组大小 >1 时）：

```python
id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
id2std[idx] = torch.std(...)
scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + 1e-6)
```

当组内 reward 全 0（epoch 3–5 实况）：mean=0、std=0 → **advantage=(0−0)/(0+1e−6)=0**，
所有 advantage 恒为零，策略更新只剩 KL 正则项。诊断结论（无 RL 信号）与 n=1 版
一致，但机制是"组内全零 reward 归一化后为零"，而非单样本组退化。修复杠杆不变：
组内需要有非零 reward（format_score）且组大小合理（≥4）。

### 4. 优化预算极小

8 行 smoke train、`shuffle=false`、每 epoch 1 batch、共 5 次更新、lr=3e-6、
warmup=0、`max_response_length=256`、训练采样 temperature=1.0。

### 5. 评测观测的印证

- 训练期（温度 1.0）rollout 中模型**会**输出 `<search>`（epoch 5：9/24 条）和
  `<answer>`（15/24），但 eval 贪心（温度 0）下几乎不发 —— 格式在采样分布里存在，
  但未被奖励强化、未成为贪心模式；无效动作（混合/重复标签）在训练后期增多
  （epoch 5 约 46% step 无效；eval 中 step2/5 各 5 例 vs base 1 例），是温度 1.0
  采样格式漂移的痕迹。
- heldout-32：base 0/32、step2 1/32（参数记忆命中）、step5 0/32；McNemar 全 p=1.0。

## 根因判定（按贡献排序）

| # | 根因 | 证据 | 对应修正杠杆 |
|---|---|---|---|
| 1 | **Prompt 无搜索协议指令/few-shot** | `obs = kwargs["question"]`；三模型贪心几乎不搜索 | 系统指令 + few-shot |
| 2 | **Reward 稀疏且无格式奖励** | epoch 3–5 全 0；format_score=0.0 未启用 | format_score=0.1（Search-R1 设定） |
| 3 | **GRPO 组内 reward 全零 → advantage 恒零** | 组大小 2（env.rollout.n=2）+ epoch 3–5 全 0 reward → mean=0/std=0；fork 硬断言 actor_rollout_ref.rollout.n==1（组在 env 侧） | env.rollout.n=4 + format_score |
| 4 | 优化预算极小（8 行 × 5 步 × 3e-6） | hydra/命令审计 | 暂不改，隔离变量 |

判定路线：先做"最小修正实验"（1+2+3 组合），在 smoke 规模验证搜索行为是否被学会，
再决定是否扩大数据/步数。对应决策：**不追加 20 步**，不直接放大数据。

## 参考：Search-R1 原版设置（对照）

- prompt 含搜索工具说明与搜索调用 few-shot 示例；
- reward = 格式奖励（0.1）+ EM（1.0）；
- GRPO group size 一般 ≥ 4。
