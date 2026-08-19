# Upstream patches

对`vendor/verl-agent`的算法修改应保留为可审查Patch或独立Commit，并记录：

- 上游submodule commit；
- 修改的Reward、Advantage或Loss公式；
- 对应配置与随机种子；
- 单元测试和Smoke结果；
- 与未修改上游基线的差异。

不要直接提交模型、数据、索引、Checkpoint或实验缓存。

## 0007-search-aware-step-reward.patch（2026-08-19，Phase 4B）

Search-aware GRPO v1 step-attributed reward（冻结公式与语义见
`docs/P3_PHASE4_SEARCH_CAUSAL_DIAG_RESULT_2026-08-19.md` §9）：

- **公式**：`R = R_answer + 0.15·evidence_hit + 0.30·sce − 0.20·invalid_or_error − 0.45·redundant_search_count − 0.20·new_answer_leak_in_query`；format_score=0.1；α=0
- **改动文件**：
  - `skyrl_gym/envs/search/env.py`：env 在 step 时计算每步 v1 shaping 分量（`search_v1` metadata；evidence_hit 只检查真实检索返回的 document 正文；answer-leak 规则含 question 排除与 alias 长度阈值；`env.search_aware_step_reward=false` 默认关闭、行为不变）
  - `multi_turn_rollout/rollout_loop.py`：`search_v1` 透传到 non_tensor_batch
  - `reward_manager/episode.py`：v1 模式 step 归因放置（R_answer 只在终止 step；shaping 放在对应 search step；sce 在终止 step 经 episode metadata 结算）+ 8 分量记录 + 逐 uid 分量和==放置和断言（整数分）
  - `verl/trainer/main_ppo.py`：`reward_model.search_aware_step_reward` 配置透传（与 `env.search_aware_step_reward` 必须同时开启，manager fail-closed）
- **配套（项目侧，不在 patch 内）**：`searchr1_repro/search_v1_reward.py`（纯函数单实现源）、`searchr1_repro/training_audit.py`（audit 增加 `search_v1`/`search_v1_episode`/`record_score`）、`scripts/run_p3_grpo_search_aware_v1.sh`（独立配置）、`tests/test_search_v1_reward.py`（12 条 CPU 测试）、`scripts/p3_v1_reward_replay.py`（历史回放 5 条硬门禁）
- **验证**：patch 在 pre-0007 状态干净应用；reverse-check 通过；12 条 CPU 测试 + 历史回放门禁（见 gates/）

## 0008-v1-trajectory-return-and-traj-audit.patch（2026-08-19，Phase 4B.1）

Phase 4B.1 训练语义修正（保留 0007 审计历史；构建在 0001-0007 之上）：

- **GRPO trajectory-return advantage（修正 per-record 归一化 bug）**：`algorithm.search_v1_trajectory_return=true`（默认 false，official-loose 走原实现）时，`compute_grpo_trajectory_return_advantage` 先按 record 求和、再按 traj_uid 汇总 trajectory return、在 uid 内对 5 个 trajectory return 做 GRPO mean/std 归一、把 trajectory advantage 广播回该轨迹的所有 model-action record（Observation token 保持 loss mask=0）；单个 traj_uid 映射到多个 uid 时 fail-closed 抛错
- **episode 审计按 traj_uid**：`search_v1_episode` 记录每条真实 trajectory（同一 uid 的 5 个 traj_uid 各自独立 total），`search_v1_group` 为按 uid 的信息性汇总（不替代 episode 审计）；分量和==放置和断言保留
- **question 真实透传**：`SearchMultiProcessEnv._sync_reset` 的 extras 增加 `"question"`（此前 SearchEnv.question 恒为 None，answer-leak 规则永远针对空 question 判定）；`env.search_aware_step_reward` 顶层开关传播到 per-env SearchEnv 配置；padded 槽位（ground_truth=""）防御性守卫
- **改动文件**：`skyrl_gym/envs/search/env.py`、`env_package/search/envs.py`、`reward_manager/episode.py`、`verl/trainer/ppo/core_algos.py`（新函数）、`verl/trainer/ppo/ray_trainer.py`（GRPO 分支按 flag 分派 + 调用点透传）、`verl/trainer/main_ppo.py`（`algorithm.search_v1_trajectory_return` fail-closed 校验）
- **配套（项目侧，不在 patch 内）**：`searchr1_repro/search_v1_reward.py`（token 边界 alias matcher 重写：`valid_aliases` 返回 word-lists，两字符 alias 只以完整 token 命中，多词 alias 为连续 token 短语）、`searchr1_repro/training_audit.py`（audit 增加 `trajectory_advantage`/`search_v1_group`）、`scripts/run_p3_grpo_search_aware_v1.sh`（`+algorithm.search_v1_trajectory_return=true`）、`tests/test_v1_trajectory_return.py`（T1-T5 构造测试 10 条）、`tests/test_v1_episode_traj_uid.py`（5 条）、`tests/test_v1_env_question_passthrough.py`（真实 env 路径 4 条）、`tests/test_search_v1_reward.py`（+6 条 token 边界）、`scripts/p3_v1_reward_replay.py`（trainer-exact 回放）
- **验证**：patch 在 pre-0008 状态（20bd331 + 0001-0007）干净应用；reverse-check 通过；整树 diff 与 worktree 零差异；30+10 条 CPU 测试全绿
