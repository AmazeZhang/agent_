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
