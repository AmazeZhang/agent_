# Experiment Audit Report

**Date**: 2026-08-13

**Auditor**: fresh Codex reviewer（same-family，provisional，只读）

**Project**: Project 3 / Search-R1 veRL Attempt H

**Audited Run**: `p3-grpo-resume-step5-qwen15b-s0-20260813h`

## Overall Verdict: WARN

原因码：`step5_engineering_run_verified_but_no_heldout_eval_single_seed_zero_reward_steps`

五步工程运行真实完成：从Step 2恢复，连续执行三次新增优化更新，保存Step 3/4/5
Checkpoint和六个原子Rollout文件，参数、Optimizer及Scheduler状态连续变化，Actor生命周期干净。
但Step 4/5的任务reward和success均为0，且范围仅为8题、单seed、单GPU、无held-out evaluation。
因此工程闭环成立，任何质量提升、收敛、泛化或完整Search-R1复现声明均不成立。

## Checks

### A. Ground Truth Provenance: PASS

- Hydra实际使用Search-R1 smoke parquet；train/test为8/16行；
- Ground truth来自NQ/HotpotQA数据行，经环境传入规则Exact Match评分器；
- GT-derived fixture只用于协议、超时、格式和reward smoke测试，未进入该训练Run；
- 实际Retriever为21,015,324条Wiki-18向量/语料，成功结果均为数字Wiki ID。

### B. Score Normalization: PASS

- Step 3/4/5原始action score分布分别为`{1:2,0:10,-0.1:9}`、
  `{0:15,-0.1:7}`、`{0:13,-0.1:11}`；
- 日志保留critic score和episode reward；
- `norm_adv_by_std_in_grpo=true`只对训练advantage做组内标准化，不是外部评价成绩；
- 未发现自我max/min归一化或人为制造接近1的指标。

### C. Result Existence and Numeric Fidelity: PASS

- `metadata.env`记录21:23:28至21:29:24、exit code 0；
- stdout明确加载Step 2 model、optimizer和extra/data state；
- Step 3/4/5真实执行并分别保存完整Checkpoint和两种Rollout JSONL；
- 392/392 LoRA张量在每段均改变；Adam计数6→9→12→15；Scheduler连续推进；
- 最终checkpoint tracker为5，无`.partial`或重复Adapter。

先前tracker过时问题已由Attempt H完成报告、执行日志、进度同步和本审计文件修正。

### D. Runtime and Dead-Code Detection: PASS

- 普通与audit Rollout的独占partial、fsync、原子rename路径在真实Run执行；
- 67条audit记录全部`prompt_policy_loss_tokens=0`；
- typed retrieval metadata实际落盘，累计success=10、invalid_query=9；
- RegisterCenter、GPU Worker、TaskRunner均`INTENDED_USER_EXIT`并观察到DEAD；
- Actor/训练Worker无SYSTEM_ERROR、RAY_WORKER_FAILURE、异常SIGTERM或segfault；
- Ray daemon正常关闭时存在`EXPECTED_TERMINATION` SIGTERM，不应混同训练Actor失败。

### E. Scope Assessment: WARN

- 8个训练问题、seed 0、单GPU、从Step 2新增3次更新；
- 每步16条trajectory，Action数21/22/24；
- `val_before_train=false`、`test_freq=-1`，final validation为None；
- Step 3 episode reward/success为0.125，Step 4/5均为0。

只支持短程恢复、优化、持久化、检索、mask和生命周期工程链路。末两步的零任务成功是对
“质量改善”的直接反证，非零梯度和参数变化不得解释为模型效果提升。

### F. Evaluation Type: PASS

分类：`real_gt_training_reward_no_heldout_evaluation`。

训练reward来自dataset target与真实Wiki Retriever，不是synthetic proxy或human evaluation；
它仍是训练batch reward，不能表述为验证集或测试集性能。

## Claim Impact

- Step 2→5恢复连续性与三次真实优化更新——**supported**；
- Step 3/4/5 Checkpoint和六个原子JSONL——**supported**；
- Prompt loss token为0、数字Wiki ID及typed failure——**supported**；
- Actor/训练Worker级干净退出——**supported with qualifier**；
- “五步工程晋级通过”——**supported only as single-seed smoke engineering stability**；
- “所有Ray日志无SIGTERM”——**unsupported**；
- 质量提升、收敛、泛化、完整Search-R1复现、held-out性能——**unsupported**。

## Required Next Evidence

1. 对Step 2和Step 5执行同配置、独立held-out evaluation；
2. 增加未训练baseline和多个seed；
3. 依据评测结果决定长训练或数据扩展，不以参数变化替代效果证据；
4. 持续固化Adapter hash/delta、Optimizer/Scheduler、JSONL hash和Actor/daemon分层退出证据。

完整审计trace：`.aris/traces/experiment-audit/2026-08-13_run04/`。
