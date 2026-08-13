# Experiment Audit Report

**Date**: 2026-08-13

**Auditor**: fresh Codex reviewer（same-family，provisional，只读）

**Project**: Project 3 / Search-R1 veRL Attempt G
**Audited Run**: `p3-grpo-shutdown-gate-qwen15b-s0-20260813g`

## Overall Verdict: WARN

## Integrity Status: warn

原因码：`literal_no_sigterm_gate_failed_expected_ray_daemon_sigterm_scope_limited_and_tracker_stale`

未发现伪造Ground Truth、自我最大值归一化、幽灵更新、缺失结果文件或训练Actor异常退出。
Attempt G的恢复、Step 2更新、Checkpoint、两类Rollout证据和Actor级主动退出均有原始文件支撑。
整体保持WARN有两个原因：原门禁把“所有Ray日志无SIGTERM”写得过宽，而Ray基础设施正常关闭
本身使用预期SIGTERM；实验仍只是8题、单seed、单次恢复更新且无held-out evaluation。

## Checks

### A. Ground Truth Provenance: PASS

- Hydra使用Search-R1 smoke train/test parquet，不使用GT衍生Fixture；
- 训练target来自parquet中的NQ/HotpotQA dataset target；
- `SearchEnv`从外部extras读取ground truth，规则评分器对最终`<answer>`执行规范化Exact Match；
- 实际Retriever健康信息为21,015,324条Wiki-18向量和等量语料。

本Run属于dataset-target训练奖励，不是官方held-out benchmark。

### B. Score Normalization: PASS

- `2.jsonl`保存原始action score：`1.0×2`、`0.0×12`、`-0.1×7`；
- 日志原样报告critic score和episode reward；
- `norm_adv_by_std_in_grpo`是GRPO组内advantage算法步骤，不是对外质量分数归一化；
- 未发现用模型自身max/min/mean制造接近1的质量指标。

### C. Result Existence and Numeric Fidelity: WARN

通过项：

- `metadata.env`记录19:57:42至20:00:16、顶层exit code 0；
- 日志明确从Global Step 1加载模型、Optimizer和Extra State；
- `2.jsonl`和`2.audit.jsonl`各21行，无partial；
- Step 2模型、Optimizer、LoRA、Extra State和Data State均存在；
- 实际指标为`grad_norm=0.275`、`throughput=96.596 token/s`、`step=87.623s`；
- 日志按RegisterCenter、物理GPU Worker、TaskRunner顺序记录主动退出。

WARN项：Raylet、GCS和Dashboard在正常`ray.shutdown()`时仍记录预期SIGTERM。因此不能声称
“全部Ray日志无SIGTERM”，只能声称“Actor/训练Worker无意外SIGTERM或SYSTEM_ERROR”。

### D. Dead Code Detection and Runtime Path: PASS

- 普通Rollout和audit均实际调用独占partial、flush/fsync和atomic rename；
- 运行日志和落盘文件证明两个dump路径实际执行；
- RegisterCenter、GPU Worker和TaskRunner的退出函数均在Attempt G调用；
- GCS状态轮询会等待DEAD，否则抛出TimeoutError；
- 8项单元测试覆盖去重、timeout、DEAD和禁止覆盖；真实Run日志而非FakeRay测试提供最终运行证据。

### E. Scope Assessment: WARN

- 8条train、16条val，但validation未执行；
- seed 0、单物理GPU1、Global Step 1→2的一次恢复更新；
- 16条trajectory、21条action、最多2个environment step；
- `val_before_train=false`、`test_freq=-1`，最终validation为None。

只支持smoke工程闭环、Checkpoint恢复、一次参数更新、Loss Mask/检索证据和Actor生命周期；
不支持完整复现、质量提升、收敛、泛化或鲁棒性。

### F. Evaluation Type: PASS

分类：`real_gt_training_reward_no_heldout_evaluation`。

训练reward使用dataset target，不是synthetic proxy或human evaluation；由于未执行held-out validation，
训练reward和success rate不能表述为验证集性能。

## Exit Gate Detail

- `SYSTEM_ERROR`：训练stdout/stderr及全部归档Ray日志均未发现；
- `RAY_WORKER_FAILURE`：未发现；
- unexpected worker：未发现；
- Segmentation fault：未发现；
- RegisterCenter、GPU Worker、TaskRunner：均`INTENDED_USER_EXIT`且观察到DEAD；
- Driver：`ray.shutdown()`正常断开；
- 基础设施SIGTERM：GCS/Raylet/Dashboard存在，属于`EXPECTED_TERMINATION`，不能混同训练Worker失败。

## Action Items

1. 门禁永久拆为“训练Actor禁止意外SIGTERM/SYSTEM_ERROR”和“基础设施允许EXPECTED_TERMINATION”；
2. 后续归档带时间戳的GPU、PID、端口、Retriever和tmux清理快照；
3. 在held-out validation、多seed、更长训练和baseline比较前继续禁止完整复现/质量提升声明；
4. 后续若晋级5/20步，继续使用新Run ID、GPU1、原子证据文件和同一退出审计。

## Claim Impact

- C1：Global Step 1→2恢复训练更新和Step 2 Checkpoint——**supported**；
- C2：普通/audit Rollout原子非覆盖落盘——**supported**；
- C3：21条记录Prompt policy-loss token均为0——**supported**；
- C4：RegisterCenter→GPU Worker→TaskRunner主动退出——**supported**；
- C5：“Actor/训练Worker级干净退出门禁通过”——**supported with qualifier**；
- C6：“全部Ray日志无SIGTERM”——**unsupported**；
- C7：“Search-R1完整复现或质量提升”——**unsupported**。

完整审计trace：`.aris/traces/experiment-audit/2026-08-13_run03/`。
