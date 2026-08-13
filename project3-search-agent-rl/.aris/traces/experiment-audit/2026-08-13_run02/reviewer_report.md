# Fresh Experiment Integrity Audit — P3 Resume Step 2

**Agent:** `/root/p3_resume_integrity_audit`

**Review independence:** same-family, provisional

**Overall verdict:** **WARN** for the checkpoint-resume engineering gate

**Claim gate:** **FAIL** for clean shutdown, completed Search-R1 reproduction, or quality improvement

No fake ground truth, prediction-derived normalization, phantom checkpoint, or fabricated optimizer update was
found. The main integrity failures are the shutdown contradiction and the limited smoke scope.

## A. Ground-truth provenance — PASS

- Eight training records use dataset targets from NQ/HotpotQA; references are not derived from model output.
- `SearchEnv` receives the external target and the reward scorer applies normalized exact match to the final
  `<answer>`.
- The run used the full 21,015,324-vector Wiki-18 service and persisted numeric corpus IDs; the known
  ground-truth-derived fixture was not used.

## B. Score normalization — PASS

- No metric is divided by statistics from the model's own predictions.
- The 21 raw Action scores are two `1.0`, twelve `0.0`, and seven `-0.1`, mean `0.0619047614`, matching
  `critic/score/mean=0.062`.
- Episode Reward mean `0.125` uses 16 trajectories; the denominator distinction is legitimate and labeled.
- GRPO Advantage normalization is an algorithm operation, not a reported normalized quality score.

## C. Result existence and numeric fidelity — WARN

- Hydra and runtime both identify the source `global_step_1`; runtime loads model, optimizer, extra state and
  DataLoader state.
- Runtime saves Step 2 and reports `training/global_step=2`, `grad_norm=0.283`; the tracker contains `2`.
- Scheduler advances from `last_epoch=1/_step_count=2` to `2/3`.
- Both optimizers have 421 states; internal Adam counters advance 3→6 because each Global Actor Update uses
  three mini-batch optimizer steps.
- All 392 Adapter tensors changed; LoRA-B has 20,643,840 nonzero elements. Adapter hashes differ between steps.
- Both Step 2 JSONL files exist and contain 21 complete records.
- WARN: wrapper `exit_code=0` contradicts Ray's post-save `SYSTEM_ERROR` and SIGTERM worker-exit report.

## D. Dead code and observability — WARN

- The structured audit has 21 records, 8 Group UIDs, 16 Trajectory UIDs and environment steps 16/5.
- All 21 Prompt Loss Masks are zero; Policy Loss Token total 2629 equals Active Response Token total.
- Five prompt records contain retrieval observations. The audit and `dp_actor` use the same
  Response Attention Mask in this non-multi-turn run.
- Three successful searches persist nine numeric Wiki IDs. Two invalid queries persist typed errors, empty IDs
  and `retrieval_failed=true`.
- WARN: the audit JSONL is exclusive/fsynced/atomic, but the ordinary generation JSONL still uses overwrite
  mode. No evidence indicates actual overwrite in this run.

## E. Scope — WARN

- Scope is eight questions, sixteen trajectories, twenty-one actions, seed 0, maximum two environment steps,
  and one resumed Global Actor Update.
- No held-out evaluation occurred (`val_before_train=false`, `test_freq=-1`, final validation `None`).
- Evidence supports engineering continuity only, not convergence, generalization, robustness, quality gain or
  completed reproduction.

## F. Evaluation classification — PASS

`real_gt_training_reward_no_heldout_evaluation`.

## Claim impacts

- C1 checkpoint recovery Step 1→2: **SUPPORTED**.
- C2 genuine second Global Actor Update and Adapter change: **SUPPORTED**, with Global/Mini-batch wording.
- C3 retrieved observation/prompt tokens excluded from loss: **SUPPORTED for this run**.
- C4 real Wiki statuses/IDs persisted and failures distinguished: **SUPPORTED**.
- C5 clean infrastructure shutdown: **CONTRADICTED / FAIL**.
- C6 completed Search-R1 reproduction or quality improvement: **UNSUPPORTED / FAIL**.

## Required actions

1. Fix Ray shutdown until no `SYSTEM_ERROR`, unexpected worker death or SIGTERM remains.
2. Make the ordinary generation dump exclusive and atomic.
3. Preserve the exact “second Global Actor Update” wording.
4. Run held-out validation, multiple seeds and longer training before quality/reproduction claims.
