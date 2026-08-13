# Fresh Reviewer Response

Overall verdict: **WARN**  
Reason code: `step5_engineering_run_verified_but_no_heldout_eval_single_seed_zero_reward_steps_and_trackers_stale`

This is a same-family provisional, read-only review. No files were modified by the reviewer.

## A. Ground Truth provenance — PASS

Hydra points to the smoke train/test parquet, not the fixture corpus. The manifest limits the
GT-derived fixture to protocol, timeout, format and reward smoke tests and forbids benchmark claims.
The parquet ground truth enters `SearchEnv` through external `env_kwargs`; the reward implementation
performs normalized exact match against that dataset target. The actual retriever reports 21,015,324
Wiki vectors, and all persisted successful document IDs are numeric Wiki IDs rather than `fixture-*`.
No official held-out evaluation was run.

## B. Score normalization — PASS

Ordinary JSONL retains raw action scores. Recomputed distributions are Step 3
`{1:2,0:10,-0.1:9}`, Step 4 `{0:15,-0.1:7}`, and Step 5 `{0:13,-0.1:11}`.
GRPO's `norm_adv_by_std_in_grpo` is group-wise training-advantage normalization, not an externally
reported quality score. There is no self-max normalization.

## C. Result existence and numeric fidelity — WARN

The artifacts themselves pass. Metadata records exit code 0; stdout explicitly loads the Step 2
model, optimizer and extra/data state; Steps 3, 4 and 5 each execute, dump both JSONL forms and save
complete checkpoints. All 392 LoRA tensors change at each transition; approximate changed elements
are 36,928,795 / 36,928,375 / 36,928,267 with delta L2 0.026777 / 0.021472 / 0.019240.
Adam counters progress 6→9→12→15 and scheduler state progresses continuously. These prove real
updates, not quality improvement.

At review time, `EXPERIMENT_AUDIT.md/json`, the execution log and progress tracker still described
Attempt G or only the plan, so tracker currency was WARN. The primary agent must update them before
claiming the project record is current.

## D. Runtime and dead-code — PASS

The atomic ordinary/audit dump paths ran and left no partial files. All 67 audit actions have zero
prompt policy-loss tokens. Typed retrieval status was persisted: Step 3 success=3/invalid_query=2,
Step 4 success=2/invalid_query=4, Step 5 success=5/invalid_query=3. RegisterCenter, GPU Worker and
TaskRunner all exited with `INTENDED_USER_EXIT` and reached DEAD.

The complete Ray scan found no Actor SYSTEM_ERROR, RAY_WORKER_FAILURE, unexpected worker failure,
Actor SIGTERM or segmentation fault. Raylet/GCS/dashboard SIGTERM belongs to documented
`EXPECTED_TERMINATION` during node shutdown and must be reported separately.

## E. Scope — WARN

This run covers eight training questions, seed 0, one GPU, three updates after Step 2, and 16
trajectories per step. Validation is disabled and final validation is None. Step 3 task episode
reward/success is 0.125; Steps 4 and 5 both have zero reward and zero success. Thus only short-run
engineering stability is supported. The two zero-success steps directly prevent a quality-improvement
claim.

## F. Evaluation type — PASS

Classification: `real_gt_training_reward_no_heldout_evaluation`. Dataset-provided targets and a real
Wiki retriever are used, but this remains training-batch reward rather than formal evaluation.

## Claim impacts

- Supported: Step 2→5 continuity; three genuine optimizer updates; Step 3/4/5 checkpoints and six
  atomic JSONL files; zero prompt-loss tokens; numeric Wiki IDs and typed failures; clean Actor exits.
- Needs qualifier: “five-step promotion passed” means only single-seed, eight-question, no-validation
  engineering stability; “clean shutdown” applies to training Actors, while Ray daemons show expected
  shutdown SIGTERM.
- Unsupported: quality improvement, convergence, generalization, complete Search-R1 reproduction or
  held-out performance.

## Actions

1. Update the completion report, execution log, progress sync and audit trackers for Attempt H.
2. Preserve raw score distributions and explicitly report zero reward/success at Steps 4 and 5.
3. Run an independent held-out evaluation, then multi-seed and baseline comparisons before longer
   training or quality claims.
4. Continue machine-readable hashes/deltas and Actor-versus-infrastructure shutdown classification.
