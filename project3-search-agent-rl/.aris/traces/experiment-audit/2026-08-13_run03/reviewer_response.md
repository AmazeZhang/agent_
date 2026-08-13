# Fresh Reviewer Response

**Reviewer**: `/root/p3_attempt_g_audit`

**Independence**: same-family

**Acceptance**: provisional

**Overall verdict**: WARN

**Reason code**: `literal_no_sigterm_gate_failed_expected_ray_daemon_sigterm_scope_limited_and_tracker_stale`

## Core finding

Attempt G's training, Checkpoint, atomic Rollout evidence, and
`RegisterCenter → GPU Worker → TaskRunner → Driver/Ray` Actor shutdown are supported by archived evidence.
No `SYSTEM_ERROR`, `RAY_WORKER_FAILURE`, unexpected worker, or segmentation fault was found.

The literal claim “all Ray logs contain no SIGTERM” is false: Raylet, GCS and Dashboard retain expected shutdown
SIGTERM from `ray.shutdown()`. Under the original literal gate the result is WARN, not PASS. Under a correctly
scoped “no unexpected Actor/training-worker SIGTERM or SYSTEM_ERROR” gate, the Actor-level shutdown passes.

## A. Ground Truth provenance: PASS

- Actual Hydra paths use smoke train/test parquet, not Fixture.
- Rule reward uses parquet NQ/HotpotQA dataset targets and normalized exact match.
- Real Retriever health reports IndexFlatIP with 21,015,324 vectors and corpus rows.
- This is dataset-target training reward, not an official held-out benchmark.

## B. Score normalization: PASS

- Raw action scores are `1.0×2`, `0.0×12`, `-0.1×7`.
- Logged critic score and episode reward retain the raw ranges.
- GRPO advantage standardization is algorithm behavior, not a normalized quality headline.
- No self-maximum/minimum normalization was found.

## C. Result existence and fidelity: WARN

Verified:

- wrapper exit 0 from 19:57:42 to 20:00:16;
- explicit Step 1 checkpoint load;
- two 21-line Rollout files without partials;
- Step 2 model, optimizer, LoRA, extra and data state;
- `grad_norm=0.275`, `throughput=96.596`, `step=87.623s`, `global_step=2`;
- RegisterCenter, physical worker and TaskRunner shutdown lines.

Warning: the old tracker and audit still said Attempt G was pending, and the original gate prohibited any SIGTERM
while daemon logs contain expected termination SIGTERM.

## D. Dead code/runtime path: PASS

- Atomic exclusive JSONL code is on the ordinary and audit dump runtime paths.
- Both dump calls and all three Actor shutdown functions executed.
- GCS polling waits for DEAD and fails closed on timeout.
- Unit tests use FakeRay, while archived Attempt G logs provide real-run evidence.

## E. Scope: WARN

- 8 train rows, 16 val rows but no validation;
- one seed, one physical GPU, one resumed global update;
- 16 trajectories, 21 actions, max two environment steps;
- no held-out evaluation.

Evidence supports a smoke engineering loop, recovery, one update and shutdown lifecycle only—not complete
reproduction, convergence, generalization, robustness, or quality improvement.

## F. Evaluation type: PASS

`real_gt_training_reward_no_heldout_evaluation`.

## Exit-specific verification

| Check | Finding |
|---|---|
| SYSTEM_ERROR | none |
| RAY_WORKER_FAILURE | none |
| unexpected worker | none |
| segmentation fault | none |
| RegisterCenter | INTENDED_USER_EXIT and DEAD |
| GPU Worker | INTENDED_USER_EXIT and DEAD |
| TaskRunner | INTENDED_USER_EXIT and DEAD |
| Driver | normal ray.shutdown disconnect |
| SIGTERM | expected daemon shutdown entries remain in GCS, Dashboard and Raylet logs |

`ACTOR_UNAVAILABLE / IntentionalSystemExit` on an `exit_actor()` RPC is expected mechanism-level behavior and is
not `SYSTEM_ERROR`.

## Claim impact

- Supported: Step 1→2 recovery update and Step 2 Checkpoint.
- Supported: ordinary/audit Rollout atomic persistence and zero prompt loss tokens in all 21 audit records.
- Supported: ordered intentional shutdown and DEAD observation for the three Actor layers.
- Qualified: Actor/training-worker clean shutdown passed; daemon expected SIGTERM remains.
- Unsupported: all Ray logs have no SIGTERM.
- Unsupported: complete Search-R1 reproduction, quality improvement, convergence, generalization, or robustness.

## Actions

1. Split Actor failure gates from expected Ray-daemon termination signals.
2. Update trackers and Attempt G-specific audit files.
3. Archive a timestamped real-time cleanup snapshot in future runs.
4. Do not make quality/reproduction claims before held-out validation, multiple seeds, longer training and baselines.
