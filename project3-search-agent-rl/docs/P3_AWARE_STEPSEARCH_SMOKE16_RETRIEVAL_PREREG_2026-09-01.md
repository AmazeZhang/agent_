# P3 Aware-v2 vs StepSearch same-question retrieval preregistration — 2026-09-01

## Purpose

Test whether the external StepSearch policy produces more useful queries than
our Aware-v2 checkpoint on the exact same frozen smoke-16 questions. This is a
mechanism screen, not a same-initialization algorithm comparison and not a
training run.

## Frozen runs

Existing external reference:

- Run: `p3-eval-stepsearch3b-smoke16-20260828b`
- Model: `Zill1/StepSearch-3B-Base` revision `a89ec38...`
- Adapter: official StepSearch prompt and public first-action boundary

Aware-v2 run to add:

- Model: `p3-aware-v2-grpo10-seed2026-gs10-merged-20260824a`
- Tokenizer: frozen local `Qwen2.5-3B`
- New run ID: `p3-eval-aware-v2-seed2026-gs10-smoke16-20260901a`

Shared conditions: frozen Search-R1 smoke-16 manifest, real Wiki-18 E5
IndexFlatIP Retriever, top-k 3, greedy temperature 0, one rollout, seed 0,
max four steps, input 3072, output 256, physical GPU1 only, GPU0 forbidden.
The models and prompt protocols differ, so all comparisons are descriptive
external-policy comparisons.

## Metrics and decision rule

Use the same frozen answer-alias evidence matcher on returned document text.
Report per-call and per-question evidence hit, successful/invalid searches,
multi-hop rate, new-document increments, true redundancy, answer compliance,
EM, and paired question sets (both hit, StepSearch-only hit, Aware-only hit,
neither hit). Do not repair malformed answers or inspect model-generated text
for evidence.

The StepSearch planning/query-rewrite mechanism is eligible for a bounded
Aware experiment only if it gains at least two evidence-hit questions out of
16 over Aware-v2 and does not reduce EM on this screen. Otherwise stop the
external-model branch; do not run StepSearch confirm-256. No training is
authorized by this preregistration.

## Gates

The Aware run must exit 0 under `start_tmux_run.sh -> run_managed.sh`, write
exactly 16 episodes atomically, pass the round-two prompt gate, have no
Retriever errors, and leave no GPU1 compute process. Preserve and record any
failure with a new run ID rather than overwriting artifacts.

## Post-run result

Aware-v2 run `p3-eval-aware-v2-seed2026-gs10-smoke16-20260901a` exited 0,
wrote 16 episodes, passed 16/16 round-two prompt checks, and completed all
18 searches successfully. Physical GPU1 peaked at 14,045 MiB and cleanup
reported no remaining compute process. GPU0 was not used.

| Metric | Aware-v2 | StepSearch |
|---|---:|---:|
| Evidence-hit questions | 8/16 | 5/16 |
| Evidence-hit calls | 8/18 | 8/21 |
| Multi-hop episodes | 2/16 | 5/16 |
| True redundant searches | 1 | 0 |
| Answer compliance | 16/16 | 9/16 |
| EM | 4/16 | 2/16 |

Paired evidence-hit sets were: both hit `[4, 5, 11, 14, 15]`, StepSearch-only
hit `[]`, Aware-only hit `[0, 3, 10]`, and neither hit
`[1, 2, 6, 7, 8, 9, 12, 13]`. Thus StepSearch made more multi-hop calls but
did not cover any question that Aware-v2 missed; its eight hit calls were
concentrated in five questions. Aware-v2 covered three additional questions
and doubled smoke EM.

The pre-registered eligibility rule fails (`5 - 8 = -3` evidence-hit
questions, and StepSearch EM is lower). Stop the external StepSearch branch:
do not run its confirm-256 and do not transplant its planning/query-rewrite
mechanism on this evidence. This result does not imply that planning is
universally harmful; it only says this checkpoint/protocol combination did
not outperform the existing Aware-v2 mechanism screen.

Reproducible paired outputs:

- `analysis/p3_aware_stepsearch_smoke16_retrieval_2026-09-01.json`
- `analysis/p3_aware_stepsearch_smoke16_retrieval_2026-09-01.md`
