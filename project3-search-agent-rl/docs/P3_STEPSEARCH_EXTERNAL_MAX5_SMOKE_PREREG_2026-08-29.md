# P3 StepSearch external max-five smoke preregistration — 2026-08-29

## Question

Does changing only the interaction cap from the current four steps to
StepSearch's public five-turn setting rescue strict, parseable answers on the
fixed smoke-16 set?

## Frozen configuration

- Model: `Zill1/StepSearch-3B-Base`, revision
  `a89ec38cd2a21461320f9a81eb29be019c142fe5`
- Dataset: frozen Search-R1 smoke-16 manifest
- Retriever: existing shared E5 Wiki-18 IndexFlatIP, top-k 3
- Decoding: greedy, temperature 0.0, one rollout, seed 0
- Prompt/action adapter: commit `e50f79e`, including the official first
  `</search>`/`</answer>` generation boundary
- GPU: physical GPU1 only; GPU0 forbidden
- Changed variable: `max_steps=5`
- Unchanged: `history_length=4`, input/output limits, retrieval, reward,
  projection, answer parser, and all model weights

## Paired reference and decision rule

The paired engineering reference is
`p3-eval-stepsearch3b-smoke16-20260828b` (`max_steps=4`): 16/16 searched,
21/21 successful searches, answer compliance 9/16, EM 2/16, and 7/16
episodes at the cap without a parseable answer.

The max-five run must first satisfy all existing smoke gates: managed exit 0,
16 atomic episodes, prompt gate pass, at least one valid non-empty search, no
retriever error, and clean GPU teardown. Report changes in answer compliance,
EM, evidence hit, search calls, and cap-without-answer count. This experiment
is diagnostic: no confirm-256 or training is authorized by this document.
