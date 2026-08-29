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

## Post-run result

- Run ID: `p3-eval-stepsearch3b-smoke16-max5-20260829a`
- Run directory: `/media/imc/data/project3-search-agent-rl/runs/p3-eval-stepsearch3b-smoke16-max5-20260829a`
- Managed exit code: 0; prompt gate: 16/16 passed
- GPU1 physical peak: 14,083 MiB; cleanup: no compute processes
- Answer compliance: 9/16 (unchanged from max-four)
- EM: 2/16 (unchanged)
- Search calls: 22/22 successful (max-four: 21/21)
- Multi-hop episodes: 6/16 (max-four: 5/16)
- Evidence hits: 8/22, 36.36% (max-four: 8/21, 38.10%)
- True redundant and invalid searches: zero
- Training curves: not applicable; evaluation-only

All seven max-four episodes that lacked a parseable answer ran for the fifth
step. None gained a parseable answer and none became correct; all seven ended
at the five-step cap. The other nine trajectories and their scores were
unchanged. Therefore the paired smoke rejects the hypothesis that the current
failure is primarily caused by the four-step limit. The extra turn adds one
successful search but does not improve answer compliance or EM. No larger
max-five run is justified by this result.
