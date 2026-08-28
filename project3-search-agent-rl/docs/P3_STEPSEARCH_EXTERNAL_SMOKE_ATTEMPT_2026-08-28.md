# P3 StepSearch external smoke attempt — 2026-08-28

## Attempt A: preserved gate failure

- Run ID: `p3-eval-stepsearch3b-smoke16-20260828a`
- Run directory: `/media/imc/data/project3-search-agent-rl/runs/p3-eval-stepsearch3b-smoke16-20260828a`
- Managed exit code: `2` (fail-closed prompt gate)
- GPU: physical GPU1 only; physical peak 13,945 MiB; cleanup reported no compute processes
- Evaluation: 16/16 episodes written atomically, greedy, one rollout, max four steps
- Search: 16/16 episodes searched; 17/17 search calls succeeded; no invalid query
- Answer EM: 1/16 (6.25%); evidence hit: 5/17 searches (29.41%)
- Training curves: not applicable; this is evaluation-only and the managed runner records `curve_generation=skipped_no_training_metrics`

The run is not eligible to unlock confirm-256 because its round-two prompt gate
reported only 5/16 prompts as containing the exact query string. The artifacts
are preserved and must not be presented as a passed smoke.

## Root cause and bounded correction

Inspection of `episodes.jsonl` showed two distinct issues:

1. The prompt checker required the literal form `<search>query</search>` and
   rejected valid recalled tags with harmless inner whitespace, such as
   `<search>query </search>`.
2. More importantly, generation was not truncated before projection. Some
   outputs continued beyond `</search>` into fabricated
   `<information>/<observation>/<answer>` text, and that suffix entered the
   next-turn memory.

StepSearch's public generator at source commit
`43215bab9118a4c8e01b15082f74b2aea30c1fc8`
(`search_r1/llm_agent/generation.py`, `_postprocess_responses`) truncates at the
first `</search>`, or at the first `</answer>` when no search closes, before the
environment processes the action. The adapter now implements this same
boundary before projection and uses normalized tag-content equality in the
prompt gate. Twenty CPU tests pass, including regression coverage for suffix
removal and tag whitespace.

The correction changes only the external StepSearch evaluation adapter and
its fail-closed checker. It does not change retrieval, reward, training code,
the clean Search-R1 protocol, or any completed prior run.
