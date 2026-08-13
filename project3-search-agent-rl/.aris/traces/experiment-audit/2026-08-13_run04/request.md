# Experiment Audit Request

Fresh same-family provisional reviewer; read-only and adversarial.

Audited run:

`/media/imc/data/project3-search-agent-rl/runs/p3-grpo-resume-step5-qwen15b-s0-20260813h`

Inputs supplied as paths only:

- run metadata, Hydra config/overrides, stdout/stderr and complete archived Ray logs;
- Step 3/4/5 ordinary and audit Rollout JSONL;
- source Step 2 and output Step 3/4/5 model, Optimizer, LoRA, Extra/Data State;
- Search-R1 smoke manifest/parquet and retriever logs;
- training audit/lifecycle implementation, veRL patches and unit tests;
- five-step plan, execution log, progress sync and prior audit trackers.

Checklist:

- A Ground Truth provenance and fixture exclusion;
- B score normalization and raw reward fidelity;
- C result existence, resume continuity, parameter/Optimizer/Scheduler changes;
- D runtime/dead-code, loss mask, retrieval metadata and shutdown execution;
- E experiment scope versus claim language;
- F evaluation type;
- separate Actor failure signatures from expected Ray daemon shutdown;
- identify exactly what the five-step run does and does not support.

Required output: PASS/WARN/FAIL per check with exact evidence, overall verdict, reason code,
claim impact and required next actions.
