# Experiment Audit Request

Fresh same-family provisional reviewer; read-only and adversarial.

Audited run:

`/media/imc/data/project3-search-agent-rl/runs/p3-grpo-shutdown-gate-qwen15b-s0-20260813g`

Inputs supplied as paths only:

- `tests/test_p3_training_audit.py`
- `tests/test_p3_training_lifecycle.py`
- `tests/test_p25_cpu_retriever_service.py`
- `tests/test_search_p1.py`
- `searchr1_repro/training_audit.py`
- `searchr1_repro/training_lifecycle.py`
- `patches/0003-graceful-ray-shutdown-and-atomic-rollout.patch`
- Attempt G metadata, stdout, stderr, both Rollout JSONL files, Step 2 Checkpoint and complete Ray log directory
- P3 clean-shutdown, execution, progress and checkpoint-resume documents
- prior `EXPERIMENT_AUDIT.md/json`
- resolved Hydra config/overrides and experiment profile

Checklist:

- A Ground Truth provenance
- B Score normalization
- C Result existence and numeric fidelity
- D Dead code/runtime execution
- E Scope versus claim language
- F Evaluation type
- Special exit audit for SYSTEM_ERROR, RAY_WORKER_FAILURE, unexpected worker, SIGTERM and segmentation fault
- Explicit distinction between engineering shutdown gate and complete Search-R1 reproduction/quality claims

Required output: PASS/WARN/FAIL per check with exact evidence, overall verdict, reason code, claim impact and actions.
