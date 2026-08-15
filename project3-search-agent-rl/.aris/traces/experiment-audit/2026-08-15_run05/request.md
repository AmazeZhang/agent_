# Experiment Audit Request

Fresh same-family provisional, read-only adversarial audit of the Project 3 vLLM smoke and heldout-32
evaluation. Inputs were supplied as paths only: HF/vLLM evaluation and comparison scripts; environment,
projection and reward code; five vLLM Run directories; three HF-vs-vLLM paired comparison JSON files;
generation diagnosis; heldout manifest/parquet; three adapter identities; progress tracker and prior audit.

Required checks: A ground-truth provenance/leakage, B normalization, C raw result recomputation and adapter
identity, D runtime/dead-code/backend attribution and committed-code reproducibility, E paired statistics and
scope, F evaluation classification, GPU cleanup, and exact supported/qualified/unsupported claims.
