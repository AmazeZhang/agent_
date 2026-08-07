#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trajectory_root="$project_root/vendor/AgentRx/trajectories/tau-retail"
ground_truth="$project_root/vendor/AgentRx/data/ground_truth/tau_retail.json"
examples_dir="$project_root/diagnosis/few_shot_examples/magentic-real-v2"
output_root="$project_root/diagnosis/runs/upstream-tau-failure-fewshot-v2-benchmark"

cases=(
  instruction_adherence_failure
  intent_plan_misalignment
  invalid_invocation
  invention_new_info
  misinterpretation_tool_output
  underspecified_intent
  hallucination_doubt
)

for case_name in "${cases[@]}"; do
  echo "=== AgentRx few-shot benchmark: $case_name ==="
  "$project_root/scripts/run_agentrx_deepseek.sh" \
    "$trajectory_root/$case_name.json" \
    --domain tau \
    --endpoint azure \
    --skip-static \
    --skip-dynamic \
    --examples-dir "$examples_dir" \
    --ground-truth "$ground_truth" \
    --run-dir "$output_root/$case_name"
done
