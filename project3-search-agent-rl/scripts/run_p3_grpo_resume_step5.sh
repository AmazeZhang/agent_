#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_root="${PROJECT3_CHECKPOINT_ROOT:-/media/imc/data/project3-search-agent-rl/runs/p3-grpo-shutdown-gate-qwen15b-s0-20260813g/checkpoints/global_step_2}"

export PROJECT3_RESUME_FROM="$checkpoint_root"
export PROJECT3_TOTAL_TRAINING_STEPS=5
# The eight-row smoke dataset produces one batch per epoch. Resume from Step 2
# and allow epochs three through five to provide exactly three more updates.
export PROJECT3_TOTAL_EPOCHS=5

exec "${script_dir}/run_p3_grpo_one_step.sh" "$@"
