#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint_root="${PROJECT3_CHECKPOINT_ROOT:-/media/imc/data/project3-search-agent-rl/runs/p3-grpo-1step-qwen15b-s0-20260813c/checkpoints/global_step_1}"

export PROJECT3_RESUME_FROM="$checkpoint_root"
export PROJECT3_TOTAL_TRAINING_STEPS=2
# The eight-row smoke dataset contains one batch. Resume first finishes that
# exhausted epoch, then epoch two provides the single batch for global_step_2.
export PROJECT3_TOTAL_EPOCHS=2

exec "${script_dir}/run_p3_grpo_one_step.sh" "$@"
