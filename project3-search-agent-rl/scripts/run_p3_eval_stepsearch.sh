#!/usr/bin/env bash
# Evaluation-only external baseline wrapper for the public StepSearch-3B model.
# It deliberately delegates every resource/tree/model/retriever/cleanup gate to
# run_p3_eval_v2.sh and changes only the prompt/history protocol in the Python
# evaluator. Must still be launched through run_managed.sh inside named tmux.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
expected_model_suffix="/models/StepSearch-3B-Base-a89ec38"

if [[ "${PROJECT3_EVAL_MODEL:-}" != *"${expected_model_suffix}" ]]; then
  echo "StepSearch external baseline requires PROJECT3_EVAL_MODEL ending ${expected_model_suffix}" >&2
  exit 21
fi
if [[ "${PROJECT3_EVAL_TOKENIZER:-}" != "${PROJECT3_EVAL_MODEL}" ]]; then
  echo "StepSearch external baseline requires its own tokenizer (PROJECT3_EVAL_TOKENIZER == PROJECT3_EVAL_MODEL)" >&2
  exit 22
fi

export PROJECT3_EXTERNAL_STEPSEARCH_PROTOCOL=1
exec bash "${script_dir}/run_p3_eval_v2.sh"

