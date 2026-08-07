#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

"$workspace/project1-harness-evolution/.venvs/agent-lightning/bin/python" -c \
  "import agentlightning; print('OK agentlightning')"
"$workspace/project1-harness-evolution/.venvs/agentrx/bin/python" -c \
  "import agentrx; print('OK agentrx')"
"$workspace/project1-harness-evolution/.venvs/tau2/bin/python" -c \
  "import tau2; print('OK tau2')"
"$workspace/project2-coding-agent-rl/.venvs/swe-tools/bin/python" -c \
  "import sweagent, swesmith; print('OK sweagent + swesmith')"
"$workspace/project2-coding-agent-rl/.venvs/rllm-base/bin/python" -c \
  "import rllm, torch; print('OK rllm + torch', torch.__version__)"

