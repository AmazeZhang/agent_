#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
env_dir="${PROJECT3_ENV_DIR:-${data_root}/project3-search-agent-rl/envs/searchr1-repro-cu124}"
cache_dir="${data_root}/project3-search-agent-rl/cache"
python_bin="${PROJECT3_PYTHON_BIN:-/home/imc/anaconda3/envs/paretotool-retriever/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python 3.10 interpreter not found: ${python_bin}" >&2
  exit 2
fi
if [[ "$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.10" ]]; then
  echo "the reproduction environment requires Python 3.10" >&2
  exit 2
fi

mkdir -p -- "$cache_dir/uv" "$cache_dir/pip" "$cache_dir/huggingface" "$cache_dir/torch" "$(dirname -- "$env_dir")"
export UV_CACHE_DIR="$cache_dir/uv"
export PIP_CACHE_DIR="$cache_dir/pip"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-300}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
export PIP_RESUME_RETRIES="${PIP_RESUME_RETRIES:-10}"
export HF_HOME="$cache_dir/huggingface"
export TORCH_HOME="$cache_dir/torch"
export CUDA_HOME=/usr/local/cuda-12.4
export MAX_JOBS="${MAX_JOBS:-4}"
# This host uses RTX 4090 (Ada, compute capability 8.9). Avoid FlashAttention's
# much slower fallback build for unrelated SM80/SM90 architectures.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if [[ -e "$env_dir" ]]; then
  echo "refusing to modify existing environment: ${env_dir}" >&2
  echo "set PROJECT3_ENV_DIR to a new versioned path for a fresh environment" >&2
  exit 3
fi

uv venv --seed --python "$python_bin" "$env_dir"
"$env_dir/bin/python" -m pip install --upgrade pip==25.1.1 setuptools==80.9.0 wheel==0.45.1
"$env_dir/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
"$env_dir/bin/python" -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation
"$env_dir/bin/python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -r "$project_dir/configs/requirements-searchr1-repro.txt"
"$env_dir/bin/python" -m pip install --no-deps -e "$project_dir/vendor/verl-agent"
"$env_dir/bin/python" -m pip install -e "$project_dir/vendor/verl-agent/agent_system/environments/env_package/search/third_party"

"$env_dir/bin/python" -m pip check
# Editable submodule packages are installed explicitly above and locked by the
# Git submodule SHA. Exclude pip's host-path-derived editable lines because they
# are not portable to another checkout location.
"$env_dir/bin/python" -m pip freeze --all | sed '/^-e /d' | LC_ALL=C sort >"$project_dir/configs/requirements-searchr1-repro.lock.txt"
echo "environment created: ${env_dir}"
