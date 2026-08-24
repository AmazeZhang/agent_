#!/usr/bin/env bash
# CPU-only, fail-closed merge for the preregistered seed2026 clean-vs-aware pair.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
data_root="${PROJECT3_DATA_ROOT:-/media/imc/data}"
project_data="${data_root}/project3-search-agent-rl"
python_bin="${project_data}/envs/searchr1-repro-cu124/bin/python"
vendor_dir="${project_dir}/vendor/verl-agent-v2"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "merge must be CPU-only (CUDA_VISIBLE_DEVICES must be empty)" >&2
  exit 10
fi

expected_upstream="20bd331bdbc9026a5668e11362178e10ab7400c8"
actual_upstream="$(git -C "${vendor_dir}" rev-parse HEAD)"
[[ "${actual_upstream}" == "${expected_upstream}" ]] || {
  echo "v2 vendor commit mismatch: ${actual_upstream}" >&2
  exit 11
}

clean_src="${project_data}/runs/p3-clean-grpo10-seed2026-fsdp6-20260824a/checkpoints/global_step_10/actor"
aware_src="${project_data}/runs/p3-aware-v2-grpo10-seed2026-fsdp6-20260824a/checkpoints/global_step_10/actor"
clean_dst="${project_data}/models/p3-clean-grpo10-seed2026-gs10-merged-20260824a"
aware_dst="${project_data}/models/p3-aware-v2-grpo10-seed2026-gs10-merged-20260824a"
gate_dir="${project_data}/gates/p3-seed2026-pair-merge-20260824a"

for src in "${clean_src}" "${aware_src}"; do
  [[ -d "${src}" ]] || { echo "missing source checkpoint: ${src}" >&2; exit 12; }
  for kind in model optim extra_state; do
    count="$(find "${src}" -maxdepth 1 -type f -name "${kind}_world_size_6_rank_*.pt" | wc -l)"
    [[ "${count}" == 6 ]] || { echo "${src}: expected 6 ${kind} shards, got ${count}" >&2; exit 13; }
  done
done

for target in "${clean_dst}" "${aware_dst}" "${gate_dir}"; do
  [[ ! -e "${target}" ]] || { echo "refusing to overwrite existing target: ${target}" >&2; exit 14; }
done

mkdir -p -- "${gate_dir}"

merge_one() {
  local label="$1" src="$2" dst="$3"
  local before="${gate_dir}/${label}-source-before.sha256"
  local after="${gate_dir}/${label}-source-after.sha256"
  local merge_log="${gate_dir}/${label}-merge.log"
  local verify_log="${gate_dir}/${label}-verify.log"

  find "${src}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum >"${before}"
  env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH="${vendor_dir}:${project_dir}" \
    "${python_bin}" "${vendor_dir}/scripts/model_merger.py" merge \
      --backend fsdp --local_dir "${src}" --target_dir "${dst}" \
      >"${merge_log}" 2>&1
  env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH="${vendor_dir}:${project_dir}" \
    "${python_bin}" "${script_dir}/verify_p3_merged_model.py" --merged-dir "${dst}" \
      >"${verify_log}" 2>&1
  find "${src}" -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum >"${after}"
  cmp --silent "${before}" "${after}" || {
    echo "${label}: source checkpoint changed during merge" >&2
    exit 15
  }
  grep -q '^VERIFY_MERGED: PASS$' "${verify_log}" || {
    echo "${label}: merged model verification did not pass" >&2
    exit 16
  }
  echo "${label}: MERGE_AND_VERIFY_PASS target=${dst}"
}

merge_one clean "${clean_src}" "${clean_dst}"
merge_one aware "${aware_src}" "${aware_dst}"

sha256sum "${gate_dir}"/* >"${gate_dir}/gate-files.sha256"
echo "PAIR_MERGE_PASS gate_dir=${gate_dir}"
