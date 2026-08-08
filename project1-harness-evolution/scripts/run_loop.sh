#!/usr/bin/env bash
# 项目一自进化闭环一键复现（DEVELOPMENT_SCOPE 2.2-1 验收项，2026-08-08 补齐）
#
# 一条命令完成: 轨迹采集 → 数据划分 → 失败诊断 → 基线 val 对照 → 四臂优化
#   → 门控评测 → 汇总报告
#
# 前置（首次运行）:
#   - .venvs/agent-lightning/bin/python 可用（uv venv，无需 pip）
#   - /home/imc/yzy/agent/.secrets/deepseek.env 存在（DeepSeek key，不入 git）
#   - /media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json 或可重新采集
#
# 用法:
#   bash scripts/run_loop.sh [--skip-baseline] [--round 3]
#
# 说明: 每步产物已存在时自动跳过（幂等）；四臂优化在 tmux 中后台运行，
#   脚本同步等待全部 round 记录落盘后输出汇总。

set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PY=.venvs/agent-lightning/bin/python
VENVS=(
  vendor/tau2-bench/src
  vendor/agent-lightning
)
PYTHONPATH=$(IFS=:; echo "${VENVS[*]}")
export PYTHONPATH

SKIP_BASELINE=0
ROUND=3
for arg in "$@"; do
  case "$arg" in
    --skip-baseline) SKIP_BASELINE=1 ;;
    --round=*) ROUND="${arg#*=}" ;;
    --round) shift ;;
  esac
done

RESULTS_JSON=/media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json
SECRETS=/home/imc/yzy/agent/.secrets/deepseek.env
DATASETS=data/datasets
DIAG_SUMMARY=data/diagnostics/summary.json
BASELINE_VAL=runs/baseline_val_rerun.json

step() { echo; echo "══════════ $1 ══════════"; }

# 0. 环境检查
step "0/6 环境检查"
[ -f "$SECRETS" ] || { echo "!! 缺少 $SECRETS"; exit 1; }
[ -x "$PY" ] || { echo "!! 缺少 venv $PY"; exit 1; }
set +u; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; set -u

# 1. 轨迹采集（M1 基线，40 任务）
step "1/6 轨迹采集（基线 retail40）"
if [ "$SKIP_BASELINE" = "0" ] && [ ! -f "$RESULTS_JSON" ]; then
  $PY scripts/run_tau2_baseline.py --task-ids 0-39 --name retail40-v1 \
    --seed 301 --max-concurrency 2 --max-steps 80 --num-trials 1
else
  echo "跳过: 基线产物已存在（$RESULTS_JSON）"
fi

# 2. 数据划分（dev/val/holdout，幂等）
step "2/6 数据划分（partition）"
if [ ! -f "$DATASETS/partition_manifest.json" ]; then
  $PY data/partition.py --manifest "$DATASETS/task_manifest.json" \
    --out "$DATASETS" --dev 0.6 --val 0.2
else
  echo "跳过: 划分已锁定（$(cat $DATASETS/partition_manifest.json | grep -o '"hash": "[^"]*"' | head -1)）"
fi

# 3. 失败诊断（AgentRx）
step "3/6 失败诊断（AgentRx）"
if [ ! -f "$DIAG_SUMMARY" ]; then
  echo "!! 缺少 $DIAG_SUMMARY——手动执行 diagnose_baseline_failures.py 后重试"
  echo "    （该步输出在数据盘 run-dir，需人工确认）"
  exit 1
else
  echo "跳过: 诊断已存在（$(python3 -c "import json;d=json.load(open('$DIAG_SUMMARY'));print(f'{d[\"num_diagnosed\"]} 条失败诊断')")）"
fi

# 4. 基线 val 对照（×3 多数票，r3 协议）
step "4/6 基线 val8 对照重跑（×3）"
if [ ! -f "$BASELINE_VAL" ]; then
  $PY scripts/run_baseline_val_rerun.py --repeats 3 --out "$BASELINE_VAL"
else
  python3 -c "
import json; r = json.load(open('$BASELINE_VAL'))
print(f\"跳过: 已有基线 val 多数票 {r['majority_rate']:.3f}（repeats={r['repeats']}）\")"
fi

# 5. 四臂优化闭环（tmux 后台 + 等待）
step "5/6 四臂优化闭环（round $ROUND，tmux）"
bash scripts/restart_all_arms_r3.sh || bash scripts/restart_all_arms.sh
echo "四臂已启动，等待 round${ROUND}.json 全部落盘 ..."

ARMS=(loop-apo-apo-plain loop-apo-apo-diagnosis loop-gepa-gepa-plain loop-gepa-gepa-diagnosis)
TIMEOUT_S=$((3600 * 4))   # 4 小时上限（APO 臂约 1.5h/轮）
WAITED=0
while [ "$WAITED" -lt "$TIMEOUT_S" ]; do
  MISSING=()
  for arm in "${ARMS[@]}"; do
    [ -f "runs/$arm/round${ROUND}.json" ] || MISSING+=("$arm")
  done
  if [ ${#MISSING[@]} -eq 0 ]; then
    echo "全部完成: ${#ARMS[@]} 臂 round${ROUND}.json 已落盘（等待 ${WAITED}s）"
    break
  fi
  sleep 60
  WAITED=$((WAITED + 60))
done
if [ "$WAITED" -ge "$TIMEOUT_S" ]; then
  echo "!! 超时（${TIMEOUT_S}s）仍有未完成: ${MISSING[*]:-全部}"
  exit 1
fi

# 6. 汇总
step "6/6 汇总"
for arm in "${ARMS[@]}"; do
  R="runs/$arm/round${ROUND}.json"
  python3 -c "
import json; r = json.load(open('$R'))
g = r['gate']
print(f\"{r['arm']:24s} 内部val={r.get('best_internal_val_score'):.3f} 重跑x{r.get('val_repeats',1)}多数票={r.get('val_rerun_success_rate'):.3f} \"+
      f\"gate={'接受→v'+str(r['new_version']) if g['accept'] else '拒绝'}({g['reason'][:40]})\")"
done
echo
echo "完成。报告: reports/ablation_2026-08-08.md（r3 结果追加）"
