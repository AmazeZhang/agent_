#!/usr/bin/env bash
# 重启项目一四臂消融（第三轮 r3，2026-08-08 协议修正后）
# r3 修正（对照 r2）:
#   1. gate 基线参照 = 基线在 val8 实测多数票（scripts/run_baseline_val_rerun.py 产物）
#   2. val 独立重跑 ×3 按任务多数票计成功率（LLM 非确定性降噪）
#   3. GEPA 反思注入真实任务内容 + 身份保真约束（防 r2 候选退化）
#   4. APO 加大 beam 预算（--beam-rounds 3）主攻诊断臂
# 前置: 先完成 scripts/run_baseline_val_rerun.py（runs/baseline_val_rerun.json）
# 用法: bash scripts/restart_all_arms_r3.sh
set -e
cd "$(dirname "$0")/.."

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [ ! -f "runs/baseline_val_rerun.json" ]; then
    echo "!! 缺少 runs/baseline_val_rerun.json——先运行:"
    echo "   .venvs/agent-lightning/bin/python scripts/run_baseline_val_rerun.py --repeats 3"
    exit 1
fi

for arm in apo-plain apo-diagnosis; do
    mkdir -p "runs/loop-apo-apo-$arm"
    : > "runs/loop-apo-apo-$arm/console_r3.log"
done
for arm in gepa-plain gepa-diagnosis; do
    mkdir -p "runs/loop-gepa-gepa-$arm"
    : > "runs/loop-gepa-gepa-$arm/console_r3.log"
done

tmux kill-session -t p1-m3-apo-plain 2>/dev/null || true
tmux kill-session -t p1-m3-apo-diag 2>/dev/null || true
tmux kill-session -t p1-m4-gepa-plain 2>/dev/null || true
tmux kill-session -t p1-m4-gepa-diag 2>/dev/null || true

# APO 臂: 各用独立 LightningStoreServer 端口（AGL_SERVER_PORT），避免绑定竞争
tmux new-session -d -s p1-m3-apo-plain \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
       --arm apo-plain --round 3 --feedback plain \
       --beam-rounds 3 --val-repeats 3 \
       > runs/loop-apo-apo-plain/console_r3.log 2>&1"

tmux new-session -d -s p1-m3-apo-diag \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   export AGL_SERVER_PORT=4748; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
       --arm apo-diagnosis --round 3 --feedback diagnosis \
       --beam-rounds 3 --val-repeats 3 \
       > runs/loop-apo-apo-diagnosis/console_r3.log 2>&1"

# GEPA 臂
tmux new-session -d -s p1-m4-gepa-plain \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_gepa.py \
       --arm gepa-plain --round 3 --val-repeats 3 \
       > runs/loop-gepa-gepa-plain/console_r3.log 2>&1"

tmux new-session -d -s p1-m4-gepa-diag \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_gepa.py \
       --arm gepa-diagnosis --round 3 --val-repeats 3 \
       > runs/loop-gepa-gepa-diagnosis/console_r3.log 2>&1"

echo "==> 四臂 r3 已重启:"
tmux ls | grep p1-m[34]
