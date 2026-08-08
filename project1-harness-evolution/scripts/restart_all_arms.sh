#!/usr/bin/env bash
# 重启项目一四臂消融（第二轮，2026-08-08 修复后）
# 修复项: CandidateFilter 白名单对齐 seed / GEPA 反射双层包装 / runner 过滤失败落盘
# 用法: bash scripts/restart_all_arms.sh
set -e
cd "$(dirname "$0")/.."

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

for arm in apo-plain apo-diagnosis; do
    mkdir -p "runs/loop-apo-apo-$arm"
    : > "runs/loop-apo-apo-$arm/console.log"
done
for arm in gepa-plain gepa-diagnosis; do
    mkdir -p "runs/loop-gepa-gepa-$arm"
    : > "runs/loop-gepa-gepa-$arm/console.log"
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
       --arm apo-plain --round 2 --feedback plain \
       > runs/loop-apo-apo-plain/console.log 2>&1"

tmux new-session -d -s p1-m3-apo-diag \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   export AGL_SERVER_PORT=4748; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
       --arm apo-diagnosis --round 2 --feedback diagnosis \
       > runs/loop-apo-apo-diagnosis/console.log 2>&1"

# GEPA 臂
tmux new-session -d -s p1-m4-gepa-plain \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_gepa.py \
       --arm gepa-plain --round 2 \
       > runs/loop-gepa-gepa-plain/console.log 2>&1"

tmux new-session -d -s p1-m4-gepa-diag \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_gepa.py \
       --arm gepa-diagnosis --round 2 \
       > runs/loop-gepa-gepa-diagnosis/console.log 2>&1"

echo "==> 四臂已重启:"
tmux ls | grep p1-m[34]
