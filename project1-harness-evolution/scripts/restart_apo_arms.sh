#!/usr/bin/env bash
# 重启 M3 两个 APO 闭环臂（adapt 修复后）
# 用法: bash scripts/restart_apo_arms.sh
set -e
cd "$(dirname "$0")/.."

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

for arm in apo-plain apo-diagnosis; do
    # 清理旧日志（保留原名目录，round 记录独立）
    mkdir -p "runs/loop-apo-$arm"
    : > "runs/loop-apo-$arm/console.log"
done

tmux kill-session -t p1-m3-apo-plain 2>/dev/null || true
tmux kill-session -t p1-m3-apo-diag 2>/dev/null || true

tmux new-session -d -s p1-m3-apo-plain \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
       --arm apo-plain --round 1 --feedback plain \
       > runs/loop-apo-apo-plain/console.log 2>&1"

# 两臂各用独立 LightningStoreServer 端口（AGL_SERVER_PORT），避免绑定竞争
tmux new-session -d -s p1-m3-apo-diag \
  "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
   export AGL_SERVER_PORT=4748; \
   source /home/imc/yzy/agent/.secrets/deepseek.env && \
   PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
   .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
       --arm apo-diagnosis --round 1 --feedback diagnosis \
       > runs/loop-apo-apo-diagnosis/console.log 2>&1"

echo "==> 两臂已重启:"
tmux ls | grep p1-m3-apo
