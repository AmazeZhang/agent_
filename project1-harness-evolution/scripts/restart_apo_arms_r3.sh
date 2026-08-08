#!/usr/bin/env bash
# 重启 APO 两臂 r3（2026-08-08 13:45 双臂因 API 瞬时 402 崩溃）
# 402 根因（2026-08-08 19:2x 确认）: 14:41 四臂并发高峰时 DeepSeek 侧瞬时错误——
# 手动 curl/同步/异步/litellm 全部 200 OK；单独跑两臂（19:15 重启后）0 次 402，运行正常。
# 因此不是代码 bug，带自动重试（最多 4 次，间隔 120s）即可。
# 注意: console 日志路径用 runs/loop-apo-<arm>/（arm 已含 "apo-" 前缀，不能写 loop-apo-apo-）。
# 用法: bash scripts/restart_apo_arms_r3.sh
set -e
cd "$(dirname "$0")/.."
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

tmux kill-session -t p1-m3-apo-plain 2>/dev/null || true
tmux kill-session -t p1-m3-apo-diag 2>/dev/null || true

# 通用: 运行 APO 单臂，崩溃（含 402）时保存现场并重试
run_apo() {  # $1=session  $2=arm  $3=feedback  $4=port
  tmux new-session -d -s "$1" \
    "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; \
     export AGL_SERVER_PORT=$4; \
     source /home/imc/yzy/agent/.secrets/deepseek.env && \
     cd /home/imc/yzy/agent/project1-harness-evolution && \
     for i in 1 2 3 4; do \
       echo \"[attempt \$i] $2\"; \
       PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning \
       .venvs/agent-lightning/bin/python optimizers/run_apo_loop.py \
           --arm $2 --round 3 --feedback $3 --beam-rounds 3 --val-repeats 3 \
           > runs/loop-apo-$2/console_r3.log 2>&1 && exit 0; \
       echo \"[attempt \$i FAILED] 120s 后重试\"; sleep 120; \
     done; echo \"[GIVE_UP] $2 连续 4 次失败\"; exit 1"
}

run_apo p1-m3-apo-diag   apo-diagnosis diagnosis 4748
sleep 45
run_apo p1-m3-apo-plain  apo-plain    plain      4747

echo "==> APO 两臂已重启（带自动重试）:"
tmux ls | grep p1-m3
