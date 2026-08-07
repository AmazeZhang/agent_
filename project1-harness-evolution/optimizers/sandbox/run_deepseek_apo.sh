#!/usr/bin/env bash
# M0: 在 DeepSeek 上跑通 Agent Lightning 官方 APO 示例（room_selector）
# 用法: bash run_deepseek_apo.sh
# 约束: 必须在 tmux 中运行; 不写任何密钥到日志; 输出到 runs/loop-sandbox/
set -euo pipefail

# 定位
WORKSPACE="/home/imc/yzy/agent"
PROJ1="$WORKSPACE/project1-harness-evolution"
SECRETS="$WORKSPACE/.secrets/deepseek.env"
VENV="$PROJ1/.venvs/agent-lightning"
SANDBOX="$PROJ1/optimizers/sandbox"
OUT="$PROJ1/runs/loop-sandbox"

[ -f "$SECRETS" ] || { echo "缺少 $SECRETS"; exit 1; }

# tmux server 全局环境带有 socks 代理（httpx 不认 socks scheme 会崩）;
# DeepSeek 可直连（已实测 401 可达），对本进程清掉代理变量，不动 tmux 全局环境
unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy FTP_PROXY ftp_proxy

# 加载密钥与模型配置（只加载环境变量，不回显）
set -a; source "$SECRETS"; set +a
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-flash}"
# poml 已复制进 venv site-packages（uv 网络下载会卡，见 runs/loop-sandbox 记录）

mkdir -p "$OUT"
cd "$SANDBOX"

echo "==> 模型: $DEEPSEEK_MODEL"
echo "==> 输出目录: $OUT"
echo "==> 启动 APO 训练 ($(date '+%F %T'))"

source "$VENV/bin/activate"
exec python -u room_selector_apo.py 2>&1 | tee "$OUT/apo_console.log"
