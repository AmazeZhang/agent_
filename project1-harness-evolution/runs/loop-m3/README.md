# M3 APO 闭环记录（2026-08-07）

## 两臂启动

- apo-plain: tmux `p1-m3-apo-plain`（19:24 启动）
- apo-diagnosis: tmux `p1-m3-apo-diag`（19:40 启动，加载 4 条诊断）

## 问题 1（已修）: adapt KeyError('role')——两臂同崩

**现象**: 两臂都在完成 val 集 seed 评测（8/8 仿真）后崩溃：
`TraceToMessages.adapt → convert_to_openai_messages → KeyError: 'role'`。

**根因（scripts/debug_adapt.py 定位，AgentOps 0.4.21 插桩三处缺陷）**:
1. 带 tool_calls 的 assistant 消息**漏记 role** 属性
   （有 `gen_ai.prompt.N.content` + `tool_calls.*`，无 `role`）
2. tool 消息（带 tool_call_id）也可能漏记 role
3. 多 tool_calls 的 assistant 消息，**最后一个 call 漏记 id**
   （有 arguments/name 无 id）

**修复**: vendor/agent-lightning/.../adapter/messages.py 的
`convert_to_openai_messages` 容错：
- 缺 role 时按内容判定：tool_calls/name → assistant；tool_call_id → tool；
  无法判定 → 跳过该消息
- tool_calls 的 call 缺 id → 兜底空字符串
脚本 scripts/debug_adapt.py 留作回归定位工具（DebugAdapter dump span 属性）。

## 验证

- debug_adapt 复现（console.log 原始 dump）→ 逐缺陷修复重跑：
  console2.log（role 判定）→ console3.log（tool 消息 role）→ console4.log（tool_call id）
- 最终验证通过后重启两臂（同配置）

## 环境备注

- 启动命令必须 unset 代理（httpx 不支持 socks:// 代理 URL）：
  `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy`
- DeepSeek env 用绝对路径：`source /home/imc/yzy/agent/.secrets/deepseek.env`
