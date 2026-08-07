# patches/ — vendor 依赖修复补丁

vendor 子模块指向上游官方仓库（microsoft/agent-lightning 等），工作区修复
不推送到上游，统一以 patch 文件保存在此，保证实验可复现。

## agentlightning-m3-agentops-role-fix.patch

- 修复对象: `vendor/agent-lightning/agentlightning/adapter/messages.py` 的
  `convert_to_openai_messages`
- 背景: AgentOps 0.4.21 插桩三处缺陷导致 APO 训练在 val 评测后崩溃
  （KeyError('role') / KeyError('id') / KeyError('name')）：
  1. 带 tool_calls 的 assistant 消息漏记 role
  2. 带 tool_call_id 的 tool 消息漏记 role
  3. 多 tool_calls 消息的最后一个 call 记录不完整（id/name/arguments 缺失）
- 修复方式: 适配层按消息内容推断 role，call 字段 .get() 兜底（详见
  `runs/loop-m3/README.md` 的 5 轮验证记录）
- 应用方式:
  ```
  cd vendor/agent-lightning
  git apply ../../patches/agentlightning-m3-agentops-role-fix.patch
  ```
