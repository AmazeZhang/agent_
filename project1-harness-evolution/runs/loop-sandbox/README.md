# M0 沙箱：Agent Lightning + DeepSeek 接入记录

- 日期：2026-08-07
- 目标：跑通官方 APO 示例（room_selector）在 DeepSeek 上的完整一轮训练

## 环境适配发现（重要，M1/M3 会复用）

### 1. pip 与 venv 的坑
- 本机 venv 由 uv 管理，venv 内**没有 pip**（`python -m pip` 报 No module named pip）。
- 裸 `pip install` 会装到用户目录 `/home/imc/.local/lib/python3.10`（Python 3.10），**不是 venv**。
- 正确的安装方式：`uv pip install --python <venv>/bin/python <pkg>`，但**uv 下载会卡住**（网络/代理问题，8 分钟无进展）。
- 本项目应对：把 poml 纯 Python 包**直接复制**进 venv site-packages（poml/、nodejs_wheel*、dist-info，共约 310MB）。不要用 PYTHONPATH 指到整个 3.10 用户目录——会遮蔽 venv 的 tiktoken/regex 等 3.11 包导致循环导入崩溃。

### 2. 代理环境变量
- tmux server 全局环境带 `ALL_PROXY=socks://127.0.0.1:7890/` 等代理变量（服务器启动时注入）。
- httpx/litellm 不认 `socks://` scheme，直接抛 `ValueError: Unknown scheme for proxy URL`。
- DeepSeek API 实测可直连（无代理 401 可达）。
- 对策：启动脚本内 `unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy FTP_PROXY ftp_proxy`，**不改 tmux 全局环境**（其他会话可能依赖）。

### 3. DeepSeek 兼容性（openai SDK 2.53.0 + DeepSeek API）
- ✅ function calling 工具调用正常（`get_rooms_and_availability`）。
- ✅ 推理内容（reasoning_content）正常。
- ✅ `response_format={"type": "json_object"}` 可用，但 **prompt 必须包含 "json" 字样**，否则 400。
- ❌ `chat.completions.parse` + Pydantic `response_format`（json_schema 类型）DeepSeek 不支持——官方示例 judge 用 parse()，已改为 create() + json_object + 手动 `model_validate_json`。
- 模型名：`deepseek-v4-flash`，通过 `DEEPSEEK_MODEL` 环境变量注入（APO 的 gradient_model/apply_edit_model、rollout 模型、judge 模型统一替换）。

## 运行方式

```bash
tmux new-session -d -s p1-m0-apo "bash optimizers/sandbox/run_deepseek_apo.sh"
```

产物：`runs/loop-sandbox/apo_console.log`（tee 全量输出）。

## 状态

- [x] 依赖安装与导入验证
- [x] 首轮运行发现 judge 400（缺 json 字样）→ 已修复重启
- [x] 完整 APO 训练完成（2 轮 beam search，全程 CPU + API，无 GPU）

## 结果（2026-08-07 18:44 完成）

- 基线 val 分：**0.886**（baseline prompt）
- Round 01：best 未更新（0.838 < 0.886，正确拒绝）
- Round 02：best 更新为 **0.966**（+0.080，val 上 28/29 满分）
- 完整循环验证通过：rollout → 文本梯度（deepseek-v4-flash 生成针对性批评，如"要求单一最终选择"）→ 编辑生成候选 → val 评测 → beam 选择
- 官方示例证明 APO 能力可用；本项目不采用示例数据作为实验结果（按调研报告口径）

## 遗留

- 最终优化 prompt 未落盘为独立文件（示例用 InMemoryLightningStore）；M3 需配置持久化 store 或自行导出 best resource。
- 示例级成本未单独统计（AgentOps 会话有记录）；M1 起在自研链路中统计 token/成本。
