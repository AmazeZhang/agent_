# OpenSearch-VL RL API 与资源门禁

> 日期：2026-08-21  
> 当前结论：已到需要用户提供联网搜索/API 方案的停止点；不得启动 RL rollout 或训练。

## 已完成的前置条件

- SFT 工程 smoke 已完成：合成 agentic 数据 1→2→5 step LoRA、断点续训、checkpoint
  加载和离线多模态推理均通过。
- 所有 GPU Run 只使用物理 GPU1；GPU0 未参与，GPU5 未使用；tmux、独立进程组和精确 cleanup
  均正常。
- 上述结果只证明工程链路，不等于 Search-VL-SFT-36K 上的论文级 SFT 复现。

## RL 主路径实际需要的外部能力

### 1. 文本/网页搜索：必须为有意义 rollout 配置

当前主入口注册的是 `TextSearchTool` 和 `WebSearchTool`，实际代码读取：

- 直连 Serper：`SERP_API_KEY`；
- 或 HMAC gateway：`API_GATEWAY_HOST`、`API_GATEWAY_USER`、`API_GATEWAY_KEY`。

注意：根 README 写过 `SERPER_API_KEY`，但 RL 主路径代码读取的是 `SERP_API_KEY`。无 key 时
注册工具仍会向 Serper 发请求，只携带空 key，rollout 将得到 401/错误字符串；这不是可接受的
OpenSearch-VL RL 复现环境。

### 2. Query-utility judge：忠实 reward 需要配置

组合 reward 为：

`r_format * (0.8 * r_accuracy + 0.2 * r_query)`

`r_query` 调用 OpenAI-compatible chat-completions，需要：

- `JUDGE_API_BASE_URL`
- `JUDGE_API_KEY`
- `JUDGE_MODEL`（或 `JUDGE_MODEL_MARKER`）

没有 judge key 时代码允许训练继续，但 `r_query` 固定回退到 0.0。这会改变论文方法的奖励函数，
只能作为消融/退化工程测试，不能称为主复现。

### 3. 页面访问与图像搜索：上游发布仍有缺口

- `VisitTool` 和 Jina reader 实现存在、读取 `JINA_API_KEY`，但主执行器没有注册 `visit`。
- 已注册的 `TextSearchTool` 仅在 gateway 路径尝试 reader；直连 `JINA_API_KEY` 在这条主路径中
  没有被使用。
- 已注册的 `image_search` 依赖仓库未提供的 `upload.py`（把本地图像变成公共 URL）以及
  `env.lens_scan`；当前 vendor 树中两者均不存在。只有 Serper/Jina key 仍不能恢复该工具。
- `layout_parsing` 另需 `LAYOUT_PARSING_API_URL` / `LAYOUT_PARSING_TOKEN`，可作为后续可选工具。

## 必须先修的安全与一致性问题

- 主执行器默认注册 `PythonInterpreter`。现实现使用进程内 `exec`、字符串黑名单和不可终止线程
  超时，不满足项目安全要求；任何 rollout 前必须从默认注册表移除。
- 需要注册并测试 `visit`，统一 SFT/RL/inference 的工具名称与 schema。
- 搜索/reader/image provider 必须有显式 preflight：缺少凭证时启动失败，不能静默用错误结果训练。
- secret 只能通过未跟踪的环境文件或进程环境注入；不得写入 Git、Run 的 `command.txt`、配置、
  stdout/stderr 或状态文档。API 测试先做单请求、限并发、限额和脱敏日志。

## 7×4090 资源现实与下一步边界

- 上游声明 8B RL 最低为 H100/H800/A100 80GB；官方 single-node preset 仍要求 8 GPU，
  rollout TP=4、Megatron train TP=2、70k response、256 prompts × 8 responses。
- 本机按约束排除 GPU0 后最多 7 张 24GB 4090；若不启用历史异常 GPU5，则只有 6 张候选卡。
  官方 preset 不能原样运行，也不能在当前资源上称为论文规模复现。
- 安全路线应为：API 单请求门禁 → 移除危险工具/补齐注册测试 → RL 独立环境 CPU import →
  1 条数据 rollout-only → 缩短序列和批次的 1→resume→5 step 工程 smoke。
- 在用户复核前，不执行多卡大规模 RL；GPU5 若启用仍须单卡压力/NCCL 专项晋级。

## 需要用户决定/提供

最低可继续到“真实联网 rollout”的配置为：

1. 搜索二选一：
   - `SERP_API_KEY`；或
   - `API_GATEWAY_HOST` + `API_GATEWAY_USER` + `API_GATEWAY_KEY`。
2. 忠实 query reward：`JUDGE_API_BASE_URL` + `JUDGE_API_KEY` + `JUDGE_MODEL`。
3. 图像搜索方案：提供现有 `upload.py` / `env.lens_scan` 接口，或授权改造成可用的替代 provider。
4. 是否允许对外部 API 做极小额度 preflight；不会使用 Clash 7890/7891 承载批量流量。

`JINA_API_KEY` 可一并提供，但在使用前必须先修复 `visit` 注册和直连 reader 路径，不能假装当前
主入口已经消费它。
