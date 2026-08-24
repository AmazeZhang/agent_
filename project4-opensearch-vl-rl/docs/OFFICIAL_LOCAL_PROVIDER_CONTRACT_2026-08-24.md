# OpenSearch-VL 官方协议与本地 Provider 适配契约

> 日期：2026-08-24
> 状态：CPU 合约、官方 960-safe SFT-50 与 QVA 字节不变的 v10 rollout 适配已通过；尚未启动 RL optimizer。

## 目标

模型、SFT 轨迹、RL rollout 和推理始终使用官方 `image_search` / `text_search` 接口。本地图片库、
Wikipedia 索引、缓存和故障注入全部隐藏在执行层，不创建模型可见的 `text_lookup(entity_id)` 等私有工具。

```text
official tool call
  -> OfficialLocalSearchProvider
  -> frozen image/Wikipedia backend
  -> official-shaped observation
```

## 模型可见契约

- `image_search({"url": "img_1"})`：只接受已注册的 `img_N` 引用；离线 provider 不访问任意 URL。
- `text_search({"q": "...", "hl": "en", "top_k": 5})`：兼容官方 `query`/`lang` 别名，`top_k` 限制
  为 1～20。
- image observation 只包含编号、标题和来源。
- text observation 保持官方 `[Passage N] / Title / URL / Summary` 布局。
- 空结果和错误使用稳定的工具消息；模型看不到异常栈、文件路径、SQLite、内部实体 ID、相似度、corpus
  revision 或 provider 名称。

## 本地执行与安全边界

- 本地图片引用由 Run 环境注册；当前实现拒绝 HTTP(S)、绝对路径和未注册形式。
- 文本搜索只读打开固定 revision 的 SQLite FTS5 索引。
- provider 不修改 vendor；上游 commit 保持 `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0`。
- 后续 SFT 使用已记录 observation，不调用 provider；rollout/推理通过本适配层执行。
- v7/v8 的私有协议继续作为历史实验保留，但不再作为官方 SFT→RL 主链路。

## 验收证据

- `tests/test_official_local_provider.py` 覆盖官方字段、别名、URL fail-closed、内部字段隐藏、schema 深拷贝
  和 backend 异常脱敏。
- 真实 WIT 首条任务 replay：`image_search(url=img_1)` 返回 3 个 title/source，随后
  `text_search(q="Genny Lim", hl="en", top_k=3)` 返回 Wikipedia Passage；输出中不存在
  `entity_id/similarity/local_*_index/corpus_revision`。
- 全部测试 CPU-only、无网络、无 API、无 GPU。

## 下一门禁

受管 GPU1 上先运行冻结的 `official_provider_rollout_gate_v1.json` phase A 单题 greedy。只有首个工具为
`image_search`、至少一次工具调用且无 fatal，才允许 phase B 四类 train task、每题 4 条随机 rollout；
phase B 通过预声明方差/格式/fatal 门后，才允许另行审查 1-step RL。

## 2026-08-25 适配审计

- 官方 960-safe SFT 的 960 行 `system` 与 `tools` 分别各只有一个 SHA256；模型学习的是
  `image_search({"url":"img_1"})` 和 `text_search({"q":...})`。
- 历史 evaluator 却把官方分支接成 `image_search({"image":"img_1"})`，并输出含本地 `entity_id` / similarity
  的 compact observation；随机 rollout 审计器还漏传 `tool_protocol`，会退回 legacy。以上均已修复并由
  CPU tests 覆盖，官方分支现在实际调用 `OfficialLocalSearchProvider`。
- 非覆盖派生集 `wit-rl-official-provider-v10` 保持 v8 QVA `tasks.jsonl` 字节完全一致，SHA256
  `2ccdb0ef507ebbd20dfba54c199a724ab9056a868856d9291dd65949325cce55`，120 条任务/120 张图片；
  `rows_modified=0`。只改变隐藏执行层和 observation schema 声明，manifest SHA256
  `70142a78d31efcd97a041d7b11eb3d727ed964667c22d3941598d889ca01546c`。
- 首次 v9 派生发布继承了未复制文件的旧 `sft_sha256` 字段，故不进入实验；builder 已 fail-closed 修复，
  v10 不含悬空 SFT provenance。此步无 GPU、网络或 API。

Phase A Run `official-provider-sft50-rollout-phasea-v2-20260825` 已通过预声明行为门：checkpoint-50 首个
assistant turn 严格输出 `image_search({"url":"img_1"})`，完整工具序列为
`image_search → text_search → text_search → text_search`，与 oracle path 一致，fatal=null。最终 response
格式与标题正确；模型返回了多句证据而非要求的精确首句，因此 evidence_exact/full_success 为 0。该结果
只授权 phase B 方差审计，不当作准确率提升证据。evaluation SHA256
`b9155e96774982cfa8d4907f5d2f45b18c21955ea9b6d649581a8b7d801055f2`；Run `exit_code=0`，GPU1
cleanup 后 18 MiB、无 compute process，GPU0/GPU5 未参与，精确 tmux dead/status 0 后关闭。

Phase B Run `official-provider-sft50-rollout-phaseb-g4n4-20260825` 按冻结配置完成四类 train task × 4
stochastic rollout。3/4 组 reward variance>0，variable group fraction `0.75`；format valid fraction
`0.8125`，fatal fraction `0.1875`，均通过 `0.25/0.75/0.25` 预声明门槛，并存在真实非零 advantage。
rank2 组 4/4 full success 但组内零方差；transient 组 3/4 fatal；no-match 组有 3 条 query-only reward，
这些风险均保留披露。报告 SHA256
`d758cf88815122360139f4b1e0028576a1b4d8e3a5e159e9560fc4d7fbcfb6b0`。Run `exit_code=0`，GPU1
最高观察约 58°C、cleanup 后 18 MiB/无 compute process；GPU0/GPU5 未参与，无网络/API，精确 tmux
dead/status 0 后关闭。该门只授权单独实现并审查的 1-step RL smoke。
