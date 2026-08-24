# OpenSearch-VL 官方协议与本地 Provider 适配契约

> 日期：2026-08-24
> 状态：CPU 合约与真实 WIT replay 已通过；尚未接入官方 SFT-36K 或 RL optimizer。

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

下载官方 SFT-36K 后，先逐条审计其 `tools` schema 是否与本契约一致，并用真实 tokenizer 验证一个不超过
1,000 条的固定分层子集。任何字段差异先记录和兼容，不批量改写官方原始数据。
