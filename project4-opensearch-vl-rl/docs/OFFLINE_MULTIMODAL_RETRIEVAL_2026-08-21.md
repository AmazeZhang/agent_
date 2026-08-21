# OpenSearch-VL 本地多模态检索研究与接入决策

> 日期：2026-08-21  
> 状态：RL-8K JSONL/图片已完成校验和逐图审计；OVEN 当前账号无权限，转向公开 WIT；尚未启动 RL。
> 目标：判断 OVEN/WIT 是否适合作为低成本 RL 工具后端，以及是否需要改写 RL 数据。

## 1. 已固定的数据源

### Search-VL-RL-8K

- 仓库：`OpenSearch-VL/Search-VL-RL-8K`
- 固定 revision：`8ef567289043eef004b13da83b0e7bb7f5ae2daa`
- 官方仓库 used storage：2,698,065,384 B
- 已下载并校验：
  - `rl_data.jsonl`：8,933,719 B
  - SHA256：`3af5b3c188e604817b6c4731c369b7ca1f829414fc514d109a7f468cfd8b0144`
- 已完成：`images.zip`，2,693,241,993 B，整体 SHA256
  `589a67c263c8dcd9697bc762df3d3d6cc5b369017b1f196186a69d23142f4236` 与官方 LFS 一致；
  使用 8 路有界 Range、可恢复 part 和每路自动重试，校验通过后才发布最终 ZIP。
- ZIP CRC/安全审计、非覆盖解压和 7,992 张逐图 decode 全部通过；图片为 4,026 JPEG、
  3,149 PNG、817 WEBP，7,992 个 JSONL 引用均唯一且存在。

下载只使用固定 revision 的 `hf-mirror.com` 直连，显式清空大小写代理变量，未使用 Clash
7890/7891，且禁用了隐式 HF token 和 Xet。

### OVEN

- 仓库：`ychenNLP/oven`
- 固定 revision：`0f1568c5bcb16b70b189d77164c6bcdc38ee43c2`
- 访问状态：HF `gated=auto`；2026-08-21 使用官方 HF 端点检查，本机现有凭据请求 gated
  下载脚本返回 HTTP 401。当前不得下载，也不得借镜像绕过条款或泄露 token。
- 目标资产：
  - `all_wikipedia_images.tar`：32,353,424,368 B；
  - `ovenid2impath.csv`：529,959,904 B。
- 暂不下载 `shard01.tar` 至 `shard08.tar`；八个 shard 合计约 261 GB，属于 OVEN 查询图像，
  不是建立第一版 Wikipedia 实体候选库的必要条件。

### WIT / Wikimedia Image-Caption Matching

- 采用公开 HF 仓库 `wikimedia/wit_base`，固定 revision
  `ff6d4fb32fd566d3a1fa20e946cba3234179465e`，许可为 CC-BY-SA-4.0；无需搜索 API key。
- 固定 revision 的 `data/` 共 330 个 Parquet 分片、308,150,150,366 B；单片约
  910–969 MB。文件清单通过公开 API 分页核对，未下载数据本体。
- 数据卡给出 6,477,255 个样本；每行带 300px 图像、Wikimedia 元数据、多语言描述/上下文，
  以及 2,048 维 ResNet-50 特征，适合先验证本地图像近邻和实体候选链路。
- 第一片 `data/train-00000-of-00330.parquet` 为 932,699,916 B，LFS SHA256
  `8a9a449a0db937920b7c0dd13cd0bc9b1cadb19071567169dbac41a956273ec2`。
- OVEN 权限恢复前，WIT 从“第二阶段补召回”提升为首个公开视觉候选库；先做单片 pilot，
  索引与 observation 验收后才考虑 308 GB 全量下载，避免先下载再发现 schema/性能不适配。

## 2. RL 数据的实测组成

仓库脚本 `scripts/audit_rl_dataset.py` 对固定 JSONL 做流式审计，报告保存在数据盘：

`datasets/manifests/search-vl-rl-8k-audit-8ef5672.json`

结果：

| 子集 | 行数 | 占比 |
|---|---:|---:|
| `new_livevqa` | 3,746 | 46.87% |
| `WebQA` | 1,507 | 18.86% |
| `demo_1k` | 1,000 | 12.51% |
| `wiki_zh` | 527 | 6.59% |
| `wiki_en` | 406 | 5.08% |
| `palace` | 369 | 4.62% |
| `wikiart` | 253 | 3.17% |
| `new_fvqa` | 184 | 2.30% |

其他事实：

- 7,992 行全部包含 `question`、`answer`、`images` 和 `dataset`；每行恰好一张图；
- 7,992 个图像引用全部唯一；问题有 1,144 个重复行，但对应图片不同；
- 719 个问题含 CJK 字符，7,273 个不含；
- answer 平均 845 字符，p90 1,534 字符，3,775 个 answer 含 `\\boxed{...}`；
- 原始数据没有实体 QID、原始网页 URL、gold evidence 或预计算检索结果。

这推翻了“RL-8K 全部可以由 Wikipedia 图像库完整覆盖”的过强假设。OVEN/WIT 与
`wiki_*`、`wikiart`、`palace` 高度匹配，也可能覆盖部分 WebQA/FVQA，但无法在没有实测的情况下
替代 LiveVQA 和开放网页检索。

## 3. 适用性结论

OVEN/WIT **适合做本地优先的视觉实体检索后端**，不适合直接宣称等价替代论文的开放网络工具。

推荐工具路由：

```text
image_search
  -> 当前：WIT 公开图像/特征索引
  -> OVEN 获得正规访问权后：OVEN 实体索引优先、WIT 补召回
  -> 仍低置信度时返回可审计的 no-match
  -> 只有得到独立授权和预算时才允许在线 fallback

text_search
  -> Wiki-18 BM25 / 后续较新 Wikipedia 索引
  -> 无证据时返回 no-match
  -> 只有得到独立授权和预算时才允许在线 fallback
```

第一阶段验收不使用答案做检索或筛选，避免泄漏。覆盖率报告必须按数据子集分层，不能只报告
总体数字掩盖 LiveVQA 与 Wiki 子集差异。

## 4. RL 数据是否修改

### 主复现数据：不修改

保留官方 `question`、`answer`、`images`、`dataset` 和固定 90/10 划分逻辑。不得把以下内容写回
原始样本：

- OVEN QID 或预测实体；
- WIT top-k；
- gold evidence；
- 本地检索是否命中；
- 根据答案判定的“可离线检索”标签。

这些信息只能进入独立 trajectory/run 日志，否则会造成答案泄漏、选择偏差，并破坏与官方数据
的一致性。

### 工程 pilot：允许派生清单，但不伪装成原数据

可以创建只含样本 ID 的 `offline-pilot` 清单，用于 rollout/工具工程 smoke。清单的选择只能基于
预先声明的数据来源或与答案无关的检索置信度，并必须报告它不是完整 RL-8K 结果。

### 实际需要修改的部分

1. 工具后端：保留 `image_search` / `text_search` 名称和参数，替换 provider；
2. observation 格式：统一为稳定的 title/source/summary/entity-id/similarity 字段；
3. 系统提示：去除“Serper + Jina + Qwen”硬编码，准确描述当前 backend；
4. reward：无 judge 时当前 `r_query=0`，需本地 judge 或作为显式消融；
5. 日志：记录 corpus revision、index SHA256、top-k、阈值、延迟、cache hit 和 fallback 次数。

## 5. 下一步门禁

1. 下载并校验 WIT 单个约 0.93 GB 分片，验证 Parquet schema、图片和预计算特征；
2. 把真实 WIT 字段接入已实现的离线索引适配器和稳定 observation schema；
3. 使用已固定的 200 条清单，按八个 dataset 子集分层报告检索覆盖率；
4. 根据覆盖率与索引成本决定是否分批扩展 WIT，不能把单片 pilot 冒充完整语料；
5. OVEN 只有在用户正规接受条款、凭据授权后才恢复下载；初期仍不下载八个 query shard；
6. 任何 20 步以上 RL、WIT 全量 308 GB 下载或在线 fallback 均需重新获得用户确认。

## 6. 已实现的本地检索契约

`local_retrieval/visual_index.py` 已实现小语料 pilot 使用的只读精确余弦索引：

- 建库时校验有限值、零范数、向量/元数据数量以及 `title/source/entity_id` 必填字段；
- 向量归一化后保存为可 memory-map 的 `.npy`，元数据与 corpus revision 分离保存；
- 输出目录拒绝覆盖，通过同盘 staging 原子发布；
- 查询限制 `top_k` 为 1–50，支持显式相似度阈值，稳定返回
  `title/source/summary/entity_id/similarity/corpus/corpus_revision`；
- `tool_observation()` 输出带 `Tool execution result:` 前缀的稳定 JSON，可接到原有
  `image_search` 轨迹格式；没有匹配时返回 `match_count=0`，不会伪造实体。

当前实现是 `numpy-exact-cosine.v1`，只用于单片 WIT 或更小的 schema/工具 pilot。它不是 647 万
样本的最终检索引擎；全量阶段需要引入分块或近似索引，并重新记录索引构建参数与召回验证。
当前也没有擅自定义查询图像的 ResNet-50 预处理，必须在读取真实 WIT shard 的 schema/数据卡字段后
与发布特征严格对齐。

本地文本侧已实现 `local_retrieval/text_index.py`：

- 使用 SQLite FTS5 `unicode61` 构建固定 corpus/revision 的实体证据索引；
- 建库校验 `entity_id/title/source/text`，拒绝重复实体、空语料和覆盖已有索引；
- 支持参数化的实体 ID 精确 lookup 和经过安全 token 化的全文检索，不接受原始 FTS 表达式；
- 运行时以 `mode=ro&immutable=1` 打开，返回与视觉侧一致的 title/source/summary/entity-id、
  corpus 与 revision 字段；
- no-match 返回空列表或 `None`，不调用网络、不伪造页面内容。

当前只通过合成 Wikipedia 文档验证接口；真实 Wikipedia 语料 revision、许可、字段映射和按实体
split 仍需在数据 pilot 中固定。
