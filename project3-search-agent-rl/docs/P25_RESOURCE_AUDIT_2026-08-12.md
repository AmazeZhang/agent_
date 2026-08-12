# P2.5真实Retriever资源审计

- 审计日期：2026-08-12
- 状态：来源与预算门禁完成；尚未下载大文件
- 目标：用真实Wikipedia Corpus替换Ground-Truth衍生Fixture
- GPU状态：本步骤未使用GPU

## 1. 上游实际资源链路

`verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`的Search示例使用：

```text
PeterJinGo/wiki-18-e5-index: part_aa + part_ab
                         -> e5_Flat.index
PeterJinGo/wiki-18-corpus: wiki-18.jsonl.gz
                         -> wiki-18.jsonl
intfloat/e5-base-v2      -> query encoder
                         -> POST /retrieve
```

上游`examples/search/searchr1_download.py`虽然声明了`--repo_id`参数，实际代码仍硬编码两个
PeterJinGo仓库，并且没有固定revision。项目脚本不能直接照搬。

## 2. 固定来源和体积

Hugging Face API在2026-08-12返回：

| 资源 | Revision | 文件 | Bytes | LFS SHA256 |
|---|---|---|---:|---|
| `PeterJinGo/wiki-18-e5-index` | `a4d31160...` | `part_aa` | 42,949,672,960 | `a8a6a246...3023` |
| 同上 | 同上 | `part_ab` | 21,609,402,413 | `b6d9bc94...3c11` |
| `PeterJinGo/wiki-18-corpus` | `69c1c00f...` | `wiki-18.jsonl.gz` | 5,123,307,260 | `7abd9292...17db` |
| `intfloat/e5-base-v2` | `f52bf8ec...` | `model.safetensors` | 437,955,512 | `d0d559c4...f693` |

索引分片合计64,559,075,373 bytes。上游文档给出的整体估算为下载60–70GB，解压和拼接后
约132GB。本项目保留原分片与压缩Corpus以支持核验和恢复，因此按至少250GiB空闲空间门禁。

机器实测：

- `/media/imc/data`可用3,408,163,352,576 bytes；
- 系统可用内存约995GiB；
- 96逻辑CPU，双路NUMA；
- 当前项目索引目录近乎为空。

容量和CPU内存足以先运行CPU FAISS，不需要为了P2.5占用GPU。

## 3. 许可证边界

- `intfloat/e5-base-v2`的仓库Metadata声明MIT；
- 两个PeterJinGo数据仓库的Metadata均未声明许可证，仓库也没有README或LICENSE文件；
- 不从Wikipedia的一般许可反推该打包数据和索引的再分发许可；
- 当前仅用于内部复现，不把Corpus、索引或其内容提交到Git或重新分发；
- 秋招报告必须标记“上游数据仓库许可证未声明，使用范围待人工核验”。

这不阻止内部技术门禁，但阻止我们把数据产物当作自有可分发资产。

## 4. 上游方案的本机风险

1. 上游Dense Retriever在`faiss_gpu=True`时调用`index_cpu_to_all_gpus`，会使用所有
   `CUDA_VISIBLE_DEVICES`内的卡；若未隔离，可能占用承载桌面的GPU0。
2. 上游用`cat part_* > e5_Flat.index`，没有先验哈希、磁盘门禁或目标存在检查。
3. 上游用`gzip -d`删除压缩源，不利于中断恢复和证据保留。
4. 上游下载脚本不固定revision，未来同名文件可能漂移。
5. 现有`paretotool-retriever`虽有`faiss-cpu==1.8.0`，但缺`datasets`，且Python导入到了
   用户级不同版本的Transformers/Uvicorn，环境不够隔离。

## 5. 本项目的安全策略

1. 使用`configs/resource_manifests/searchr1_retriever_sources.json`固定revision、大小和SHA256；
2. 下载脚本只下载与验哈希，不拼接、不解压、不删除；
3. 完成标记`download-complete.json`存在时拒绝覆盖；中断时允许Hugging Face续传；
4. 下一门禁采用保留源文件的拼接与`gzip -dk`等价操作，并验证JSONL行数和FAISS维度；
5. 建立独立CPU Retriever环境并禁用用户site-packages；
6. 首轮只跑CPU FAISS，设置`CUDA_VISIBLE_DEVICES=''`，从机制上无法触碰GPU0；
7. GPU Retriever仅在CPU协议通过后考虑，并且必须通过项目GPU锁显式选择非0物理卡；
8. 训练不会使用Ground-Truth Fixture。

## 6. 下一门禁

运行固定下载脚本，将约70GB源文件写入：

```text
/media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5
```

下载完成后核验四个大文件SHA256，记录耗时、实际占用和剩余磁盘。只有该门禁通过，才执行
索引拼接、Corpus解压和CPU Retriever环境构建。
