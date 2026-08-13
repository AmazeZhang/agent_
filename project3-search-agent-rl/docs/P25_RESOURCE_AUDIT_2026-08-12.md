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

## 7. 下载传输诊断

第一次下载使用环境内`huggingface_hub==0.34.4`，客户端自动进入Xet路径，但首个40GB分片
约一分钟保持0 bytes。尝试设置`HF_HUB_DISABLE_XET=1`仍进入`xet_get`；该版本常量中没有
此开关，因此环境变量无效。两次尝试均由当前前台会话Ctrl-C终止，退出码130，未生成完成标记。

对固定revision地址执行只读HTTP Header检查得到302后200，CDN正确返回
`Content-Length: 42949672960`、`Accept-Ranges: bytes`和预期LFS ETag。下载器因此增加显式
`--transport curl`默认路径：

- 写入`<filename>.partial`；
- `curl --continue-at -`支持中断续传；
- 仅curl成功后原子改名为最终文件；
- 最终文件仍必须通过Manifest大小和SHA256检查；
- Hugging Face客户端路径保留为可选回退，不再作为本机默认。

curl试传32秒得到约303MiB，稳定在约10MiB/s，随后为切换到tmux而正常中断。该partial文件
将被后续tmux会话续传，不重复下载已完成字节。

## 8. 首次tmux下载退出与恢复方案（2026-08-13）

首次tmux续传从323,710,976 bytes开始，推进至6,561,538,522 bytes后，CDN连接反复出现
`OpenSSL SSL_read unexpected eof`。curl内部重试10次耗尽并以56退出，Python随之退出，
tmux会话自然结束。验收时：

- `part_aa.partial`保留6,561,538,522 bytes，约占完整`part_aa`的15.3%；
- 没有最终`part_aa`，没有`download-complete.json`；
- `part_ab`、Corpus和E5模型尚未开始；
- 没有下载进程或GPU占用；data盘仍约3.1TiB可用。

处理方式是恢复而不是删除重下。下载器新增有界外层恢复循环：

1. 每轮重新请求固定revision URL，避免复用失效的CDN签名或连接；
2. 每轮从`.partial`当前精确字节执行Range续传；
3. 记录attempt、curl退出码、续传前后字节；
4. 只要字节增加就重置“无进展”计数；
5. 默认最多100轮、连续5轮无字节增长才失败，并采用最高60秒退避；
6. 已知大文件在达到Manifest精确大小后才原子改名；
7. 全部资源下载完仍逐文件计算SHA256，未通过时不生成完成标记。

恢复会话继续设置`CUDA_VISIBLE_DEVICES=''`，不使用任何GPU。新日志追加到原日志，保留首次
失败证据而不覆盖。
