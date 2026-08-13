# P2.5-B真实Retriever资源下载完成报告

- 完成时间：2026-08-13 13:24:42（Asia/Shanghai）
- 验收时间：2026-08-13 16:09（Asia/Shanghai）
- 状态：下载完成，四项关键大文件大小与SHA256全部通过
- GPU：全程设置`CUDA_VISIBLE_DEVICES=''`，未使用任何GPU
- 训练：未开始

## 1. 固定资源

| 资源 | Revision | 文件 | Bytes | SHA256 |
|---|---|---|---:|---|
| `PeterJinGo/wiki-18-e5-index` | `a4d31160a035f30764604f4827cd8f1d0315eb86` | `part_aa` | 42,949,672,960 | `a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023` |
| 同上 | 同上 | `part_ab` | 21,609,402,413 | `b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11` |
| `PeterJinGo/wiki-18-corpus` | `69c1c00ffe7c5554c68d8548355cb22e46aabc51` | `wiki-18.jsonl.gz` | 5,123,307,260 | `7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db` |
| `intfloat/e5-base-v2` | `f52bf8ec8c7124536f0efb74aca902b2995e5bcd` | `model.safetensors` | 437,955,512 | `d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693` |

完整E5模型还包括固定revision下的Config、Tokenizer、Pooling和README文件。完成标记位于：

```text
/media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5/download-complete.json
```

## 2. 下载过程与恢复

1. Hugging Face Xet路径在首分片保持0 bytes，前台安全中止；
2. 改用支持Range的标准HTTPS/curl，先验证可续传；
3. 首次tmux运行因TLS unexpected EOF在6,561,538,522 bytes退出；
4. 下载器增加有界外层恢复循环、连续无增长熔断和逐轮字节审计；
5. 从精确断点恢复，没有删除或重新下载已有partial；
6. 每个文件先写`.partial`，达到预期大小后才原子改名；
7. 全部下载完成后由下载器流式计算SHA256，全部通过后才写完成标记。

原始追加日志保存在数据盘：

```text
/media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5/download.log
```

## 3. 验收状态

- 资源目录占用约66GB；
- data盘验收时约3.1TiB可用；
- tmux下载会话已自然结束；
- 无Python、curl或下载子进程残留；
- 没有拼接索引、解压Corpus或启动Retriever；
- 两个PeterJinGo数据仓库许可证仍为未声明，资源不进入Git且不重新分发。

## 4. 下一门禁

P2.5-C将保留全部源分片和压缩包，另外生成：

```text
prepared/e5_Flat.index
prepared/wiki-18.jsonl
prepared/prepare-complete.json
```

准备过程必须使用临时目标和原子改名，拒绝覆盖现有完成产物，并验证：

- 拼接索引字节数等于两个分片之和；
- 压缩Corpus可完整解码；
- 每行是合法JSON且具备检索服务需要的字段；
- 最终记录索引SHA256、Corpus SHA256、Corpus行数、耗时与磁盘余量。
