# P2.5-C索引拼接与Corpus准备门禁

- 日期：2026-08-13
- 状态：准备器与单元测试完成，尚未执行大文件准备
- 输入：已通过P2.5-B大小/SHA256验收的固定资源
- GPU：本阶段不使用GPU

## 1. 上游封装格式异常

虽然上游文件名为`wiki-18.jsonl.gz`，其解压后的前512 bytes是POSIX TAR Header，不能直接
作为UTF-8 JSONL读取。TAR内唯一普通文件为：

```text
data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
```

成员大小为14,393,573,105 bytes，抽样首行是具有`id`和`contents`字段的合法JSON。下载文件
SHA256与上游LFS SHA完全一致，因此这是上游资源的真实封装，而非下载损坏。

上游文档建议`gzip -d wiki-18.jsonl.gz`，这只会移除外层gzip并留下TAR内容，却把输出命名成
`.jsonl`。随后按JSON加载存在失败风险。本项目不照搬该命令，而是识别并流式读取指定TAR成员。

## 2. 准备器安全属性

`scripts/prepare_p25_retriever_resources.py`执行：

1. 要求`download-complete.json`存在且内容与固定大小/SHA256一致；
2. 再次在复制过程中验证两个索引源分片SHA256；
3. 按明确顺序`part_aa`、`part_ab`流式写入`e5_Flat.index.partial`；
4. 记录拼接后索引SHA256和精确字节，完成后原子改名；
5. 再次验证压缩Corpus SHA256；
6. 要求TAR只有一个普通成员，且名称和大小与审计值一致；
7. 流式抽取成员到`wiki-18.jsonl.partial`，逐行验证UTF-8 JSON及字符串`id/contents`；
8. 记录Corpus SHA256、行数、无效行数和字段；
9. 只有两项均成功才写`prepare-complete.json`；
10. 不删除、不改名、不覆盖源分片或压缩包；不覆盖既有目标或partial。

最低空闲空间门禁为200GiB。准备前data盘约3.1TiB可用，预计新增约79GB。

## 3. 测试

三个无网络、无GPU测试已通过：

- 流式复制、字节数与SHA256一致；
- 单成员TAR正确抽取为JSONL并原子完成；
- 非法JSONL被拒绝且不产生完成目标。

## 4. 执行与验收

大文件准备使用独立tmux会话`project3-p25-prepare`，并设置：

```text
CUDA_VISIBLE_DEVICES=''
PYTHONNOUSERSITE=1
```

日志写入数据盘`prepare.log`。完成后再独立使用CPU FAISS读取`e5_Flat.index`，验证索引类型、
维度、向量数，并抽查首尾Corpus ID能被索引结果安全映射。准备成功本身不等同于FAISS服务
已经跑通。
