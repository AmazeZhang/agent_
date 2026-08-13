# P2.5-D CPU FAISS与真实E5检索完成报告

- 完成时间：2026-08-13 16:42:53（Asia/Shanghai）
- 状态：CPU FAISS加载、E5编码、Top-3搜索和全量ID对齐全部通过
- 执行方式：独立tmux `project3-p25-cpu-validate`
- GPU：`CUDA_VISIBLE_DEVICES=''`
- 训练：未开始

## 1. 执行环境

```text
faiss-cpu 1.8.0
torch 2.4.0+cpu
transformers 4.47.1
FAISS/Torch线程上限 24
```

运行时索引驻留后系统约67GiB内存已用，仍有约934GiB可用；结束后内存恢复至约7.1GiB
已用。没有Swap、GPU、Ray或残留验证进程。

## 2. FAISS真实加载

| 字段 | 结果 |
|---|---|
| 类型 | `IndexFlatIP` |
| 维度 | 768 |
| 向量数 | 21,015,324 |
| 已训练 | `true` |
| Metric type | 0（Inner Product） |
| 加载耗时 | 39.24秒 |

这证明拼接文件不只是体积吻合，而是FAISS 1.8.0能够完整反序列化的Flat IP索引。

## 3. 真实Query检索

从固定Smoke测试集取前8条题目，使用本地固定revision的`intfloat/e5-base-v2`：

```text
query: <question>
Mean Pooling
L2 Normalize
FAISS Top-3
```

- 8条Query编码耗时：0.78秒；
- 8条Query批量全库Flat Top-3耗时：6.27秒；
- 24个返回ID全部映射到Corpus文档；
- Corpus全量对齐扫描耗时：114.13秒；
- 总验收耗时：160.44秒。

结果保存在：

```text
/media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5/prepared/cpu-validation.json
```

结果SHA256：

```text
2aaef82dd38314231672df781b64e04daf6e295cd115ea48b6555d5f5f3bc87f
```

## 4. Corpus与索引ID对齐

- 实际扫描21,015,324行；
- 首ID为`0`，末ID为`21015323`；
- 每一行均满足`id == str(零基行号)`；
- 对齐错误示例为空；
- 24个需要的检索文档全部找到。

因此上游`load_docs(corpus, doc_idxs)`的按行索引假设在该资源上成立。

## 5. 有限检索诊断

仅检查保存的每篇文档前500字符是否出现答案别名，8条中5条命中：

- How You Remind Me：Rank 1；
- Treaty of Versailles相关人物：Rank 2；
- Saint-Domingue：Rank 1/3；
- In Their Skin genre：Rank 1；
- Cover producer：Rank 3。

另外3条在前500字符未出现答案。这不是正式Recall@3，因为：

- 只取8条非随机Smoke；
- 只检查每篇前500字符，可能漏掉后文；
- 使用简单字符串别名匹配；
- 多跳题需要跨文档推理，直接答案不一定应出现在首跳文档中。

该数字不能写成模型准确率、Retriever正式质量或论文对齐结果。

## 6. 安全退出

- tmux会话自然结束；
- FAISS索引内存已释放；
- 没有CPU验证进程残留；
- GPU0仍只有GNOME远程桌面进程约354MiB；
- 项目没有GPU计算进程；
- data盘约3.0TiB可用。

## 7. 下一门禁

P2.5-E需要建立固定且隔离的CPU Retriever服务环境，解决现有上游服务依赖`datasets`而
当前CPU FAISS环境缺该包的问题。之后：

1. 启动只监听`127.0.0.1`的`/retrieve`服务；
2. 使用8条再16条固定Query检查HTTP结构、Top-k、延迟和错误率；
3. 将P2模型驱动脚本从Fixture切换到真实服务；
4. 确认服务停止后无进程和端口残留；
5. 真实Retriever链路通过后才进入P3 veRL单步训练。
