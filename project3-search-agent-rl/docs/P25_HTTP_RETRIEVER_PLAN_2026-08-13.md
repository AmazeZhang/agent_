# P2.5-E本机CPU HTTP Retriever计划

- 日期：2026-08-13
- 状态：已完成，结果见`docs/P25_HTTP_RETRIEVER_COMPLETION_2026-08-13.md`
- 服务地址：仅`127.0.0.1:18080`
- GPU：完全隐藏

## 1. 为什么不直接运行上游服务

上游`retrieval_server.py`存在以下本机风险与成本：

1. `uvicorn.run(..., host="0.0.0.0")`会暴露到所有网络接口；
2. Corpus通过`datasets.load_dataset("json")`建立Arrow缓存，会增加约14GB以上副本；
3. GPU模式调用`index_cpu_to_all_gpus`，若可见卡未隔离可能使用GPU0；
4. 单请求服务没有显式并发锁，多请求可能同时占用FAISS和Torch线程；
5. 没有Top-k上限、健康检查或Corpus ID对齐门禁。

P2.5只需要验证真实检索HTTP链路，因此采用功能等价、边界更严格的CPU服务。

## 2. 项目实现

### Corpus偏移表

`scripts/build_p25_corpus_offsets.py`全量扫描JSONL并生成`uint64` NPY：

- 长度为`21,015,324 + 1`；
- 第一个偏移为0，最后一个为Corpus字节数；
- 约168MB；
- 先写partial，完成后原子改名；
- 输出SHA256和完成Manifest；
- 服务通过`os.pread`线程安全随机读取，不创建Arrow副本。

### CPU Retriever

`searchr1_repro/cpu_dense_retriever.py`：

- 加载`IndexFlatIP`和固定本地E5；
- 使用上游一致的`query: `前缀、Mean Pooling和L2 Normalize；
- FAISS索引、Corpus和Encoder必须满足向量数/行数一致；
- 模型编码、FAISS搜索和文档读取由单锁保护；
- 每个返回ID再次检查`document.id == str(index)`。

### HTTP边界

`scripts/serve_p25_cpu_retriever.py`：

- 硬编码监听IPv4 Loopback `127.0.0.1`；
- 默认Top-3，最大Top-10；
- 空Query、超长Query、非法Top-k返回422；
- `/health`报告索引类型、维度、向量数和Corpus行数；
- `/retrieve`结构严格兼容上游：

```json
{"result": [[{"document": {"id": "...", "contents": "..."}, "score": 0.0}]]}
```

### 固定验收客户端

`scripts/validate_p25_http_retriever.py`只接受Loopback URL，依次执行8条和16条：

- 验证HTTP状态、JSON嵌套、Top-k数量、文档字段和Score；
- 记录每请求延迟、P50/P95/Max和错误；
- 保存每条Top-3 ID、Score和前500字符；
- 全文答案字符串命中只作为有限诊断，不作为正式Recall或模型质量。

## 3. 环境隔离

执行环境将从已通过P2.5-D的CPU环境克隆到：

```text
/media/imc/data/project3-search-agent-rl/envs/searchr1-retriever-cpu
```

克隆后生成完整`pip freeze --all`锁文件。所有命令设置`PYTHONNOUSERSITE=1`，不修改原
`paretotool-retriever`环境。直接依赖和完整锁分别记录在
`configs/requirements-retriever-cpu.txt`与`configs/requirements-retriever-cpu-lock.txt`。

## 4. 执行顺序

1. 克隆并验收隔离CPU环境；
2. 构建并验收Corpus偏移表；
3. 检查18080端口未占用；
4. tmux启动服务并等待`/health=ready`；
5. 运行8条HTTP门禁；
6. 通过后运行16条；
7. 向该tmux发送Ctrl-C，等待优雅退出；
8. 检查端口、Python、内存和GPU残留；
9. 写完成报告、提交并推送。

## 5. 当前测试

小型3文档、2维`IndexFlatIP`测试已经验证：

- NPY偏移构建与随机读取；
- Top-2排序；
- 上游HTTP嵌套结构；
- `/health`；
- 空白Query和超最大Top-k拒绝。

测试和CLI静态检查均在CPU环境、GPU隐藏、用户site-packages禁用条件下通过。
