# P2.5-D CPU FAISS与真实检索验收计划

- 日期：2026-08-13
- 状态：验收脚本待执行
- GPU：完全隐藏
- CPU线程上限：24

## 目标

P2.5-C只证明产物在字节、JSON和体积关系上自洽。本步骤必须用FAISS实际读取索引，并用
固定E5模型执行真实搜索，排除“文件拼接正确但FAISS不可读”或“向量ID无法映射Corpus”的情况。

## 检查项

1. 使用隔离的`paretotool-retriever`环境中`faiss-cpu==1.8.0`加载索引；
2. 限制FAISS和Torch为24线程，避免占满96逻辑CPU；
3. 要求索引维度768、向量数21,015,324、已训练；
4. 使用固定本地`intfloat/e5-base-v2`，按上游`query: `前缀、Mean Pooling和L2 Normalize编码；
5. 从固定Smoke测试集中取8条Query，执行Top-3 Flat搜索；
6. 全量扫描Corpus，要求每行`id == str(零基行号)`；
7. 要求全部检索ID都能映射到真实文档；
8. 记录索引加载、编码、搜索、Corpus扫描耗时及每条Top-3预览；
9. 结果只证明真实检索集成，不作为模型准确率或论文分数。

## 资源与安全

- 索引加载预计占用约65GB CPU内存，服务器可用内存约995GiB；
- 不启动HTTP服务、不启动训练、不使用Ray；
- 设置`CUDA_VISIBLE_DEVICES=''`，不触碰GPU0；
- 输出`prepared/cpu-validation.json`若已存在则拒绝覆盖；
- 使用独立tmux，结束后检查进程和GPU残留。
