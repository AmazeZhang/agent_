# P2.5-E本机CPU HTTP Retriever完成报告

- 日期：2026-08-13
- 结论：真实Wiki-18 HTTP检索链路门禁通过
- 运行方式：CPU-only、localhost-only、顺序请求
- GPU：`CUDA_VISIBLE_DEVICES=''`，未使用任何GPU

## 1. 本阶段完成内容

1. 将已验证的CPU环境克隆到data盘专用环境，未修改来源环境；
2. 在全量21,015,324行Corpus上构建`uint64`行偏移表；
3. 加载真实64.56GB `IndexFlatIP`、本地E5 Encoder和Corpus随机访问层；
4. 启动仅监听`127.0.0.1:18080`的HTTP Retriever；
5. 分别完成8条和16条Top-3顺序请求及响应契约检查；
6. 向唯一目标tmux会话发送Ctrl-C并确认优雅退出；
7. 确认端口关闭、tmux关闭，GPU上只保留原有GNOME进程。

## 2. 隔离环境

```text
来源：/home/imc/anaconda3/envs/paretotool-retriever
副本：/media/imc/data/project3-search-agent-rl/envs/searchr1-retriever-cpu
大小：3.2G
Conda包数：88
```

关键版本：

```text
Python       3.10
faiss        1.8.0
numpy        1.26.4
torch        2.4.0+cpu
transformers 4.47.1
fastapi      0.139.2
uvicorn      0.51.0
pydantic     2.13.4
```

自检确认`torch.cuda.is_available() == false`。小型FAISS、Corpus随机访问和HTTP契约
`unittest`退出码为0；仅出现Starlette TestClient弃用警告，不影响实际Uvicorn服务。

完整环境锁见`configs/requirements-retriever-cpu-lock.txt`。锁文件由以下命令生成：

```text
SHA256 65d4f7fd1bf512331174ee0b8d42f9a780da579f29cbbd4914affedf4a393b0e
```

```bash
env PYTHONNOUSERSITE=1 \
  /media/imc/data/project3-search-agent-rl/envs/searchr1-retriever-cpu/bin/python \
  -m pip freeze --all --local
```

一次非正式探测因遗漏`PYTHONNOUSERSITE=1`而混入用户级包，显示了错误的Transformers和
Uvicorn版本；实际服务及正式锁均禁用用户site，模块路径与`importlib.metadata`一致。该探测
未安装、升级或修改任何包。

## 3. Corpus偏移表

```text
文件：prepared/wiki-18.offsets.npy
行数：21,015,324
数组元素数：21,015,325（含末尾sentinel）
大小：168,122,728 bytes
末尾偏移：14,393,573,105
构建耗时：12.15 s
SHA256：e87da022d0a2ea7955b8c458f9f6c4428a4e495b0b6cf746851ddc6224e3ea50
```

末尾偏移等于Corpus文件字节数。服务使用内存映射偏移表和`os.pread`按ID读取，未生成
约14GB Arrow Corpus副本。

## 4. 服务验收

健康检查报告：

```json
{
  "status": "ready",
  "index_class": "IndexFlatIP",
  "dimension": 768,
  "vectors": 21015324,
  "corpus_rows": 21015324
}
```

服务索引加载耗时40.15秒，24个CPU线程，固定Top-3。每条响应均检查HTTP状态、JSON嵌套、
返回数量、`document.id/contents`和浮点Score。

| 批次 | 成功/请求 | 错误 | P50 | P95 | Max | 全文答案字符串命中 |
|---|---:|---:|---:|---:|---:|---:|
| 8条 | 8/8 | 0 | 4.234 s | 4.371 s | 4.371 s | 5/8 |
| 16条 | 16/16 | 0 | 4.240 s | 4.366 s | 4.370 s | 10/16 |

结果文件：

```text
prepared/http-validation-n8.json
SHA256 8901c330bc6e1586a0387625d8925f0023a803798dec8ba3c0e777f687efe01e

prepared/http-validation-n16.json
SHA256 3baf1bf0fa3853c319b080681f3756eeaafdeab186ae84703bde0ac955e35053
```

“全文答案字符串命中”只是对Top-3文档全文做大小写不敏感字符串查找。它受别名、日期格式、
多跳问题和答案标注影响，不能当作正式Recall，更不能据此声称模型质量或训练收益。

## 5. 安全与异常记录

- 启动脚本内部硬编码`127.0.0.1`，不接受`0.0.0.0`；
- 首次命令误传不支持的`--host`参数，argparse在任何索引加载前退出；删除冗余参数后正常启动；
- 服务全程显式隐藏GPU，并使用CPU版Torch；
- 只停止`project3-p25-http-retriever`，未使用`tmux kill-server`或模糊进程清理；
- 日志包含Uvicorn完整`Shutting down`与`Finished server process`；
- 停止后tmux不存在，端口不可访问；
- `nvidia-smi`仅见`gnome-remote-desktop-daemon`占用354MiB，本实验无GPU进程残留。

## 6. 复现与观察命令

启动：

```bash
tmux new-session -d -s project3-p25-http-retriever \
  "cd /home/imc/yzy/agent/project3-search-agent-rl && \
   env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 \
   /media/imc/data/project3-search-agent-rl/envs/searchr1-retriever-cpu/bin/python \
   scripts/serve_p25_cpu_retriever.py \
   --resource-root /media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5 \
   --port 18080 --threads 24 \
   > /media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5/prepared/http-server.log 2>&1"
```

观察：

```bash
tmux attach -t project3-p25-http-retriever
tail -f /media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5/prepared/http-server.log
curl http://127.0.0.1:18080/health
```

安全停止：

```bash
tmux send-keys -t project3-p25-http-retriever C-c
```

## 7. 门禁结论与下一步

P2.5-E证明了真实Search-R1检索资源可以在本机以受控HTTP接口稳定运行，响应结构兼容
上游Search工具，且没有使用GPU或产生进程残留。该结论只覆盖“真实检索服务可运行和可调用”。

下一步进入P3配置审计：对固定veRL版本、Hydra参数、GPU物理编号、显存预算、rollout并发、
Checkpoint和退出清理进行逐项翻译。审计通过后，先用物理GPU1在tmux执行最小1步非零参数
更新，不直接启动全量训练。
