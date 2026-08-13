# Search-R1复现进度同步

- 同步日期：2026-08-12
- 上一同步提交：`4543cf3`（P0/P1）
- verl-agent固定提交：`20bd331bdbc9026a5668e11362178e10ab7400c8`
- 当前阶段：P0、P1、P2完成；P2.5资源审计完成、真实资源待下载
- 训练状态：未开始

## 已完成内容

### P0：环境、存储和非破坏性实验门禁

1. 确认`data`盘为`/dev/nvme0n1`，挂载在`/media/imc/data`，约3.2 TiB可用；
2. 建立项目专用的cache、datasets、models、indexes、runs、checkpoints目录；
3. 记录8×RTX 4090 D、驱动595.45.04、CUDA Toolkit 12.4.131及NUMA/PCIe拓扑；
4. GPU0硬禁用，避免影响Xorg/GNOME Remote Desktop；GPU5默认禁用；
5. 固定Python 3.10、Torch 2.6.0+cu124、vLLM 0.8.5.post1、Ray 2.43.0、
   TensorDict 0.8.3和Transformers 4.51.1；
6. 在data盘建立隔离环境并生成完整`pip freeze`锁；
7. 建立GPU空闲检查、项目锁、独立Session/Process Group、定向TERM/KILL、退出后显存核查；
8. 建立tmux训练包装。GPU1基础MatMul和FlashAttention SM89测试通过，退出后无残留进程。

对应文档：

- `docs/ENVIRONMENT_BASELINE_2026-08-12.md`
- `docs/EXPERIMENT_SAFETY.md`
- `configs/requirements-searchr1-repro.lock.txt`

### P1：数据、Retriever协议与Reward闭环

1. 从`PeterJinGo/nq_hotpotqa_train` revision
   `b7d80abfee334a7a91cb377544f09180d58b34f6`下载原始train/test Parquet；
2. 使用上游脚本处理为veRL需要的`prompt/reward_model/extra_info/env_kwargs`结构；
3. 得到训练169,615条、测试51,713条；发现原始数据10个规范化问题跨Split重叠；
4. 使用稳定SHA256排序生成8条训练、16条验证Smoke，重建前后哈希一致；
5. 实现只用于协议测试的确定性Fixture Retriever，其文档明确标记为Ground Truth衍生；
6. 跑通`<search>`→HTTP `/retrieve`→`<information>`→`<answer>`→严格EM；
7. 增加Retriever状态观测补丁，使API错误可与模型错误分开统计；
8. 12项CPU测试通过，耗时2.54秒；生成一条Reward=1的模型无关Trace；
9. 未下载Wikipedia Corpus或E5 Flat Index，未启动训练。

对应文档：`docs/P1_SEARCH_PIPELINE_AUDIT_2026-08-12.md`。

### P2：冻结模型与Search环境集成

1. 固定并下载`Qwen/Qwen2.5-1.5B-Instruct` revision
   `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，许可证Apache-2.0；
2. 权重落在data盘，3.087GB safetensors的SHA256与上游LFS对象一致；
3. 直接复用上游环境管理器、动作投影、SearchEnv和Chat Template实现模型驱动多轮链路；
4. 使用物理GPU1按1、4、16条逐级执行，所有Run退出码为0且无显存进程残留；
5. 正式16条Run有9个投影search、8次实际检索，8次均成功，16条全部终止；
6. 发现主动搜索不足、多动作格式无效以及最大步数search不实际执行三类基线问题；
7. 通过强制首轮搜索的独立诊断验证完整回环，但不把它混入自然策略指标；
8. Fixture由Ground Truth衍生，Reward结果只作为集成证据，不作为模型质量或论文分数；
9. P2达到Fixture环境下R1功能复现，训练仍未开始。

对应文档：`docs/P2_MODEL_SEARCH_INTEGRATION_2026-08-12.md`。

### P2.5-A：真实Retriever资源审计

1. 锁定上游`wiki-18-e5-index`、`wiki-18-corpus`和`e5-base-v2`的revision；
2. 记录两个索引分片、压缩Corpus和模型权重的大小及LFS SHA256；
3. 确认索引分片合计64,559,075,373 bytes，上游解压/拼接后整体约132GB；
4. 确认data盘可用3,408,163,352,576 bytes，系统可用内存约995GiB；
5. 两个数据仓库均未声明许可证，禁止把资源当作可重新分发资产；
6. 发现上游GPU FAISS默认使用所有可见GPU，严禁未隔离照搬；
7. 设计固定revision、250GiB磁盘门禁、保留源文件和逐文件验哈希的下载脚本；
8. 决定先使用`CUDA_VISIBLE_DEVICES=''`的CPU FAISS验证，不占用GPU0或任何训练卡。
9. 发现当前Hugging Face Xet客户端下载保持0 bytes，未产生完成文件；
10. 增加显式curl Range续传、`.partial`和成功后原子改名，试传约303MiB稳定；
11. 大文件下载将放入独立tmux会话，不依赖交互终端，完成后仍执行SHA256门禁。
12. 首次tmux下载因CDN TLS unexpected EOF在6,561,538,522 bytes退出，partial完整保留；
13. 下载器增加最多100轮外层Range恢复、连续5轮无增长熔断和逐轮字节审计；
14. 采用恢复续传，不删除或重新下载已有约6.56GB，恢复日志以追加方式保留失败证据。
15. 2026-08-13 13:24完成全部约70GB源资源下载；
16. 两个索引分片、压缩Corpus和E5权重的精确大小与SHA256全部匹配固定Manifest；
17. 下载完成标记已生成，tmux自然结束且无Python/curl/GPU进程残留；
18. P2.5-B完成，详细验收见`docs/P25_DOWNLOAD_COMPLETION_2026-08-13.md`。
19. 发现`wiki-18.jsonl.gz`真实为gzip压缩TAR，而非直接gzip压缩JSONL；
20. TAR内唯一JSONL成员大小14,393,573,105 bytes，抽样字段为`id/contents`；
21. 新增非覆盖、保留源文件、逐行JSON校验和原子完成的P2.5-C准备器；
22. 三项小型准备器测试通过，执行计划见`docs/P25_PREPARATION_PLAN_2026-08-13.md`。
23. P2.5-C完成：拼接索引64,559,075,373 bytes，SHA256为`69c98463...d166`；
24. 全量Corpus为14,393,573,105 bytes、21,015,324行、0无效JSON行；
25. 源分片与压缩包全部保留，无partial或进程残留，准备总耗时457.55秒；
26. 详细结果见`docs/P25_PREPARATION_COMPLETION_2026-08-13.md`，下一步为CPU FAISS实际加载。
27. P2.5-D实际加载为`IndexFlatIP(d=768, ntotal=21,015,324)`，耗时39.24秒；
28. 本地E5编码8条耗时0.78秒，CPU全库Top-3搜索耗时6.27秒；
29. 全量21,015,324个Corpus ID均与行号一致，24个返回文档全部可映射；
30. 验收tmux自然结束、内存释放、无GPU/进程残留，详细结果见
    `docs/P25_CPU_VALIDATION_COMPLETION_2026-08-13.md`。
31. P2.5-E实现行偏移随机访问Corpus，避免上游Arrow副本；
32. CPU服务固定监听`127.0.0.1`、限制Top-k、串行锁保护模型与FAISS；
33. 小型FAISS/HTTP契约测试通过，8→16执行计划见`docs/P25_HTTP_RETRIEVER_PLAN_2026-08-13.md`。
34. 专用CPU环境已克隆到data盘，关键依赖自检通过且`torch.cuda.is_available()==false`；
35. 全量Corpus偏移表构建完成，21,015,324行，大小168,122,728 bytes，耗时12.15秒；
36. localhost服务加载真实`IndexFlatIP`耗时40.15秒，健康检查向量数与Corpus行数一致；
37. 8条与16条真实HTTP Top-3验证分别8/8和16/16成功，均为0错误；
38. 16条延迟P50为4.240秒、P95为4.366秒、最大4.370秒；
39. 服务收到定向Ctrl-C后完整优雅退出，端口和tmux关闭，无GPU或Python进程残留；
40. P2.5-E完成，详细结果见`docs/P25_HTTP_RETRIEVER_COMPLETION_2026-08-13.md`。
41. P3-A核对固定veRL提交、CUDA 12.4、Driver 595.45.04和8×4090D状态；
42. 将8卡7B上游Search脚本翻译为物理GPU1、1.5B LoRA32、Group 2的一步GRPO配置；
43. 固定8条训练问题、16条轨迹、最多32次环境动作和180秒CPU Retriever超时；
44. Hydra完整解析成功，32项关键配置断言通过，0 mismatch；
45. 非受管启动、GPU0映射和未应用Patch分别由exit 13/14/15拒绝；
46. P3-A未启动Ray或GPU训练，详细审计见`docs/P3_GRPO_ONE_STEP_CONFIG_AUDIT_2026-08-13.md`。
47. P3-B Attempt A发现Ray Socket路径超过107字节，失败Run保留且修复为短`mktemp`路径；
48. Attempt B完成Ray/FSDP/LoRA初始化，但0.35显存预算不能创建vLLM KV Cache；
49. Attempt C在物理GPU1完成1次真实Actor Update，`grad_norm=0.300`并保存Global Step 1；
50. LoRA-B共20,643,840个元素全部非零，模型、Optimizer、Adapter和状态文件均已保存；
51. Run共8个问题、16条轨迹、21条Action，Reward均值0.125，不能据此声称质量提升；
52. 保存后Ray Worker在关闭阶段Segfault，顶层exit 0但不能称为干净退出；
53. 独立实验审计为WARN：单步更新成立，恢复、Token Mask和结构化Retriever证据待补。
54. P3-C加入逐记录Token Loss Mask、Retriever状态和Document ID的原子审计JSONL；
55. 两份项目Patch在干净veRL提交上顺序/反向应用，并与当前vendor代码比较一致；
56. 定位退出异常为Ray 2.43.0 TaskEventBuffer关闭路径，CPU-only禁用缓冲探针正常退出；
57. 恢复配置确认从Global Step 1加载Actor/Optimizer/DataLoader，以第二Epoch执行Step 2；
58. Mask单元测试2项、Search联合回归14项通过；Hydra确认`resume_path`、Step 2、Epoch 2，尚未启动GPU训练。

对应文档：`docs/P25_RESOURCE_AUDIT_2026-08-12.md`。

## 数据和产物位置

大文件不进入Git：

```text
/media/imc/data/project3-search-agent-rl/datasets/searchr1-upstream
/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke
/media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124
/media/imc/data/project3-search-agent-rl/envs/searchr1-retriever-cpu
/media/imc/data/project3-search-agent-rl/indexes/searchr1-wiki18-e5
/media/imc/data/project3-search-agent-rl/models/Qwen2.5-1.5B-Instruct
/media/imc/data/project3-search-agent-rl/runs/p2-qwen15b-fixture-*
```

关键哈希：

```text
processed train.parquet  aa98bf95dec9466899395e5d44e56e1b765cef7bc6b9ea226f5e6129bd0d360a
processed test.parquet   b00cd074f2e5de5eb464d17a0289217159d332a13879dd1ac2bae5312cc167de
Smoke manifest           c2f92e66702e1e5597a9e47759172bef6db5b5d17e988a08e3d448a89ef6c3b3
deterministic trace       5c2ac4dc40462720b44578ee64be507321c43440501419279cbdef1f7da28a62
direct requirements       dd03ff2a705b4b0446dce48358e83fc8846a26a2979855b2456dd7f76d36c530
third-party package lock  758ef9afec5a0796ecec4a9c072bcd72f2c973719ab756206959fe35bd177a65
Qwen model weights         dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
P2 official result JSON   2a905571b1388ebf44a65710da09741e2ef570a6b4c4243e6bba36c838a7900c
```

上游数据集Metadata没有许可证字段。当前状态为“许可证未声明、待人工核验”，不能推断为
Apache或据此重新分发数据。Git仅保存生成脚本、Manifest规则和哈希，不保存数据内容。

## 从干净仓库同步P1

```bash
cd /home/imc/yzy/agent/project3-search-agent-rl
git submodule update --init --recursive
bash scripts/apply_project_patches.sh

export PROJECT3_DATA_ROOT=/media/imc/data
source /media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124/bin/activate

python vendor/verl-agent/examples/data_preprocess/preprocess_search_r1_dataset.py \
  --local_dir /media/imc/data/project3-search-agent-rl/datasets/searchr1-upstream

CUDA_VISIBLE_DEVICES='' python scripts/build_p1_smoke.py \
  --source-dir /media/imc/data/project3-search-agent-rl/datasets/searchr1-upstream \
  --output-dir /media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke

CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/vendor/verl-agent:$PWD" \
  python -m pytest -q tests/test_search_p1.py
```

预期：`12 passed`。localhost HTTP测试需要允许绑定`127.0.0.1`随机端口。

## 下一步：P2.5与P3

1. P2.5真实资源下载、准备、CPU加载和localhost HTTP门禁均已完成；
2. 禁止使用Ground-Truth Fixture训练；
3. P3-A固定veRL版本下的Hydra单步配置审计已完成；
4. P3-B物理GPU1真实单步参数更新与Checkpoint保存已完成；
5. 补充Token级Loss Mask和结构化Retriever状态持久化；
6. 处理Ray/vLLM退出段Segfault并单独执行Checkpoint恢复Step；
7. 上述门禁通过后才讨论5/20步晋级。
