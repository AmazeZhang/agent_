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
59. P3-C在物理GPU1从Global Step 1恢复并完成Global Step 2，`grad_norm=0.283`；
60. Scheduler推进到`last_epoch=2`，392/392个LoRA张量变化，Checkpoint Tracker为2；
61. 21条运行时Mask记录全部Prompt Loss为0，5条检索Observation Prompt同样通过；
62. 3次真实Wiki-18检索保存9个Document ID，2次空查询被类型化为`invalid_query`；
63. Ray TaskEventBuffer Segfault已消失，但SIGTERM关闭Worker仍被记为`SYSTEM_ERROR`，退出门禁WARN；
64. GPU1、Ray/Python和CPU Retriever全部定向清理；独立完整性审计总体WARN，不支持质量/完整复现结论。

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

## 2026-08-13：Ray干净退出修复预验证

Checkpoint恢复实验暴露的退出WARN已完成代码级修复：显式关闭环境与Torch process group，按
Actor ID去重调用`ray.actor.exit_actor()`，等待退出任务并通过GCS确认`DEAD`，最后由Driver
关闭其自行初始化的Ray。普通rollout JSONL也改为独占partial写入、fsync、原子rename和拒绝
覆盖。7项单元测试、真实Ray 2.43 CPU Actor探针、0001至0003干净补丁重放均通过。

当前准确状态是“代码和CPU/Ray预验证通过，等待物理GPU1恢复复验”，尚未把退出WARN改判为
通过。详细记录见`docs/P3_CLEAN_SHUTDOWN_FIX_2026-08-13.md`。

首次物理GPU1复验Attempt E完成Step 2、Checkpoint和JSONL证据，GPU融合Worker已变为
`INTENDED_USER_EXIT`；但Ray关闭时CPU TaskRunner仍被SIGTERM回收，整体门禁保持WARN。现已增加
TaskRunner主动退出与GCS DEAD确认，形成“GPU Worker → TaskRunner → Driver/Ray”的完整顺序，
等待Attempt F复验。

Attempt F确认GPU Worker和TaskRunner均已`INTENDED_USER_EXIT`，但进一步扫描发现内部
`WorkerGroupRegisterCenter`仍被引用计数回收并在core日志留下SIGTERM/SYSTEM_ERROR，故继续WARN。
现已将它加入主动退出，顺序扩展为“RegisterCenter → GPU Worker → TaskRunner → Driver/Ray”；
8项测试和真实Ray父子Actor无SYSTEM_ERROR探针通过，等待Attempt G最终复验。

Attempt G最终复验已完成：Global Step 1→2更新、Checkpoint和两类Rollout完整；RegisterCenter、
GPU Worker、TaskRunner均主动退出并进入DEAD，Actor/训练Worker无SYSTEM_ERROR、unexpected
failure、SIGTERM或段错误，资源释放。独立审计保持WARN，因为Ray基础设施正常关闭仍使用
EXPECTED_TERMINATION SIGTERM，且8题单seed smoke不构成完整复现或质量评测。下一阶段可讨论
5步工程晋级，但held-out validation和baseline前禁止质量提升声明。

五步受控晋级计划已制定：从Attempt G `global_step_2`恢复到Step 5，只新增三次更新，继续使用
8题smoke、物理GPU1和CPU Wiki-18，不扩大到全量数据或多卡。预计5–7分钟、数据盘新增约23GiB。
计划和通过门禁见`docs/P3_FIVE_STEP_PROMOTION_PLAN_2026-08-13.md`。

五步受控晋级Attempt H现已完成：从Attempt G Step 2准确恢复，新增Step 3、4、5三次真实更新，
三个Checkpoint、六个Rollout证据、Prompt loss mask、typed retrieval metadata和Actor生命周期均通过
核验；GPU1、Ray与Retriever资源已安全释放。独立审计为`WARN`：Step 4/5任务reward和success为0，
且未执行held-out validation，因此只能认定短程工程闭环通过，不能声称质量提升、收敛、泛化或
完整复现。完成报告见`docs/P3_FIVE_STEP_PROMOTION_COMPLETION_2026-08-13.md`；下一步优先建立
Step 2与Step 5的同条件held-out评测和多seed/baseline，而不是盲目扩大到20步。

## 2026-08-13：Held-out 评测管线（CPU 阶段完成，未启动 GPU）

已拍板：评测集 = smoke-16 管线门禁 + heldout-32 正式对比；实现 = HF 侧轻量评测
（复用`run_p2_model_search.py`骨架，构造性零训练：无优化器/无 backward/无 Ray）。

1. `scripts/build_p3_heldout_eval.py`（CPU-only）从上游 test 51,713 条确定性抽样32条
   （SHA256升序+分源配额），排除上游 train 169,615 条、smoke train 8 条、smoke test
   16 条中出现的规范化问题；拒绝覆盖已有输出；重建两次 `heldout.parquet` 与
   `records.jsonl` SHA256 完全一致；泄漏计数全为0。产物在
   `/media/imc/data/project3-search-agent-rl/datasets/searchr1-heldout32/`
   （heldout.parquet sha256 `1f8caca3…`、records.jsonl `63ddd14a…`、manifest.json）；
2. `scripts/run_p3_eval_heldout.py`：固定评测参数与训练一致（seed 0、max_steps 2、
   history 2、topk 3、timeout 180、prompt 2048 / new 256、贪心解码）；启动门禁全部
   abort（受管运行、单卡非GPU0、Retriever `/health` 且 vectors==21015324、数据SHA256
   与manifest核对、评测问题∩smoke-train=∅）；逐步证据 JSONL + `results.json` 原子写；
   指标含总体/分源 EM、success、无效查询率、格式错误率、离线EM复核；
3. `scripts/run_p3_eval_heldout.sh` 受管 wrapper：commit pin、路径、loopback URL、
   受管运行、单卡GPU1、补丁应用、EVAL_DATA 合法值与 adapter 目录门禁，与训练同构；
4. `tests/test_eval_heldout.py` 10项 CPU 测试全绿；相关既有套件 33 passed
   （test_p25_cpu_retriever_service.py 需 retriever CPU 环境，本环境跳过）；
5. 详细计划与结果规范见`docs/P3_HELDOUT_EVAL_PLAN_2026-08-13.md`。

本轮**未占用 GPU、未启动真实 Retriever、未训练**。GPU 阶段（preflight → tmux 启动
真实 Retriever → 6 个受管评测 Run：3模型×smoke-16 + 3模型×heldout-32 → 退出验收 →
汇总+Wilson置信区间）需另行批准；有明确提升才用 verl/vLLM 原生评测复核并进入
多seed/更大评测，无提升则排查训练配置，不直接跑20步。

### 遇到的问题与解决（2026-08-13 CPU 阶段）

1. `atomic_write_jsonl` 对 numpy 标量（`np.int64`）抛
   `TypeError: Object of type int64 is not JSON serializable`：写入前统一经
   `jsonable()` 归一化（info/observation 本就走 jsonable，写入器现在同样防御性
   归一化，避免证据文件在写盘阶段静默失败）；修复后 10/10 测试通过。
2. 全量 `pytest` 收集 `tests/test_p25_cpu_retriever_service.py` 报
   `ModuleNotFoundError: No module named 'faiss'`：既有环境差异（faiss 只在
   `searchr1-retriever-cpu` 环境），与本轮改动无关；本环境相关套件 33 passed，
   该文件按惯例在 retriever CPU 环境运行。
3. 决策记录：smoke manifest 将 smoke 集限定为协议测试、`forbidden_use` 含质量声明
   → 采用两层结构：smoke-16 只做管线门禁（方向信号），heldout-32（新建、去重、
   记录 SHA256）作为 Step 0/2/5 第一轮正式对比证据。

## 2026-08-14：下一阶段已固定（尚未启动GPU）

只读交接检查确认heldout-32、Base/Step 2/Step 5评测代码和Adapter路径均已准备完成；宿主机
物理GPU1当前为18MiB，数据盘约3.0TiB可用，无评测/Ray/Retriever进程。下一阶段严格串行执行
三组smoke-16门禁和三组heldout-32对比，均使用HF纯评测、物理GPU1、真实CPU Wiki-18和新Run ID。
任一smoke失败即停止；六个Run结束后做逐题配对分析、精确停止Retriever和资源验收。详细顺序、
Run ID、停机条件与结论边界见`docs/P3_NEXT_ACTIONS_2026-08-14.md`。本轮仅写计划，未启动GPU、
Retriever或训练。
