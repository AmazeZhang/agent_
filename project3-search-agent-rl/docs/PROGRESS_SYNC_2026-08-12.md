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

## 2026-08-14：Held-out 评测 GPU 阶段执行完成

按 `docs/P3_NEXT_ACTIONS_2026-08-14.md` 完成六个纯评测 Run（全部 exit_code=0、逐 Run 验收通过）：

| Run ID | 模型 | 数据 | EM | success |
|---|---|---|---|---|
| p3-eval-smoke-base-s0-20260814a | Base | smoke-16 | 0/16 | 0/16 |
| p3-eval-smoke-step2-s0-20260814b | Step 2 | smoke-16 | 1/16 | 1/16 |
| p3-eval-smoke-step5-s0-20260814c | Step 5 | smoke-16 | 2/16 | 2/16 |
| p3-eval-heldout32-base-s0-20260814d | Base | heldout-32 | 0/32 | 0/32 |
| p3-eval-heldout32-step2-s0-20260814e | Step 2 | heldout-32 | 1/32 | 1/32 |
| p3-eval-heldout32-step5-s0-20260814f | Step 5 | heldout-32 | 0/32 | 0/32 |

smoke-16 三个 Run 只作为管线门禁（模型加载、LoRA 挂载、门禁、原子产物均正常），不用于质量声明。
heldout-32 三组同数据 SHA（`1f8caca3…`）、同 seed/参数/后端（HF greedy, temperature 0）、
leakage=0、Retriever health `ready`+`vectors=21015324`。

**heldout-32 结果解读（详见 `docs/eval-heldout32-20260814/comparison.md` 与 comparison.json）：**

- EM：base 0/32（95% CI 0.0–10.7%）、step2 1/32（0.6–15.7%）、step5 0/32（0.0–10.7%）；
  success 与 EM 相同；answer 合规率 96.9%/96.9%/100%。
- 配对 McNemar：base↔step2、base↔step5、step2↔step5 的不一致对分别为 1/0/1，两尾精确 p 均为 1.0，
  三组无统计显著差异（32 题小样本，区间大幅重叠）。
- 失败分类（EM=0 的 episodes）：未搜索直接作答占绝对主导（base 28、step2 24、step5 26）；
  无效动作 step2/5 各 5 例（base 1 例，混合/重复标签为主）；检索到但答错 2/1/1；检索失败 0；
  无 answer 格式 1/1/0。
- 单条 EM（step2 的 nq "celebrity big brother"）是参数记忆命中（CBS），未检索即答对，
  不能视为搜索行为改善。训练后模型（step2/5）搜索调用次数没有增加，无效动作反而更多。

**结论（按既定判定路线）：Step 5 在 heldout-32 上无一致正向信号，Step 2 仅 1/32 且不显著。**
因此**不追加 20 步训练**。下一步进入"排查训练配置"分支：围绕 reward（检索成功/信息增益是否进 reward）、
prompt 动作格式（<think>+<search> 引导、few-shot）、rollout n/group size、学习率与格式奖励，
设计最小修正实验（如动作格式对齐 + 检索奖励）后重训对照。若后续配置修正产生明确提升，
再用 verl/vLLM 原生评测复核关键结论（HF greedy 与训练 vLLM rollout 存在 backend 差异，已在
results.json 的 decoding_note 中声明）。

**遇到的问题与解决：**

1. Retriever 首启失败：`serve_p25_cpu_retriever.py` 在未设 `PYTHONNOUSERSITE=1` 时从
   `~/.local/lib/python3.10/site-packages` 加载 transformers，与 env 内 importlib 混装导致
   导入 traceback。解决：按旧会话已验证模式 `env CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1
   OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 …` 重启，/health 即返回 ready + 21015324。
   后续受管评测 wrapper 自带 `PYTHONNOUSERSITE=1` 导出，不受此问题影响。
2. 一次 adapter 路径"不存在"误报：检查命令用了仓库相对路径，而 runs 在数据盘根下，属检查
   命令基目录错误，非产物缺失；改用数据盘绝对路径后确认 Step 2/Step 5 adapter 均在位。
3. 分析脚本首轮测试 3 个断言失败均为测试期望手算错误（Wilson 宽度≈0.35 而非 <0.25；
   McNemar 平衡不一致对为 3 而非 4；base↔step5 不一致对各 1 → p=1.0 而非 0.5），
   实现本身正确，修正测试后 12/12 通过。
4. 资源检查时 pgrep 自匹配问题：进程名模式出现在 pgrep 自身命令行导致误报；改用
   `ps -eo pid,comm,args` + comm 过滤后确认无 python 残留进程。

**资源验收（Gate 4）**：Retriever 仅以精确 Ctrl-C 停止（tmux 会话 `p3-eval-retriever-20260814`），
端口 18080 无监听、无 python retriever/评测进程；八卡显存全部回基线（GPU1 18 MiB，0% util；
GPU0 仅桌面进程 354 MiB，未触碰）。旧 8-13 tmux 会话与历史产物未做任何删除。

## 2026-08-14：训练配置根因排查完成，最小修正实验方案已定

只读排查（未占 GPU、未改代码）确认 Step 4/5 零 reward 与"搜索未学会"的根因链，详见
`docs/P3_CONFIG_DIAGNOSIS_2026-08-14.md`：

1. **Prompt 无搜索协议指令**：env `_sync_reset` 只把原始问题文本给模型，无系统指令/few-shot；
   heldout-32 三模型未搜索直接作答占比 24–28/32。
2. **Reward 稀疏且无格式奖励**：只有最终严格 EM，`format_score=0.0` 从未启用；训练期
   epoch 3–5 rollout 审计 24/24 条 reward 全为 0。
3. **GRPO group size n=1**：hydra 解析 `actor_rollout_ref.rollout.n=1`，verl 对单样本组用
   mean=0/std=1，advantage 退化为原始 reward；batch 全 0 时策略只剩 KL 漂移。
4. 训练期温度 1.0 采样其实会输出 `<search>`（epoch5: 9/24），但贪心 eval 坍缩回直接作答，
   且零奖励从未强化该格式；无效动作（混合/重复标签）训练后期增多与之印证。

**最小修正实验方案（`docs/P3_MINIMAL_FIX_EXPERIMENT_2026-08-14.md`）**：三项变量同时启用但
独立可审计 —— (1) env prompt 加系统指令 + 1 个 few-shot 搜索示例（改
`search/envs.py::_sync_reset`，训练评测共用同条件）；(2) `format_score=0.1`（Search-R1 原版）；
(3) `rollout.n=4` 恢复 GRPO 组基线。固定变量与 Attempt H 完全一致（8 行 smoke、LoRA r32、
lr 3e-6、5 步、GPU1、真实 retriever）。预注册判据：训练 reward 出现非零、eval 搜索调用率
≥50%、无效动作率 < 18.4%、EM 方向为正；不满足则按序调 lr/n/epochs/response 上限，不追加
20 步、不直接扩大数据。CPU 部分（补丁 0004 + wrapper + 测试）待批准后实施。

### 2026-08-14（续）：最小修正实验 CPU 部分完成

按已批准方案实施完毕，未占 GPU：

1. **补丁 0004**（`patches/0004-search-prompt-and-format-reward.patch`）两处增量：
   - `envs.py::_sync_reset` 前置 `SEARCH_PROMPT_PREFIX`（系统指令 + "Imagine" few-shot 示例，
     训练与评测共用同条件）；
   - skyrl `SearchEnv._get_reward` 传 `format_score=0.1`（Search-R1 原版格式奖励），
     intermediate steps 仍零奖励。
2. **生成方式**：vendor 工作树内 apply 0001–0003 → 提交为 base → 复制编辑后文件 →
   `git diff` 得到仅含 0004 增量的补丁；已验证 0001→0002→0003→0004 可在干净 HEAD
   依序应用，且主工作树 `git apply --reverse --check` 通过。
3. **wrapper**：`scripts/run_p3_grpo_fix_exp.sh`（复制 Attempt H 配置，仅改
   `rollout.n=4`、实验名、补丁门禁含 0004）；`apply_project_patches.sh` 与
   `run_p3_eval_heldout.sh` 门禁循环加入 0004。
4. **测试**：`tests/test_patch_0004.py` 5/5 通过；全量 CPU 套件
   （排除已知缺 faiss 的 test_p25_cpu_retriever_service.py）50 passed。

**遇到的问题与解决：**

1. 手写 patch 在 apply 时报 corrupt（空白/上下文不符）——弃用手写，改 worktree
   base-commit diff 法生成，结果干净且可逆检查通过。
2. `git diff` 一次捕获了 8 个文件（0001–0003 未提交的改动混入）——只 diff 两个目标文件，
   并确保 base commit 在复制编辑文件之前建立（曾有一次先 `git add -A` 导致 diff 为空）。
3. 测试 fixture 契约错误：`SearchEnv.ground_truth` 必须是 `{"target": [...]}` 字典
   （env kwargs 传递的契约），测试起初传 `["x"]` list 触发 utils.py:94
   `TypeError: list indices must be integers, not str`；修正三处 fixture 后 5/5 通过。

**下一步（待 GPU 批准）**：`run_p3_grpo_fix_exp.sh` 训练 5 步（run ID
`p3-grpo-fix-n4-prompt-fmt-s0-20260814a`），然后 heldout-32 三模型对比
（base / old step5 / new step5'），对照预注册标准验收。

### 2026-08-14（续）：最小修正实验 GPU 执行完成

**训练（run `p3-grpo-fix-n4-prompt-fmt-s0-20260814b`，5 步，GPU1）**：exit 0。
对比 Attempt H（reward 全 0），本轮 5 步 `episode/reward/mean` 全部非零
（0.172/0.137/0.166/0.153/0.184，max 1.0 每步都有答对），`tool_call_count/mean`
0.44–0.56（约半数轨迹输出 `<search>`，Attempt H 同期为 0），entropy 0.91→0.78
（策略在收紧）。score 分布：0.1×27 格式对答错、1.0×3 答对、0.9×2 答对+1 无效
动作、-0.1×6 无效动作惩罚（fork `apply_invalid_action_penalty`，本就生效）。

**heldout-32 三模型 eval（run f/g/h，正确数据）**：base 2/32、step5old 2/32、
step5new 2/32 —— **三模型答对的完全同一批问题（2wiki 1 + nq 1），McNemar 不一致
对 = 0，p = 1.0，无方向性差异**。无效动作率 step5new 26.8%（< old 34.9% 但 >
预注册 18.4%）；no_search 22/32（搜索调用率仅 31%，< 预注册 50%）；answer 合规
100%（few-shot 指令确实教会了 `<answer>` 标签格式）。

**预注册判据核对**：

| 判据 | 结果 |
|---|---|
| 训练 reward 非零（step5 均值 > 0） | ✓ 0.184 |
| 搜索调用率 ≥ 50%（eval） | ✗ 31% |
| 无效动作率 < 18.4%（eval） | ✗ 26.8% |
| EM ≥ 2/32 且方向为正 | △ 2/32 达到下限，但与 base 零差异 |

**结论**：修正实验部分成功（训练信号恢复、搜索行为出现、格式合规），但 8 行
smoke × 5 步预算下 heldout 无提升。按预注册失败路线进入第二轮：调 lr（1e-5）→
env.rollout.n（8）→ epochs（10）→ max_response（512），不追加 20 步。

**遇到的问题与解决：**

1. 训练首启 exit 1：fork 硬断言 `actor_rollout_ref.rollout.n==1`
   （`verl/trainer/main_ppo.py:173`，GRPO 组在 env 侧 `env_manager.py:609`）。
   修正：改 `env.rollout.n=4`（Attempt H 为 2），`actor_rollout_ref.rollout.n`
   保持 1。诊断文档 #3 根因表述同步修正（机制是"组内全零 reward → mean=0/std=0
   → advantage 恒零"，非单样本组退化）。
2. eval 首轮 c/d/e 三 run 实际加载了 smoke-16（`data_files` 指纹为
   `searchr1-smoke/test.parquet`、n=16）：外层 `PROJECT3_EVAL_DATA=... bash
   start_tmux_run.sh` 的环境变量不进入 tmux 会话（tmux server 环境继承），
   wrapper 落到默认 `smoke`。修正：把 `env PROJECT3_EVAL_DATA=heldout32
   PROJECT3_EVAL_ADAPTER=...` 放在 `--` 之后的命令里（与 8-13 六个 eval run
   的 command.txt 一致）。重跑 f/g/h（32 条、数据 SHA `1f8caca3…` 与 manifest
   一致）。错误 run c/d/e 的产物保留未删。
3. gpu_guard 拒绝并行：d/e 与 c 同时启动被 `physical GPU 1 already has compute
   processes` 拒绝（exit 3）；改为串行启动后全部成功。8-13 六个 eval run 实为
   串行启动，本次并行尝试被正确拦截。

**资源验收**：训练与三个 eval 后，GPU 八卡全部回基线（GPU1 18 MiB 0%）、无
python 残留、retriever 以精确 Ctrl-C 停止（会话 `p3-fix-retriever-20260814`），
端口 18080 释放。所有 run 目录（含失败启动 a 与错误数据 c/d/e）保留未删。

### 2026-08-14（续）：第二轮最小修正实验（lr 1e-5）执行完成

**训练（run `p3-grpo-fix-lr1e5-n4-prompt-fmt-s0-20260814a`，lr 1e-5，5 步）**：
exit 0。reward/mean 全非零（0.172/0.087/0.134/0.112/0.162），step1 与第一轮
完全一致（seed/数据/LoRA 初始化确定性复现），tool_calls 0.375–0.562，
grad_norm 0.478–0.718（略高于第一轮，lr 更大）。step5 无效动作率 48%（较第一轮
44% 略差）。wrapper 的 lr 已参数化（`PROJECT3_FIX_EXP_LR`，默认 3e-6 保持第一轮
可复现），experiment_name 内嵌 lr。

**heldout-32 eval（run `p3-eval-heldout32-fix2-step5new-s0-20260814a`）**：
step5new2 EM 2/32 —— 与 base/step5old **同一批 2 道题**（2wiki 1 + nq 1），
McNemar 不一致对 = 0，p = 1.0。无效动作率 24.4%（略优于第一轮 26.8%，仍 >
18.4% 预注册线）；no_search 22/32（搜索率 31%，< 50% 预注册线）。

**预注册判据第二轮核对**：训练 reward 非零 ✓；搜索率 ≥50% ✗（31%）；无效动作
<18.4% ✗（24.4%）；EM 方向 ✗（零差异）。

**结论**：lr 3e-6 → 1e-5 无实质改变（两轮 eval 结果高度雷同：EM 2/32 同一批题、
搜索率 31%、无效动作 24–27%）。按预注册阶梯进入第三轮：`env.rollout.n=8`
（组大小翻倍，advantage 归一化基线更稳），随后才是 epochs 10 / max_response 512。
仍未决定是否扩大数据——8 行 smoke × 5 步预算可能已达该预算下的天花板，第三轮
若仍无信号，将重新审视"先扩大数据规模"路线（预注册失败路线允许：仍无信号才
考虑扩大数据）。

**资源验收**：GPU 八卡回基线（GPU1 18 MiB 0%）、无 python 残留、retriever 以
精确 Ctrl-C 停止（会话 `p3-fix2-retriever-20260814`）、端口 18080 释放。所有
run 目录保留未删。

### 2026-08-14（续）：第三轮最小修正实验（env.rollout.n=8）执行完成，smoke-8 触顶判定

**训练（run `p3-grpo-fix-n8-prompt-fmt-s0-20260814a`，n=8，5 步）**：exit 0。
reward/mean 全非零（0.103–0.172）；step1 再次与前两轮逐项一致（reward 0.172、
valid_action 0.696、tool_calls 0.438、grad_norm 0.488——seed/数据/LoRA 确定性
完全复现）；step2–5 tool_calls 0.44–0.59、valid_action 0.52–0.61，与前两轮无
实质差异。wrapper 的组大小已参数化（`PROJECT3_FIX_EXP_N`，默认 4）。

**heldout-32 eval（run `p3-eval-heldout32-fix3-step5new-s0-20260814a`）**：
step5new3 EM 2/32 —— 三轮一致，**仍是同一批 2 道题**（2wiki 1 + nq 1），
McNemar 不一致对 = 0，p = 1.0。无效动作率 29.3%（三轮无改善趋势：26.8% →
24.4% → 29.3%）；no_search 22/32（搜索率 31%，三轮恒定）。

**三轮汇总**：

| 轮 | 变量 | EM | 搜索率 | 无效动作率 |
|---|---|---|---|---|
| 1 | n=4, lr 3e-6 | 2/32 | 31% | 26.8% |
| 2 | lr 1e-5 | 2/32 | 31% | 24.4% |
| 3 | n=8 | 2/32 | 31% | 29.3% |

**结束条件触发（用户预注册判定）**：搜索率 <50%、无效动作 >18.4%、heldout 零
配对改善 → **停止 smoke-8 调参，判定小数据预算触顶，转向扩大训练集**。按用户
指令不自动继续 epochs=10 / max_response=512。扩大训练集的具体方案（数据规模、
保留变量组合、预算）待用户批准后执行。

**资源验收**：GPU 八卡回基线（GPU1 18 MiB 0%）、无 python 残留、retriever 精确
Ctrl-C 停止（会话 `p3-fix3-retriever-20260814`）、端口 18080 释放。三轮全部 run
目录保留未删。

### 2026-08-14（续 2）：第三轮 n=8 修正重跑（run b）执行完成 —— smoke-8 触顶判定成立，train-64 已就绪

**背景**：第三轮（a）因 tmux env passthrough 陷阱实际未生效（resolved hydra 仍为
n=4）。修正方案（wrapper 自检首行 + `-- env PROJECT3_FIX_EXP_N=8` 传参，commit
e35c8cc）后重跑。

**n=8 生效核验（用户要求的三个判据全部通过）**：

| 判据 | n=4（run b 参照） | n=8（run b） | 结论 |
|---|---|---|---|
| resolved hydra `env.rollout.n` | 4 | **8**（`actor_rollout_ref.rollout.n=1` 不变） | ✓ |
| rollouts 行数/步 | 46 / 50 | **91 / 101**（1.98x / 2.02x） | ✓ |
| global_seqlen（生成 token 量） | 25933 | **49232**（1.9x）；perf/total_num_tokens 同步 | ✓ |

训练（run `p3-grpo-fix-n8-prompt-fmt-s0-20260814b`）exit 0，5 步全非零 reward，
step5：reward/mean 0.175、tool_call_count 0.344、valid_action_ratio 0.698（三轮
最高）、success_rate 0.094。每步 301s（n=4 为 175s，与 rollout 量翻倍吻合）。

**heldout-32 eval（run `p3-eval-heldout32-fix3b-step5new-s0-20260814a`）**：
step5new3b EM 2/32，与 base/step5old 持平。配对 McNemar：base↔new 不一致对=2
（1 对 new 胜：nq "celebrity big brother"；1 对 new 负：nq "Stag Night…"），
p=1.0 无方向性。搜索率 28%（no_search 23/32，较前轮 31% 略降）；无效动作率
22.5%（较前轮 29.3% 改善，仍 > 18.4% 门槛）。数据文件 SHA 三组一致
`1f8caca3…`（heldout32 manifest 核对 True），耗时 54.5s（f 85.5s / g 91.4s，
同量级无异常）。对比报告：
`/media/imc/data/project3-search-agent-rl/eval-heldout32-fix3b-20260814/`。

**判定（用户预注册结束条件）**：搜索率 <50%、无效动作 >18.4%、heldout 零配对
改善 → **停止 smoke-8 调参，判定小数据预算触顶，转向扩大训练集**。不自动继续
epochs=10 / max_response=512。

**train-64 构建器（`scripts/build_p3_train64.py`，已就绪）**：确定性分源抽取 64
行（配额 nq 16/hotpotqa 16/popqa 8/2wiki 8/triviaqa 8/musique 4/bamboogle 4），
排除 smoke-train 8 + heldout-32 32 的规范化问题（泄漏=0，跨池重复=0）。
**数据审计发现**：上游 `train.parquet`（169,615 行）只有 nq+hotpotqa 两源；
其余 5 源仅存在于上游 `test.parquet`。故双池混合：nq/hotpotqa 取 train 池、
其余 5 源取 test 池（数据盘全盘确认无其他来源）。输出
`datasets/searchr1-train64/train.parquet`（SHA `029e1a7f…`，schema 与 smoke
train 一致），重建确定性匹配。训练配置将沿用修复集：prompt、format_score=0.1、
n=4、lr=3e-6、train_batch_size=8、8 个 optimizer steps（1 epoch 覆盖 64 行）。

**资源验收**：训练与评测 exit 0，GPU1 回基线（cleanup.log compute_processes=none），
run 目录全部保留。train-64 训练（GPU）待批准后执行。

### 2026-08-14（续 3）：train64-nqh（主线扩大训练集）执行完成 —— 无 heldout 改善，转查动作格式/奖励设计

**决策（用户）**：不采用双池混合 train-64 作为主线（32 条来自上游 test，破坏
Search-R1 跨数据集泛化口径）。主线改为 **train64-nq-hotpot**：仅上游 train，
NQ 32 + HotpotQA 32，排除 smoke/heldout 问题；混合版保留为备选证据不删除。
其余 5 源保持纯净，作真正的跨源泛化测试。训练配置批准：batch=8、rollout.n=4、
lr=3e-6、8 optimizer steps、1 epoch、保留 prompt 修复与 format_score=0.1；
heldout-32 不变。

**数据集**（`scripts/build_p3_train64.py` 扩展 selection-domain/配额参数，默认
仍可重建混合版 `029e1a7f`）：主线 `datasets/searchr1-train64-nqh/train.parquet`，
domain `searchr1-p3-train64-nqh-v1`，SHA `df3464c8…`，64 行（nq 32 + hotpotqa
32），泄漏 0（smoke/heldout 规范化问题零重叠），跨池重复 0，与混合版问题
零重叠（新 domain 完全不同的抽样），重建确定性匹配。

**训练（run `p3-grpo-fix-train64-nqh-n4-prompt-fmt-s0-20260814a`，GPU1）**：
exit 0；自检首行 `[FIX_EXP] resolved: fix_exp_data=train64-nqh fix_exp_lr=3e-6
fix_exp_n=4 total_training_steps=8 total_epochs=1`；train_files/val_files 解析
正确（val 用 smoke test.parquet，train64-nqh 无 test.parquet）。8 个 checkpoint
（步 1–8），每步 ~166-169s（与 smoke n=4 的 ~175s 一致——batch 不变则每步计算量
不变，印证用户预判）。step1–8 reward/mean 全非零（0.128–0.206）、tool_call
0.34–0.59、success 0.06–0.13。

**heldout-32 eval（run `p3-eval-heldout32-train64nqh-step8-s0-20260814a`）**：
train64nqh8 EM **2/32**，与 base/step5old 完全持平；**McNemar 不一致对 = 0**
（与 base 答对的是同一批 2 道题：2wiki 1 + nq 1），p=1.0；搜索率 31%
（no_search 22/32，与前几轮一致）；无效动作率 24.4%（仍 > 18.4% 门槛）；
answer 合规 100%；数据 SHA `1f8caca3…` 核对 True。对比报告：
`eval-heldout32-train64nqh-20260814/`。

**判定（用户预注册路线）**：扩大训练集（64 行 × 1 epoch）无 heldout 改善 →
**转查动作格式/奖励设计，不继续堆数据和 epochs**（也不做多 seed / heldout-64）。

**资源验收**：训练与评测 exit 0，GPU1 回基线（cleanup.log compute_processes=none），
run 目录全保留；retriever 会话 `p3-fix3b-retriever-20260814` 继续运行（下一步
排查如需复用）。

## 2026-08-14：动作格式 + 奖励路径 CPU 只读诊断完成（6 项全查）

用户批准的 6 项只读诊断全部完成（无 GPU、无训练、无 vLLM 评测）。产物：
`/media/imc/data/project3-search-agent-rl/diag-action-reward-20260814/{diagnosis.md,diagnosis.json}`，
分析脚本 `scripts/analyze_p3_action_reward_diag.py`（本轮升级为 env 精确重建，
377/377 行完全复现，mismatch=0）。

**1/6 + 6/6. LoRA 是否真正改变输出（base vs train64nqh8，同一批 32 题）**：
字节一致 13/32（40.6%）、平均编辑距离 0.183、第一步一致 14/32、有效性翻转 2、
**EM 翻转 0**。→ LoRA 确实改变了输出与动作分布，但未跨过任何 EM 判定边界；
训练更新真实生效，只是没有转化为答案正确率。

**2/6. mixed/duplicate 为何判无效**：`projection.py` 规则 = 动作同时含
`<search>` 与 `<answer>` 标签（mixed）或任一标签 ≥2 次（duplicate）→
`valids=0`，训练时该行扣 -0.1。模型常见输出模式正是
`<think>+<search>+<answer> 一步完成`（搜答混合），因此大量被判 mixed。
**注意 `<answer />` 自闭合标签**：不含 `<answer>` 子串 → 标签计数不判 mixed、
`_is_done` 也不认（无 `</answer>`）→ 该步不终止并继续搜索/直到 max_turns。
官方 Search-R1 无 mixed/duplicate 规则（取第一个标签执行，invalid 重试无惩罚）。

**3/6. 官方 vs fork 解析器**：
- `extract_solution`：官方 `len(matches) <= 1 → None`（prompt+response 含 few-shot
  示例 = 2 个 answer 块）；fork `len(matches) < 1 → None`（chat_history 仅投影后
  actions，无示例）→ fork ≥1 是必要适配，语义自洽；
- 动作解析：官方宽松（`<(search|answer)>(.*?)</\1>` 取第一个、invalid 环境提示
  重试不终止无惩罚）；fork 严格（trim 到第一个闭合标签、mixed/duplicate 判无效、
  -0.1/行惩罚）→ **fork 特有训练信号设计，非 bug**。

**4/6. 奖励路径精确重建（关键）**：逐 (traj, step) 行按 env 语义重放：
SearchEnvironmentManager.step **先投影再传给 env**（chat_history 只含投影后
动作 + 搜索观察）→ done 步 compute_score 取历史最后一个 answer 块 →
`gather_rollout_data` 把 episode_rewards（累计值 = 最终 compute_score）写入
该 traj 每一行 → EpisodeRewardManager 放到每行最后 token →
`apply_invalid_action_penalty` 按行扣 0.1×该行 invalid。公式：
`recorded_score == episode_rewards - 0.1*(0 if row_valid else 1)`。
**377/377 行重建一致，mismatch=0**（早期 18 条/73 条 mismatch 全部归因于重建
脚本未建模投影先于 env、episode 累计按行写入、按行惩罚三层语义，非训练实现问题）。
精确分解：256 个 episode（8 步 × 32 行）中最终 EM 25（9.8%）、仅格式分 186
（72.7%）、无 answer 45（17.6%）；377 行中 invalid 147（39.0%）→ 惩罚单元 147。

**5/6. heldout 失败分类（train64nqh8）**：no_search 23（69%）、searched_then_wrong
8、searched_then_correct 1、format_error 0、invalid_action_only 0。→ 贪心模式下
**69% 的题完全不搜索**是 EM 卡 2/32 的主因（训练用采样、评测用 HF 贪心）。

**结论（按用户指示决策）**：解析/奖励语义**完全正确，无偏差，无需修复**。训练
低效的原因不在实现，而在：(a) 规模（64 行 × 8 步 × 1.5B LoRA 远小于 Search-R1
原版）；(b) 评测用 HF 贪心 vs 训练采样（69% 不搜索）；(c) 中间步 credit 平坦
（每行都拿最终 episode reward，搜索步与答案步信号无区分）。→ 下一步按用户
指示：**用 verl/vLLM 原生贪心评测排除 HF backend 差异**，不修实现、不改训练
temperature。

**资源**：诊断全程 CPU 只读；retriever 会话 `p3-fix3b-retriever-20260814`
继续运行（vLLM 评测可能复用）。`scripts/analyze_p3_action_reward_diag.py`
本轮升级，随文档提交。

## 2026-08-14（续）：vLLM 原生评测准备（CPU 阶段）完成

**目标（用户批准的两级门禁，第一部分）**：准备 vLLM 原生贪心评测脚本，禁止
optimizer/backward；固定 temperature=0、相同 heldout SHA、env、projection、
Retriever；验证 Base 与 LoRA 加载路径。

**交付（本 commit）**：
- `scripts/run_p3_eval_vllm.py` — HF 评测孪生脚本，唯一差异是解码后端：
  vLLM 原生引擎 + `SamplingParams(temperature=0)` greedy；LoRA 用 vLLM 原生
  `LoRARequest` 加载 PEFT 目录（inference-only，无优化器状态）。全部门禁
  （受管运行 / 单卡 / GPU0 禁 / Retriever health 21015324 / SHA / 泄漏）与
  HF 版逐条一致。
- `scripts/run_p3_eval_vllm.sh` — 受管 wrapper（镜像 heldout wrapper），
  export `VLLM_USE_V1=0`（与训练 rollout 同一 V0 引擎路径）。
- `scripts/compare_hf_vllm_eval.py` — HF↔vLLM 逐题对比（CPU 纯分析）：
  EM、搜索率、无效动作、原始动作文本字节一致 + 归一化编辑距离；不比速度。

**引擎 parity（与 run_p3_grpo_fix_exp.sh 训练 rollout 对齐）**：
`VLLM_USE_V1=0`（脚本 gate，非 "0" 即 abort）、dtype=bfloat16、
tensor_parallel_size=1、gpu_memory_utilization=0.6、enforce_eager=True、
max_model_len=2304。tokenizer 输入侧与 HF 版字节级一致（同一
apply_chat_template + truncation 2048），token ids 直传引擎。

**问题与解决**：
1. vLLM 0.8.5.post1 的 `LoRARequest` 不在顶层命名空间（`vllm.LoRARequest`
   ImportError）→ 从 `vllm.lora.request` 导入（实测可导入、`LLM.generate`
   接受 `lora_request` 参数）。
2. adapter_config.json 的 target_modules 是展开后的 7 个 Qwen2 投影
   （q/k/v/o/gate/up/down_proj），非 "all-linear" 字符串 → vLLM 原生
   LoRA loader 支持模块名列表；r=32、alpha=32（alpha/r=1，无缩放差异）、
   base_model_name_or_path 与 --model 一致，全部通过 `validate_adapter_for_vllm`。

**CPU 验证结果（全部通过）**：
- 静态审查：无 torch.optim / ray / main_ppo import、无 backward() 调用点
- 引擎 parity gate：`VLLM_USE_V1=0` ✓、vLLM 0.8.5.post1 记录进结果
- 真实 train64nqh8 adapter（global_step_8）校验通过：LORA、r=32、7 targets
- 真实数据门禁：heldout32=32 条泄漏 0 / SHA `1f8caca3…` 核对一致；
  smoke=16 条泄漏 0 / SHA 一致
- 纯函数冒烟：aggregate_metrics / action_quality（mixed + `<answer />`
  自闭合陷阱）/ offline_rescore 全部正确
- 对比脚本合成数据：4 题 EM 2/3、flips 各 1、字节一致 2 全部命中

**注意**：LoRA 权重真正加载到 vLLM 引擎需要 GPU → 这一步是 GPU smoke 门禁
run 本身（加载失败 run 直接 abort）。GPU 状态一律以 run_managed.sh /
preflight.sh 门禁输出为准，不主动查询 GPU0。

## 2026-08-14（续）：vLLM 原生评测完成 — backend 差异定位，正式 backend = vLLM

**GPU 两阶段门禁全部通过**（preflight 门禁输出确认，不主动查询 GPU0）：
- smoke-16 × 2（Base + train64nqh8）：退出 status 0、16 条 episodes、SHA 核对、
  泄漏 0、cleanup `physical_gpu=1 compute_processes=none`（显存回基线）。
- smoke 即验证了 LoRA 加载路径的 GPU 侧：`enable_lora=True`、max_lora_rank=32、
  adapter LORA r=32 7 targets 加载成功（无 tokenizer 警告属预期，用 base tokenizer）。

**heldout-32 vLLM 原生贪心结果（正式数字，与训练 rollout 同引擎路径）**：

| 模型 | HF EM | vLLM EM | vLLM 搜索 | HF↔vLLM 字节一致 | 归一化编辑距离 |
|---|---|---|---|---|---|
| Base | 2/32 | **3/32** | 6（HF 11） | 0/32 | 0.650 |
| step5old | 2/32 | **5/32** | 6（HF 11） | 0/32 | 0.653 |
| train64nqh8 | 2/32 | **5/32** | 5（HF 9） | 0/32 | 0.648 |

（对比存档：`analysis/hf-vs-vllm-heldout32-20260814/{base,step5old,train64nqh8}.json`）

**判定（按用户规则：vLLM 明显不同 → 定位差异）**：
1. 定位：新增最小诊断 run（`scripts/diag_hf_vllm_gen.py`，同 8 题、同进程、无 env）
   → **纯 LLM 生成即分歧**：8/8 题在前 14–76 字符处措辞级分歧后雪崩式分叉。
   差异源在生成层：HF（eager attention + tf32-off + bf16）vs vLLM V0
   （FlashAttention，训练 rollout 同配置）的算子/数值差异，env/projection/
   retriever/数据门禁全部排除（两遍完全一致）。
2. LoRA 在 vLLM 后端真实生效：base vs train64nqh8 输出字节一致仅 19/32
   （13 题被 LoRA 改变），EM flips 2 全部 0→1 正向。
3. **正式 eval backend = vLLM**：与训练 rollout 同引擎（V0、bf16、FA、
   gpu_mem 0.6、max_model_len 2304、enforce_eager）。HF eval 是与训练路径
   数值不一致的第三方后端，其 2/32 是 backend 特有数字。

**结论转变（backend 选择改变训练有效性判断）**：
- HF 下：训练模型与 base 持平（2/32 vs 2/32）→ 曾判定"无提升"。
- vLLM（训练同路径）下：训练模型 5/32 vs base 3/32 → **LoRA 有 +2 EM 的
  初步正向效果**（32 题小样本，Wilson 95% 区间宽，属初步证据非断言）。
- 搜索率 vLLM 下更低（5–6 vs HF 9–11）但 EM 更高：vLLM 下模型更倾向直接
  `<answer>` 且答对率更高，是生成差异的连锁效应（非策略行为改变）。
- 与诊断结论呼应：reward 语义正确、LoRA 生效；vLLM 后端下 LoRA 跨过了 EM
  边界（5 vs 3）。根因拆解（训练规模小、中间步 credit 平坦）仍然成立，但
  "策略不搜索"应修正为"贪心解码下搜索收益有限、直接作答反而更稳"。

**问题与解决（本轮 4 个运行级 bug）**：
1. `return_tensors="pt"` + padding=False 批量长度不一 → 改 ragged list
   （`return_tensors=None`），vLLM 引擎内部自行 batching。
2. vLLM 0.8.5 `LLM` 无 `shutdown()` → `del llm` + gc（释放引擎 GPU 分配）。
3. `LoRARequest(path=...)` 参数不存在（`path` 是只读 property）→
   `lora_path=`（0.8.5 字段）。
4. tmux server 环境不透传 `PROJECT3_EVAL_*`（start_tmux_run 只带
   DATA_ROOT/MIN_FREE_GIB）→ pane 内 `bash -c 'export ...; exec ...'`
   显式设置（首跑 base run 时 adapter=None 暴露此问题）。

**下一步（用户指示的拆线）**：
- 线 1 官方宽松语义（`<(search|answer)>(.*?)</\1>` 取第一、invalid 环境提示
  重试无惩罚）→ Search-R1 复现基线（对照官方论文数字）。
- 线 2 严格投影 + 每行 invalid 惩罚 → 我们的 fork 改进/对照实验（可继续
  训练规模与 credit 整形路线）。
