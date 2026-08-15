# P3 官方模型验证预注册：官方 Search-R1 3B GRPO vs Qwen2.5-3B Base（官方宽松语义线）

**预注册时间**：2026-08-15（本文件先于任何官方线评测运行提交）
**实验线**：official-loose（`docs/P3_EXPERIMENT_LINES_2026-08-15.md` §1 左列，
`run_p3_eval_vllm_official.py` 独立实现）
**目的**：分离"我们的 8 步 LoRA 训练太弱没效果"与"我们的环境观察不到 Search-R1 效应"
这两个混淆（严格线 confirm-256：train64nqh8 31/256 vs Base 37/256，p=0.109，H1 不支持）。
官方 checkpoint 是官方环境训练出的强模型；它在**我们评测链路上**的表现是对
"环境能否观察 Search-R1 效应"最便宜的关键判断。

## 1. 假设

- **H1**：官方 Search-R1 3B GRPO checkpoint（`PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`，
  F32 合并权重，~13.6GB）在官方宽松语义确认集上的 EM 高于 Qwen2.5-3B Base
  （`Qwen/Qwen2.5-3B`，非 Instruct——官方模型卡指定的配对 Base）。
- **H0**：两者 EM 无差异（配对 discordant 方向无偏）。

## 2. 固定条件（评测运行前全部固定，运行后不得更改）

| 项 | 值 |
|---|---|
| 评测脚本 | `scripts/run_p3_eval_vllm_official.py`（官方宽松语义，含运行时 SHA 自记录）；wrapper `run_p3_eval_vllm_official.sh` |
| 引擎 | vLLM 0.8.5.post1，`VLLM_USE_V1=0`，bfloat16，FA，gpu_mem 0.6，enforce_eager，max_model_len 2304，`temperature=0` greedy |
| 语义 | **official-loose**：raw action 直达 vendored skyrl `SearchEnv`（无 `SearchEnvironmentManager`、无投影）；`_parse_action = re.search(r"<search>(.*?)</search>", DOTALL)`；无查询→tool 异常→错误观察文本→模型重试，**无惩罚**；`<answer>` 与 `</answer>` 均出现即终局；终局 `compute_score(chat_history, gt, format_score=0.1)`（patch 0004 对齐论文） |
| tokenizer | **固定为 Qwen2.5-3B Base**（`--tokenizer`）。理由：官方 checkpoint 的 `tokenizer_config.json` 内嵌带 tools 的 chat_template，两模型若各用各的 template 输入渲染将不同；固定 Base 使两模型收到 **byte-identical input token ids**，唯一变量是权重 |
| env/参数 | `SearchMultiProcessEnv(seed=0, group_n=1, is_train=False)`；max_steps=2、history_length=2、topk=3、timeout=180、max_new_tokens=256、`--max-envs-per-batch 32`（CPU retriever 容量，纯并发控制） |
| Retriever | 真实 Wiki-18 IndexFlatIP（e5-base-v2，21,015,324 向量），health 门禁 `vectors==21015324`，run 前重新验证。**官方 retriever 同系**（intfloat/e5-base-v2 + e5_Flat.index），不按 embedding 差异处理 |
| 数据 | `datasets/searchr1-official-confirm256-v1/heldout.parquet`（SHA `ffebf468e756…`，256 行，domain `searchr1-p3-official-confirm-v1`）：排除上游 train、smoke train/test、**dev32**、**旧 confirm-256**（均已在构建器内核对，manifest 记录 overlap=0）；旧 confirm-256 已被严格线使用，**本实验不引用"未查看"字样**，只声明"与本实验的决策无关" |
| 运行 | run_managed.sh 受管，物理 GPU1，`compute_processes=none` 退出验收；proxy 净化（wrapper 内 unset + NO_PROXY） |
| 比较模型 | Base `Qwen2.5-3B`（无 adapter）vs 官方 GRPO checkpoint（完整权重，无 adapter）——官方线无 LoRA 概念 |
| 模型核验（运行前） | 两模型下载后：config（architectures=Qwen2ForCausalLM）、safetensors 分片清单、SHA256、dtype（官方 F32、Base bf16）、磁盘驻留大小；核验结果作为运行记录加注（§8），不改变本表任何固定条件 |

## 3. 指标与统计（预先固定）

- **主指标**：EM（env reward ≥ 1.0；官方线终局 reward 含 format_score=0.1，EM 判定仍为 ≥1.0，与严格线口径一致）。
- **主检验**：配对 McNemar 精确检验（双侧，discordant 方向上的精确二项 p）；Wilson 95% CI（各自 EM 率）；discordant 明细（0→1、1→0、1→1、0→0）。
- **次要指标**（仅描述性，不做假设检验）：检索状态分布（success/tool_exception/no_results/api_error）、error observation 步数（宽松线中"invalid"的对应物）、format_scored（reward=0.1）episode 数、answer_compliance、逐源 EM、生成文本差异（字节一致数、归一化编辑距离）、离线 EM 复核（format 0.0）与 env reward 一致性。

## 4. 判定规则（三档，预先固定，评测后不得更改）

1. **PASS**：p < 0.05 **且** 官方 EM 严格 > Base EM
   → "环境能观察到 Search-R1 效应"，批准进入 3B 复现训练阶段（第二阶段门禁另行预注册）。
2. **FAIL-TO-OBSERVE**：p < 0.05 **且** 官方 EM 严格 ≤ Base EM（显著负向或显著持平）
   → "官方模型在我们的环境下观察不到正向效应"，环境与官方不一致的强证据；
   停止训练计划，先诊断环境（首选：检索质量对比——官方 retriever 同系 e5，索引内容/分块差异；其次动作语义、prompt 构造）。
3. **INCONCLUSIVE**：p ≥ 0.05（无论方向）
   → "无法观察到显著效应"。**明确不作为环境不一致的证据**（样本功效、效应量、
   或环境差异均可能）；结合次要指标（检索成功率、error 率、搜索行为）与
   后续诊断形成方向，不直接批准/否决训练。
4. **报告义务**：无论结论，必须报告两侧 Wilson CI、McNemar p、discordant 明细；
   结论措辞严格挂钩 p 值与方向。
5. **设备/门禁失败**：任何 run 未通过退出验收（exit≠0、缺结果、显存未回基线、
   检索超时>0、retriever health 失败）→ 该模型重新受管运行，仅替换失败 run，
   不重抽数据、不改规则。

## 5. 禁止事项（评测前/中/后）

- 评测前不得查看 official-confirm256-v1 题目或任何模型输出；
- 不得以任何结果调参、换模型、改语义/引擎/数据；
- 不得修改预注册规则；若发现脚本/数据 bug，修复后**重新预注册并重跑全部**；
- 构建器与评测脚本保持提交状态（SHA 记录在案）；
- 官方线数字**不得**与严格线或论文数字直接对照（两线语义不同，`P3_EXPERIMENT_LINES_2026-08-15.md` §3.4）。

## 6. 规模选择说明

与严格线 confirm-256 同规模（256 题，配额 nq64/hotpotqa64/popqa32/2wiki32/triviaqa32/musique16/bamboogle16）。
若官方 checkpoint 效应显著（官方论文 3B EM 远高于 base），预计 discordant pair 足以在
p<0.05 判定；若效应很小，256 题与严格线同功效水平（严格线 8:2 discordant 对应 p≈0.109）。
256 题 GPU 成本约每模型 15–20 分钟（受管、分块 32 envs）。

## 7. 声明边界

- 单 seed、greedy、官方宽松语义下的单次验证实验；
- 本实验判定的是"我们的评测链路能否观察到官方训练出来的效果"，不是官方模型
  在官方环境上的复现成绩（后者需官方 retriever 索引/官方评测脚本，超出本实验）；
- 即使 PASS，也只批准进入训练阶段，不代表 3B 复现一定成功（训练阶段有独立的
  Step 50/100/300 门禁）。

## 8. 运行记录（追加；不改变第 4 节规则）

**2026-08-15 加注 A（模型下载与核验，先于任何评测）**：

- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`（gated=false，无需 token）：
  config `architectures=[Qwen2ForCausalLM]`、`torch_dtype=float32`（**F32 合并权重**）、
  hidden 2048 / 36 layers（=Qwen2.5-3B 规格）；3 个 safetensors 分片共
  13,588,463,912 字节（4,982,131,536 + 4,932,949,336 + 3,673,383,040），
  safetensors 头部可读（152/192/91 = 435 tensors），与 index `total_size`
  13,588,414,464 匹配（差额为分片头部元数据）；SHA256 见下表。`tokenizer_config.json`
  的 chat_template 为 2427 字符的 **tools 版本**（含 `{%- if tools %}` system 段）——
  实证确认 §2 固定 Base tokenizer 的决定（两模型输入 byte-identical，只有权重是变量）。
- `Qwen/Qwen2.5-3B`（gated=false）：config `torch_dtype=bfloat16`，36 层；2 个分片
  共 6,171,877,376 字节（3,968,658,944 + 2,203,268,048），safetensors 可读
  （264/170 = 434 tensors），与 index `total_size` 匹配。
- 两模型均通过代理下载（clash 7890；hf 客户端在代理下分块下载停滞，改 curl 并行
  分片 + `-C -` 断点续传，聚合约 10MB/s）。
- SHA256（safetensors 分片，逐文件）：
  - 官方 00001 `7ac54e1b9762c3c6d639da28a2cca177fe7db092ff5cf6e5a9a7849a36a9dabf`
  - 官方 00002 `98b373c4a6805af7723f2b31a5e72a919f4d7c021b6f4e67d91f579a08db8c67`
  - 官方 00003 `f1607045409131e298ad87b485a7fb74d02891178a0106a2df79cf8daf7b2c54`
  - Base 00001 `f9558df91d3b89b4826e4db37439edb52f1d62a4fd602685013e7ca6b9f60f8f`
  - Base 00002 `51410930d5cf19a998fdb17ef0c46e4d9ace72c97a975a3331395a8a500f5edb`

**2026-08-15 加注 B（评测入口与数据，先于任何评测）**：

- `scripts/run_p3_eval_vllm_official.py` + `run_p3_eval_vllm_official.sh`（独立官方线
  入口，无投影/无惩罚/format 0.1，raw action 直达 vendored skyrl `SearchEnv`）；
  CPU 测试 `tests/test_eval_vllm_official.py` 通过（40 行/8 分块：chunk 构造、
  全局索引、raw-action 直通、error-observation 13/13、schema 标记、`--tokenizer`
  默认逻辑）；严格线测试未受影响（2 个测试全绿）。
- `datasets/searchr1-official-confirm256-v1/heldout.parquet`：256 行，domain
  `searchr1-p3-official-confirm-v1`，SHA `ffebf468e756a673da267f5830cfc67f2e9c4dc44ec41c979a389c1efebfff60`，
  重建确定性一致（MATCH）；排除上游 train、smoke train/test、dev32（32 行）、
  旧 confirm-256（256 行），manifest `inputs.extra_exclusions` 记录；泄漏 0。

**2026-08-15 加注 C（smoke-16 门禁与正式评测结果，评测后追加，不改规则）**：
（待评测运行后填写。）
