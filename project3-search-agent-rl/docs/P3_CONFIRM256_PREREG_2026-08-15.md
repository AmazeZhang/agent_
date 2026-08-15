# P3 Confirm-256 预注册：Base vs train64nqh8 配对比较（vLLM 原生 greedy）

**预注册时间**：2026-08-15（评测运行之前，本文件先于任何 confirm-256 评测结果提交）
**目的**：解决 EXPERIMENT_AUDIT.md 的 Required Action 3——dev32（32 题）上
`5/32 vs 3/32` 是初步信号但精确 McNemar `p=0.5` 不显著，且 dev32 已被多轮调参
反复查看。本实验用**从未被任何决策查看过**的 256 题确认集做单次、预注册的配对比较。

## 1. 假设

- **H1**：train64nqh8（`p3-grpo-fix-train64-nqh-n4-prompt-fmt-s0-20260814a`
  global_step_8 LoRA）在确认集上的 EM 高于 Base（Qwen2.5-1.5B-Instruct）。
- **H0**：两者 EM 无差异（配对 discordant 方向无偏）。

## 2. 固定条件（与 dev32 vLLM 评测完全一致，除数据外零改动）

| 项 | 值 |
|---|---|
| 评测脚本 | `scripts/run_p3_eval_vllm.py`（含运行时 SHA 自记录） |
| 引擎 | vLLM 0.8.5.post1，`VLLM_USE_V1=0`，bfloat16，FA，gpu_mem 0.6，enforce_eager，max_model_len 2304，`SamplingParams(temperature=0, top_p=1.0, top_k=-1)` greedy |
| tokenizer 输入 | `apply_chat_template(add_generation_prompt=True)`，truncation 2048，token ids 直传引擎 |
| env | `SearchMultiProcessEnv(seed=0, group_n=1, is_train=False)` + `SearchEnvironmentManager` + `search_projection`（严格 fork 语义） |
| 参数 | max_steps=2、history_length=2、topk=3、timeout=180、max_new_tokens=256 |
| Retriever | 真实 Wiki-18 IndexFlatIP，health 门禁 `vectors==21015324`，run 前重新验证 |
| 数据 | `datasets/searchr1-confirm256/heldout.parquet`（SHA `20e260d76221809b…`），manifest 核对 + 泄漏门禁（train/smoke/dev32 重叠=0） |
| 运行 | run_managed.sh 受管，物理 GPU1，`compute_processes=none` 退出验收 |
| 比较模型 | Base（无 adapter）vs train64nqh8（`…/global_step_8/actor/lora_adapter`，已核验 LORA r=32） |

## 3. 指标与统计（预先固定）

- **主指标**：EM（环境 reward ≥ 1.0，skyRL 严格 EM，format_score 不计入）。
- **主检验**：配对 McNemar 精确检验（双侧，`binom(1:0 discordant, 0:1 discordant)` 的
  精确二项 p 值）；同时报告 Wilson 95% 置信区间（各自 EM 率）与 discordant 明细
  （0→1、1→0、1→1、0→0）。
- **次要指标**（仅描述性，不做假设检验）：搜索 episode 数、invalid 动作数、
  answer_compliance、逐源 EM、生成文本差异统计（字节一致数、归一化编辑距离）。

## 4. 判定规则（预先固定，评测后不得更改）

1. **支持 H1**：`p < 0.05` 且 train64nqh8 EM 严格大于 Base EM。
2. **不支持 H1**：`p ≥ 0.05`（无论方向）。
3. **报告义务**：无论结论，必须报告两侧 Wilson CI、McNemar p、discordant 明细；
   结论措辞与 p 值严格挂钩（"显著提升"仅当 p<0.05）。
4. **方向反转**：若 train64nqh8 EM 更低且 p<0.05 → 明确记录为负向。
5. **设备/门禁失败**：任何 run 未通过退出验收（exit≠0、缺结果、显存未回基线）
   → 该模型重新受管运行，仅替换失败 run，不重抽数据、不改规则。

## 5. 禁止事项（评测前/中/后）

- 评测前不得查看 confirm-256 题目或任何模型输出；
- 不得以任何结果调参、换 adapter、换模型、改投影/奖励语义；
- 不得修改预注册规则；若发现脚本/数据 bug，修复后**重新预注册并重跑全部**；
- 构建器与评测脚本保持提交状态（SHA 记录在案）。

## 6. 规模选择说明

dev32 的 discordant 方向比例为 2:0（0→1=2）。256 题（配额按 dev32 同比例 ×8：
nq64/hotpotqa64/popqa32/2wiki32/triviaqa32/musique16/bamboogle16）在真实效应
相似时预计约 15 个 discordant pair；`b=15, p≈2e-4`（2:0 极端方向）到
`b=11/15, p≈0.12`（弱方向）覆盖"显著/不显著"两种可能，均可明确判定。256 是
用户给定范围（128–256）的上限，统计功效与 GPU 成本（每模型约 15–20 分钟）均可接受。

## 7. 声明边界

- 单 seed、greedy、fork 严格投影/惩罚语义下的单次确认实验；
- confirm-256 题从未参与任何调参或决策（构建时显式排除 dev32）；
- 本实验不声称 Search-R1 复现（官方宽松语义基线为另一条线，见第 4 步拆线）；
- 若支持 H1，也只支持"在该固定条件下 train64nqh8 优于 Base"这一有限声明。
