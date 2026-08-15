# P3 官方模型验证：官方 Search-R1 3B GRPO vs Qwen2.5-3B Base（官方宽松语义线）

预注册：`docs/P3_OFFICIAL_CHECKPOINT_PREREG_2026-08-15.md`（先于任何评测提交）
数据：`searchr1-official-confirm256-v1` heldout.parquet（SHA `ffebf468e756a673…`，
排除 dev32、旧 confirm256、训练集；构建确定重建一致）
语义：official-loose（raw action 直达 skyrl SearchEnv，无投影无惩罚，format_score=0.1）
后端：vLLM 0.8.5 V0 引擎 greedy；tokenizer 固定 Qwen2.5-3B Base（两模型输入 byte-identical）
运行：p3-eval-official-confirm256-base3b-s0-20260815a / p3-eval-official-confirm256-official3b-s0-20260815a（受管，GPU1，cleanup `physical_gpu=1 compute_processes=none`）

## 主指标（EM = env reward ≥ 1.0）

| 模型 | EM | 率 | Wilson 95% CI |
|---|---|---|---|
| Qwen2.5-3B Base | 20/256 | 0.0781 | [0.0511, 0.1176] |
| 官方 Search-R1 3B GRPO | 32/256 | 0.1250 | [0.0900, 0.1711] |

## 配对明细（discordant）

- 1→1（双方都对）：12
- 0→0（双方都错）：216
- 1→0（Base 对、官方错）：8
- 0→1（Base 错、官方对）：20

**精确双侧 McNemar p = 0.035698**（8:20，discordant n=28）

## 判定（预注册第 4 节，三档）

- **PASS**：p=0.0357 < 0.05 and official EM (32) > Base EM (20): environment can observe the Search-R1 effect
- PASS → 批准进入 3B 复现训练阶段（第二阶段门禁另行预注册）；
  FAIL-TO-OBSERVE → 停止训练计划，先诊断环境（检索质量对比优先）；
  INCONCLUSIVE → 不作为环境不一致证据，结合次要指标与后续诊断定方向。

## 次要指标（仅描述性）

| 指标 | Base | 官方 |
|---|---|---|
| 检索次数（成功/其余） | 124（{"invalid_query": 114, "success": 10}） | 40（{"success": 40}） |
| error observation 步 | 114/380 | 0/296 |
| format_scored（0.1）episode | 112 | 184 |
| answer_compliance rate | 0.516 | 0.844 |

分源 EM：{"2wikimultihopqa": {"n": 32, "base_em": 7, "official_em": 4}, "bamboogle": {"n": 16, "base_em": 0, "official_em": 2}, "hotpotqa": {"n": 64, "base_em": 0, "official_em": 3}, "musique": {"n": 16, "base_em": 0, "official_em": 0}, "nq": {"n": 64, "base_em": 8, "official_em": 9}, "popqa": {"n": 32, "base_em": 1, "official_em": 3}, "triviaqa": {"n": 32, "base_em": 4, "official_em": 11}}

生成对比：0 个 episode 逐字节一致，
全部对齐步 0/274，
平均归一化编辑距离 0.6750。

## 声明边界（预注册第 7 节）

- 单 seed、greedy、官方宽松语义下的单次验证；判定"我们的评测链路能否观察官方
  训练效果"，不是官方模型在官方环境上的复现成绩；
- 本实验数字不与严格线或论文数字直接对照；
- discordant 逐题明细见随附 JSON（`p3_official_pair_2026-08-15.json`）。
