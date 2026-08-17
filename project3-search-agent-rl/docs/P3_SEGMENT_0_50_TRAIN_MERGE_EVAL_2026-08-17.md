# P3 正式段 Step 0–50：训练 + 合并 + 开发评测记录（2026-08-17）

**授权**：用户批准正式训练 Step 0–50（从 Qwen2.5-3B Base 启动，严禁 smoke/resume
checkpoint；GPU 1,2,3,4,6,7；冻结配置 0–50 段）→ 成功后复用已验证 merge 链路生成
独立 HF 模型 → 仅 GPU1 评测 official-confirm256-v1。**本文件记录训练、合并、评测
全链路；p 值不作开发门禁**（预注册 §3.1 门禁结构）。

---

## 1. 训练：p3-formal-segment-0-50-fsdp6-b66-n5-s0-20260816a

### 1.1 启动前基线（2026-08-16 21:04:56 +08:00）

| 项 | 值 |
|---|---|
| 代码 commit | `85b7b1f`（gate(p3): checkpoint merge + eval chain gate PASS） |
| 模型 | Qwen2.5-3B Base（10 文件 SHA 在案；tokenizer 固定 Base，含 eval 复用） |
| train 数据 | `datasets/searchr1-upstream/train.parquet` SHA256 `aa98bf95dec9466899395e5d44e56e1b765cef7bc6b9ea226f5e6129bd0d360a` |
| 配置指纹（三重验证） | 占位 full SHA `4a472c90adf4a8d9a38f007f52b1007802c21638f6a46552efa15f34f345ffc6` ✓；invariant SHA `2cc743a3cedbd957518717f7d47b0f1c3fe060abb07d92fe84b71cd270339674`（占位+真实 run 均匹配）✓；per-run resolved_config_sha256 `c6694cd1a921bf39b511442e52401575cb5e71c8619176e5d80b2ea6492b5e28` |
| GPU 基线 | 1,2,3,4,6,7 空闲；GPU0/5 全程未用 |
| Retriever | `127.0.0.1:18080` ready，21,015,324 向量 |
| 磁盘 | 2.7 TiB 可用 |

### 1.2 运行参数（冻结 0–50 段）

66 prompts × group 5 = 330 samples；全参 FSDP（world_size 6）；param/optimizer/ref
offload=true；gpu_mem 0.60；max_num_seqs 64；lr=1e-6；warmup_steps=85；kl low_var
0.001；entropy 0；seed 1234/1234；save_freq=50；schedule horizon 300（patch 0006）；
`segment_stop_step=50`；resume_mode=disable；started_at 2026-08-16T21:04:56，
finished_at 2026-08-17T07:35:05（exit_code=0）。

### 1.3 运行健康（全程监控，无停止条件触发）

- 无 NaN/Inf、无 OOM（peak max_memory_reserved 34.7 GiB/卡，24.5 GiB 卡 × 6 全参
  FSDP + offload 预期内）、无 GPU 掉卡/Xid、GPU0/5 持续空闲 0%；
- 422 API 拒绝 = 既有宽松语义行为（smoke 亦同，~0.6% 请求），retriever 全程健康；
- 无配置指纹不一致、checkpoint 每步保存完整；
- 节奏 ~866 s/step → 逐步加速至 Step 50 的 585 s/step（update_actor 495→323 s）。

### 1.4 快照（Step 1/10/25/50，授权记录点；未中途挑选/评测）

| step | reward | success_rate | tool_call/mean | response_len | timing/step |
|---|---|---|---|---|---|
| 1 | 0.064 | 0.018 | 0.642 | 164.5 | 864.3 s |
| 10 | 0.081 | 0.036 | 0.648 | 162.4 | 859.3 s |
| 25 | 0.101 | 0.045 | 0.564 | 172.9 | 819.2 s |
| 50 | 0.152 | 0.061 | 0.082 | 161.7 | 585.2 s |

趋势：训练内 reward 与 success 单调上升（0.064→0.152、0.018→0.061）；Step 50
critic/rewards/mean 0.147、max 1.0、advantages mean -0.012；gen 39.3 s / old_log_prob
100.3 s / ref 105.8 s / update_actor 323.3 s。

### 1.5 Step 50 验收（PASS）

- `SEGMENT_STOP` 标记 ✓（stdout.log 唯一一次）：
  `SEGMENT_STOP: finished global_step 50 (segment_stop_step=50); checkpoint saved with
  DataLoader/Optimizer/Scheduler/RNG state; returning normally (schedule horizon 300 untouched)`
- exit_code=0；cleanup.log 六卡 `compute_processes=none`；GPU 回基线 0%；
- `checkpoints/global_step_50/` 完整：model 6/6、optim 6/6、extra_state 6/6 rank +
  `data.pt`（DataLoader 状态）+ tokenizer/config 全套，36 GiB；
- Ray worker/register center 正常优雅停止，无残留训练进程。

---

## 2. 合并：FSDP → HF（复用已验证链路）

**命令**（与门禁 2026-08-16 相同，纯 CPU）：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/vendor/verl-agent:$PWD" \
  envs/searchr1-repro-cu124/bin/python vendor/verl-agent/scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir  runs/p3-formal-segment-0-50-fsdp6-b66-n5-s0-20260816a/checkpoints/global_step_50/actor \
  --target_dir models/p3-formal-segment-0-50-gs50-merged-20260817a
```

- `MERGE_EXIT=0`；device mesh `[0..5]`，6 shards 加载合并。
- **源字节不变证明**：合并前 27 文件全量 SHA256 manifest `/tmp/gs50_before.sha256`
  （含 data.pt）；合并后 `sha256sum -c` → **27/27 OK（CHECK_EXIT=0）**。
- **产物验证**（`scripts/verify_p3_merged_model.py`，VERIFY_MERGED: PASS）：
  - 文件齐全：config/generation_config/model.safetensors.index.json/tokenizer 全套
    + model-00001/00002-of-00002.safetensors；
  - 参数量 3,397,103,616 = Base 3,085,938,688 + 151,936×2,048（独立 lm_head.weight
    副本，**与 embed_tokens 逐字节相等**，tie 保持）；
  - NaN/Inf = 0；missing/unexpected keys 均空；dtype bf16；总大小 6,810,172,509 B
    （6.34 GiB）；
  - 逐文件 SHA256 记录于产物 manifest（model-00001
    `52cbedb8ccddda238ea0b4cbc519d3bd6f7bc098a8c9fb46a6da19d5a6913c7e`、model-00002
    `62da3cf65c5329652e93464eb0c89a4238f142e3b919345143e0e44d2b22edd6`、
    index `46e884d8996e55986cf9dd063597e5a2d76a6c9f0192475cb0d7643ebf97b80f`）。

---

## 3. 开发评测：official-confirm256-v1（仅 GPU1）

**Run ID**：`p3-eval-official-confirm256-gs50-20260817a`
**日志**：`runs/p3-eval-official-confirm256-gs50-20260817a/`（exit_code=0，
10:31:48→10:33:47，109.5 s）
**入口**：`run_p3_eval_vllm_official.sh`（official-loose 语义，vLLM 0.8.5.post1
V1=0 greedy bf16 0.6 util TP1，tokenizer 固定 Base，seed 0 max_steps 2 /
history_length 2 / topk 3 / timeout 180，真实 Wiki-18 21,015,324 向量 health 门禁，
data SHA `ffebf468e756…` vs manifest ✓、leakage 0/256、patch 0001–0004 检查、
`CUDA_VISIBLE_DEVICES=="1"` 硬门禁）。
**未查看、未运行 final-confirm512**。

### 3.1 结果 vs Base（配对描述，p 值不作门禁）

| 指标 | Base（s0-20260815a） | Step 50（本 run） |
|---|---|---|
| EM | 20/256 = 7.81%（Wilson [5.11%, 11.76%]） | **30/256 = 11.72%**（+3.91pp） |
| answer compliance | 51.56% | **85.16%**（+33.6pp） |
| 搜索执行 | 124 次（success 10 / invalid_query 114） | **38 次**（success 3 / invalid_query 35） |
| 总步数 / error 观测 | 380 / 114 | **294 / 35** |
| 泄漏 | 0 | 0（eval 256 ∩ reference 8） |
| peak GPU | — | 13.78 GiB allocated（无 OOM） |

**配对明细**（256/256 配对成功）：0→0 共 220、**0→1（base 错 step50 对）16**、
1→0（base 对 step50 错）6、1→1 共 14。McNemar 精确双侧 p=0.0525（**描述性报告，
不作开发门禁**）；方向：Step 50 净增益 10（16−6），EM 提升主要来自 16 个
Base 错答被纠正，6 个 Base 原本正确的被破坏。

分源 EM：triviaqa 25.0%（8/32）、2wiki 21.9%（7/32）、nq 10.9%（7/64）、
popqa 9.4%（3/32）、musique 6.25%、bamboogle 6.25%、hotpotqa 4.7%（3/64）。

### 3.2 观测

- 训练后模型显著提高 `<answer>` 格式合规（51.6%→85.2%）并减少无效搜索调用
  （124→38），error 观测 114→35——行为方向与训练奖励一致（format_score + 宽松
  搜索语义）；
- EM 相对 Base 方向为正向（+3.91pp），配对 0→1 多于 1→0（16 vs 6），但 McNemar
  p=0.0525 未达 0.05 阈值——**按预注册，开发门禁不设统计判据，此数字仅作
  方向性描述**；最终确认性检验由 final-confirm512 承担（另行批准后执行）。

---

## 4. 边界与后续

- 本段完成训练、合并、开发评测全部授权动作；**不自动启动 Step 50–100**。
- 全程未查看/运行 final-confirm512；三段冻结配置未修改；未重新预注册。
- Step 50–100 训练与 Step 300 final 评测须另行批准后启动。
- 若后续需要：Step 50–100 从 `checkpoints/global_step_50` 恢复（resume_mode 由
  fail-closed 强制 + 预注册 0-50 段一致；scheduler horizon 300 与 RNG 状态已在
  extra_state/data.pt 中保存，patch 0006 保证三段 LR 曲线逐点一致）。
