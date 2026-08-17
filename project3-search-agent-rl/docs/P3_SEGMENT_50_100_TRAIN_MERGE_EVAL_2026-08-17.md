# P3 正式段 Step 50–100：训练 + 合并 + 开发评测记录（2026-08-17）

**授权**：用户批准正式训练 Step 50–100（仅从正式 0–50 段产物
`global_step_50` 恢复，严禁 Base 重启与任何 smoke checkpoint；GPU 1,2,3,4,6,7，
严禁 GPU0/5；冻结 50–100 段配置 full SHA `910f216c…` / invariant SHA
`2cc743a3…`）→ 成功后复用已验证 merge 链路生成独立 HF 模型 → 仅 GPU1 评测
official-confirm256-v1（不得运行/查看 final-confirm512）。**本文件记录训练、
合并、评测全链路与「EM 提升来源」判定；p 值不作开发门禁**。

---

## 1. 训练：p3-formal-segment-50-100-fsdp6-b66-n5-s0-20260817a

### 1.1 启动前基线（2026-08-17）

| 项 | 值 |
|---|---|
| 恢复来源 | 正式 0–50 段 `runs/p3-formal-segment-0-50-fsdp6-b66-n5-s0-20260816a/checkpoints/global_step_50`（非 Base、非 smoke） |
| 配置指纹（三重验证） | 占位 full SHA `910f216cdf5bd27c46b013918948bb3cf93c9f4e09d86bf67143e100080eea21` ✓；invariant SHA `2cc743a3cedbd957518717f7d47b0f1c3fe060abb07d92fe84b71cd270339674`（真实 run 路径）✓；overrides 与冻结配置逐项一致 |
| GPU | 1,2,3,4,6,7；GPU0/5 全程未用（含 ~387 MiB 系统空闲基线之外零占用） |
| Retriever | `127.0.0.1:18080` ready，21,015,324 向量，全程健康 |
| 磁盘 | 充足（>1.5 TiB） |

**启动前 gs50 只读恢复审计**（`scripts/audit_p3_checkpoint_resume.py`，PASS）：

- 文件完整：6× model + 6× optim + 6× extra_state + data.pt，27/27；
- scheduler：last_epoch=50（_step_count=51），LR=5.882e-7 = 50/85 × 1e-6，
  与冻结 warmup 曲线（`lr = base × min(epoch/85, 1.0)`）逐点一致；
- optimizer：37 个 state 条目、step=100（= 50 步 × ppo_epochs 2）、
  exp_avg/exp_avg_sq 非零——非重新初始化；
- DataLoader：`_snapshot_step=50`，恢复后消费**下一批**（epoch 内 batch 50+）；
- RNG：rank0 extra_state 含 cpu/cuda/numpy/random 种子状态；
- 源 checkpoint 全量 SHA256 manifest 在案（`gates/gs50_before_sha256_20260817a.txt`），
  训练结束后复核字节不变（见 §2）。

### 1.2 运行参数（冻结 50–100 段，与 0–50 段一致）

66 prompts × group 5 = 330 samples；全参 FSDP（world_size 6）；param/optimizer/ref
offload=true；gpu_mem 0.60；max_num_seqs 64；lr=1e-6；warmup_steps=85；kl low_var
0.001；entropy 0；seed 1234/1234；save_freq=50；schedule horizon 300（patch 0006）；
`segment_stop_step=100`；started_at 2026-08-17T11:19:37，finished_at
2026-08-17T18:55:53（7h36m / 50 步，exit_code=0）。

### 1.3 运行健康（约 7 小时监控，无 ALERT）

- 无 NaN/Inf、无 OOM、无 GPU 掉卡/Xid；GPU0/5 持续空闲；retriever 全程 ready；
- 无配置指纹不一致；checkpoint 每步保存完整；
- 节奏 532–567 s/step（Step 51: 566.5 s → Step 75: 532.4 s → Step 100: 557.6 s），
  与 0–50 段末段节奏一致；
- cleanup 六卡 `compute_processes=none`；Ray 优雅停止，无残留进程。

### 1.4 快照（Step 51/75/100，授权记录点；未中途评测/调参）

| step | reward/mean | success_rate | tool_call/mean | response_len | timing/step |
|---|---|---|---|---|---|
| 51 | 0.118 | 0.021 | 0.058 | — | 566.5 s |
| 75 | 0.138 | 0.042 | 0.018 | — | 532.4 s |
| 100 | 0.178 | 0.088 | 0.027 | 136.5 | 557.6 s |

（Step 51 相对 gs50 的 0.152/0.061 回落为单步 batch 波动；Step 100
critic/rewards/mean 0.177、advantages mean -0.031；gen 37.7 s / old_log_prob
95.3 s / ref 100.9 s / update_actor 307.6 s。tool_call 总体趋势：gs50 0.082 →
Step 75 0.018 → Step 100 0.027，搜索调用持续被压缩。）

### 1.5 Step 100 验收（PASS）

- `SEGMENT_STOP` 唯一出现：`SEGMENT_STOP: finished global_step 100
  (segment_stop_step=100); checkpoint saved with DataLoader/Optimizer/Scheduler/RNG
  state; returning normally (schedule horizon 300 untouched)`；
- exit_code=0；cleanup 六卡 `compute_processes=none`；GPU 回基线 0%；
- **gs100 只读审计 PASS**：scheduler last_epoch=100、LR=1e-6（warmup 85 后恒定，
  与 0–50 段曲线逐点连续）；optimizer step=200（= 100 × 2）；DataLoader
  `_snapshot_step=100`；RNG 已保存；
- **无重复 0–50 数据流证明**：`samples_yielded` gs50=3300 → gs100=6600
  （= 100 × 66 恰等），50–100 段消费 epoch 内 batch 50–99，与 0–50 段零重叠；
- 源 gs50 checkpoint 合并前后 SHA 复核 27/27 OK（字节不变）。

---

## 2. 合并：FSDP → HF（复用已验证链路）

**命令**（同 0–50 段，纯 CPU）：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/vendor/verl-agent:$PWD" \
  envs/searchr1-repro-cu124/bin/python vendor/verl-agent/scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir  runs/p3-formal-segment-50-100-fsdp6-b66-n5-s0-20260817a/checkpoints/global_step_100/actor \
  --target_dir models/p3-formal-segment-50-100-gs100-merged-20260817a
```

- `MERGE_EXIT=0`；device mesh `[0..5]`，6 shards 加载合并；
- **源字节不变证明**：合并前 gs100 27 文件全量 SHA manifest
  `gates/gs100_before_sha256_20260817a.txt`；合并后 `sha256sum -c` → 27/27 OK；
- **产物验证**（`scripts/verify_p3_merged_model.py`，VERIFY_MERGED: PASS）：
  文件齐全、参数量 3,397,103,616（tie 保持，lm_head 与 embed_tokens 逐字节
  相等）、NaN/Inf=0、missing/unexpected 均空、dtype bf16、总大小 6,810,172,509 B。

---

## 3. 开发评测：official-confirm256-v1（仅 GPU1，三方并列）

**Run ID**：`p3-eval-official-confirm256-gs100-20260817a`
**日志**：`runs/p3-eval-official-confirm256-gs100-20260817a/`（exit_code=0，
86.8 s）
**入口**：`run_p3_eval_vllm_official.sh`（official-loose 语义，vLLM 0.8.5.post1
V0 greedy bf16 0.6 util TP1，tokenizer 固定 Base，seed 0 max_steps 2 /
history_length 2 / topk 3 / timeout 180，真实 Wiki-18 21,015,324 向量 health 门禁，
data SHA `ffebf468e756…` ✓、leakage 0/256、patch 检查、`CUDA_VISIBLE_DEVICES=="1"`
硬门禁）。**未查看、未运行 final-confirm512**。

### 3.1 三方并列（Base / gs50 / gs100，配对 256/256）

| 指标 | Base | Step 50 | Step 100 |
|---|---|---|---|
| EM | 20/256 = 7.81% | 30/256 = 11.72% | **38/256 = 14.84%** |
| answer compliance | 51.56% | 85.16% | **95.31%** |
| 搜索执行 | 124 次（success 10 / invalid_query 114） | 38 次（success 3 / invalid_query 35） | **12 次（success 1 / invalid_query 11）** |
| 总步数 | 380 | 294 | **268** |
| error 观测 | 114 | 35 | **11** |
| 1 步完成的题目 | 132/256 | 218/256 | **244/256** |
| reward 分布（0.0 / 0.1 / 1.0） | 124 / 112 / 20 | 38 / 188 / 30 | **12 / 206 / 38** |

（compliance 口径自洽：reward≥0.1 的题目 = 格式合规题目；gs100
206+38=244 = 95.31% ✓，Base 112+20=132 = 51.56% ✓。）

**配对 EM 转换**（p 值仅描述性，不作开发门禁）：

| 转换 | 0→1 | 1→0 | McNemar 精确双侧 p |
|---|---|---|---|
| Base→gs50 | 16 | 6 | 0.0525 |
| Base→gs100 | **22** | **4** | **0.0005** |
| gs50→gs100 | 11 | 3 | 0.0574 |

分源 EM（base / gs50 / gs100）：nq 8/7/**11**、hotpotqa 0/3/3、popqa 1/3/**4**、
2wikimultihopqa 7/7/**8**、triviaqa 4/8/**10**、musique 0/1/0、bamboogle
0/1/**2**。
被破坏题目（Base→gs100 的 1→0）仅 4 题：2wikimultihopqa ×3、nq ×1；修复 22 题中
triviaqa 6、2wikimultihopqa 4、nq 4、hotpotqa 3、popqa 3、bamboogle 2。

### 3.2 观测与判定：EM 提升**不是**来自学会搜索

**判定：EM 改善主要来自格式合规与行为收敛（抑制无效搜索），不是学会搜索。**
决定性证据链：

1. **零搜索依赖**：gs100 全部 38 个正确答案中，**0 个执行过搜索**。gs100 仅有的
   1 次真正成功检索（success status）对应的题目仍然答错（reward 0.1，Base 侧
   该题未搜索、gs50 搜索 success 也答错）。
2. **被纠正题目无搜索参与**：Base→gs100 的 22 个 0→1 题目中，gs100 侧搜索 0 次、
   全部直接作答（Base 侧 16 题曾尝试搜索、其中仅 1 题有真正成功检索且未答对）。
   没有任何一题通过「搜索 → 检索 → 答对」链路完成。
3. **搜索行为被压缩而非增强**：搜索调用 124→12（真正 success 10→1），且
   12 次中 11 次仍是 invalid_query（`<search>` 带空 query）。模型学到的策略是
   「不搜、直接答」，不是「更会搜」。
4. **增益结构**：reward 0.0 从 124→12（几乎每题都输出格式正确的 `<answer>`），
   0.1（格式对、答案错）从 112→206；EM（1.0）仅 +18。主要增益落在
   「完全不得分 → 格式合规」路径，而非「搜索正确 → 答对」路径。
5. **机制解释**：official-loose 语义下 invalid_query 不惩罚、error_observation
   仅重试不惩罚——搜索成功与搜索失败在奖励上几乎无差别，RL 自然收敛到
   「零搜索 + 高格式」策略；triviaqa/popqa/bamboogle 等短事实题靠参数记忆直接
   作答受益最大，与「直接作答策略」一致。

**含义**：Step 100 模型是「参数记忆 + 高格式合规」模型，不是「搜索 agent」。
继续训练（100–300）在现有奖励设计下有进一步压缩搜索行为的风险；最终确认评测
（final-confirm512）将验证这一行为在更大评测集上的表现。

---

## 4. 边界与后续

- 本段完成训练、合并、开发评测全部授权动作；**不自动启动 Step 100–300**。
- 全程未查看/运行 final-confirm512；三段冻结配置未修改；未重新预注册。
- 授权要求的两处文档修正（0–50 段文档 §1.3 显存口径 34.7 GiB = 跨 rank 聚合
  机器级观测值而非单卡占用、§1.4 趋势改述为四授权快照依次上升）已随前一提交
  `68e1bfa` 完成并推送，不涉及配置重冻结。
- Step 100–300 训练与 final-confirm512 评测须另行批准后启动。
