# P3 正式段 Step 100–300：训练 + 合并 + final-confirm512 盲测记录（2026-08-19）

**授权**：用户批准正式训练 Step 100–300（仅从正式 50–100 段产物 `global_step_100`
恢复；GPU 1,2,3,4,6,7 严禁 GPU0/5；冻结 100–300 段配置 full SHA `758326c6…` /
invariant SHA `2cc743a3…`；horizon=300、segment_stop_step=300；启动前只读审计
gs100；记录授权快照点，不中途评测/调参；遇 OOM/NaN/Xid/GPU0,5/retriever 最终失败
/恢复异常立即停止）→ 成功后合并生成独立 HF 模型 → **final-confirm512 一次性盲测**
（运行前只校验数据 SHA `94b39266…`，不打开题目，依次完整评测 Base → Step 300 →
官方 Search-R1，**三个模型全部完成后才统一揭示分析**；主判定仅 Base vs Step 300
配对 EM/McNemar；官方 checkpoint 仅描述性参考）。**本文件记录训练、合并、盲测、
主判定与机制解释全链路。探索性机制指标见 dated addendum
`P3_FINAL_EVAL_ADDENDUM_2026-08-17.md`，不改变任何预注册判据。**

---

## 1. 训练：p3-formal-segment-100-300-fsdp6-b66-n5-s0-20260817b

### 1.1 启动前核验（2026-08-17）

| 项 | 值 |
|---|---|
| 恢复来源 | 正式 50–100 段 `runs/p3-formal-segment-50-100-fsdp6-b66-n5-s0-20260817a/checkpoints/global_step_100`（非 Base、非 smoke、非 gs50） |
| 配置指纹 | 冻结段 3 full SHA `758326c6f923bc39707b1d09668d5a10c710a09a8bf87358d3f74cee6c331303` ✓；invariant `2cc743a3cedbd957518717f7d47b0f1c3fe060abb07d92fe84b71cd270339674` ✓；`--dump-overrides` 真实路径 invariant MATCH；per-run `resolved_config_sha256=99217d62…` |
| GPU | 1,2,3,4,6,7；GPU0/5 全程 18–387 MiB 系统基线，零训练占用 |
| Retriever | `127.0.0.1:18080` ready，21,015,324 向量，全程健康 |

**启动前 gs100 只读恢复审计**（`scripts/audit_p3_checkpoint_resume.py`，PASS）：
27/27 文件；scheduler last_epoch=100（LR=5.882e-7→1e-6 曲线逐点一致）；
optimizer step=200 非零；DataLoader `_snapshot_step=100`；samples_yielded=6600
（=100×66，与 0–100 段零重叠）；源 gs100 27 文件 SHA manifest 在案
（`gates/gs100_seg3_before_sha256_20260817b.txt`）。

### 1.2 运行参数（冻结 100–300 段，与前两段一致）

66 prompts × env.rollout.n 5 = 330 samples/步；全参 FSDP（world_size 6）；param/
optimizer/ref offload=true；gpu_mem 0.60；max_num_seqs 64；lr=1e-6；
warmup_steps=85；ppo_epochs=1（hydra resolved）；kl low_var 0.001；entropy 0；
seed 1234/1234；save_freq=50（保留 gs150/200/250/300）；schedule horizon 300
（patch 0006）；`segment_stop_step=300`；started_at 2026-08-17T20:45:21，
finished_at 2026-08-19T01:11:12（28 小时 26 分 / 200 步，exit_code=0）。

### 1.3 运行健康（全程 cron 监控 `monitor_p3_segment_run.sh`，零 ALERT）

- 无 OOM、无 NaN/Inf、无 GPU 掉卡/Xid；GPU0/5 持续空闲；retriever 全程 ready；
  无配置指纹缺失；checkpoint 每步保存完整。
- 节奏 497.9–536.1 s/step；两次"停滞"观感（step 105→106、189→190、286→289）
  均核实为 vLLM rollout（~16.3 GiB/卡）与 update_actor（offload，~2 GiB/卡）
  相位切换，rollouts/ 文件号持续前进，非真实停滞。

### 1.4 快照（Step 101/150/200/250/300，授权记录点；未中途评测/调参）

| step | reward/mean | success_rate | tool_call/mean | response_len | timing/step |
|---|---|---|---|---|---|
| 101 | 0.192 | 0.103 | 0.012 | — | 536.1 s |
| 150 | 0.231 | 0.145 | **0.000** | 50.2 | 515.2 s |
| 200 | 0.307 | 0.230 | 0.000 | 36.9 | 512.8 s |
| 250 | 0.264 | 0.182 | 0.000 | 46.0 | 513.1 s |
| 300 | 0.255 | 0.173 | 0.000 | 38.1 | 511.9 s |

（Step 250/300 较 200 的回落为单步 batch 波动；tool_call 自 Step 150 起恒为 0.000，
搜索行为在训练段完全压缩，与 Step 100 开发集判定一致。）

### 1.5 Step 300 验收（PASS）

- **SEGMENT_STOP 唯一出现**：`SEGMENT_STOP: finished global_step 300
  (segment_stop_step=300); checkpoint saved with DataLoader/Optimizer/Scheduler/RNG
  state; returning normally (schedule horizon 300 untouched)`（stdout.log 唯一 1 次）；
- exit_code=0；cleanup 六卡 `compute_processes=none`；GPU 回基线 0%；
- **gs300 只读审计 PASS**：27/27 文件（六 rank model/optim/extra_state + data.pt +
  tokenizer 文件）；scheduler last_epoch=300、LR=1e-6；optimizer 37 个 state 条目、
  step=477 非零（**见下方说明**）；DataLoader `_snapshot_step=300`；RNG 已保存；
- **samples_yielded=19800 = 300×66 恰等**（`_snapshot/_main_snapshot/
  _sampler_iter_state/samples_yielded`），100–300 段消费 epoch 内 batch 100–299，
  与 0–100 段零重叠，无重复数据流；
- 源 gs100 checkpoint 合并前后 SHA 复核 27/27（`gates/gs100_seg3_before_sha256_
  20260817b.txt`），字节不变。

**optimizer step=477 的说明（验收期望修正，非异常）**：授权预期"optimizer step
=600"基于 ppo_epochs=2 的假设；实际冻结配置 `ppo_epochs=1`（hydra resolved
config.yaml `actor_rollout_ref.actor.ppo_epochs: 1`），optimizer step 按
**实际 minibatch 数**计数（`update_policy` 内每 (epoch, minibatch) 一次
`optimizer.step()`，minibatch = ceil(batch/ppo_mini_batch_size=330)）。
batch 基准 = 66 prompts × n=5 = 330（rollouts jsonl 每行 1 样本）：
0–100 段 100/100 步 batch>330 → 每步 2 minibatch → gs50=100、gs100=200；
100–300 段仅 77/200 步 batch>330 → 增量 = 200×1 + 77×1 = 277 → gs300=477。
**rollouts 行数 >330 的文件数与 step 增量完全吻合（50/50 vs 77/200）**，
三段自洽、无异常、无跳过（全程 0 次 grad_norm 非有限 WARN）。

---

## 2. 合并：FSDP → HF（复用已验证链路）

**命令**（同前两段，纯 CPU）：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/vendor/verl-agent:$PWD" \
  /media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124/bin/python \
  vendor/verl-agent/scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir  runs/p3-formal-segment-100-300-fsdp6-b66-n5-s0-20260817b/checkpoints/global_step_300/actor \
  --target_dir models/p3-formal-segment-100-300-gs300-merged-20260817b
```

- `MERGE_EXIT=0`；device mesh `[0..5]`，6 shards 加载合并；
- **源字节不变证明**：合并前 gs300 27 文件全量 SHA manifest
  `gates/gs300_before_sha256_20260817b.txt`；合并后 `sha256sum -c` → 27/27；
- **产物验证**（`scripts/verify_p3_merged_model.py`，VERIFY_MERGED: PASS）：
  文件齐全、参数量 3,397,103,616（tie 保持，lm_head 与 embed_tokens 逐字节
  相等）、NaN/Inf=0、dtype bf16、总大小 6,810,172,573 B。

---

## 3. final-confirm512 一次性盲测（2026-08-19）

**盲测协议执行**：运行前只校验 `heldout.parquet` SHA
`94b39266c2d9c54a55b4471e90daa493ab083a889d8f23510dadd8194b304ecc`（manifest
`outputs.heldout.sha256` 同值，512 行）；**未人工打开题目**；依次完整评测
Base → Step 300 → 官方 Search-R1（仅 GPU1，同一冻结评测入口
`run_p3_eval_vllm_official.sh`：official-loose 语义、vLLM V0 greedy、tokenizer
固定 Base、retriever health 门禁 21,015,324 向量、data SHA 自动校验）；**三个
模型全部 exit_code=0 后才统一揭示**。

**Run ID**：`p3-eval-final-confirm512-{base,gs300,searchr1}-20260819a`
**结果**：`runs/<run-id>/episodes.jsonl`（512/512 配对，三模型行序一致）

### 3.1 三方并列（配对 512/512）

| 指标 | Base | Step 300 | 官方 Search-R1* |
|---|---|---|---|
| **EM** | 47/512 = 9.18% | **115/512 = 22.46%** | 88/512 = 17.19% |
| Wilson 95% CI | [7.0%, 12.0%] | [19.1%, 26.3%] | [14.1%, 20.8%] |
| answer compliance | 52.7% | **99.8%** | 81.8% |
| 搜索执行 | 242 次（success 19 / invalid_query 223） | **1 次（success 1）** | 93 次（success 90 / invalid_query 3） |
| 总步数 | 754 | **513** | 605 |
| error 观测 | 223 | **0** | 3 |
| 1 步完成的题目 | 270/512 = 52.7% | **511/512 = 99.8%** | 419/512 = 81.8% |
| reward 分布（0.0 / 0.1 / 1.0） | 242 / 223 / 47 | **1 / 396 / 115** | 93 / 331 / 88 |

（\* 官方 Search-R1 checkpoint 仅描述性参考，不参与判定。）
（compliance 口径自洽：reward≥0.1 = 1 步格式合规题；Step300 396+115=511 = 99.8% ✓。）

### 3.2 主判定（预注册确认性检验，仅 Base vs Step300）

- 配对转换：**0→1 = 74 题，1→0 = 6 题**；
- McNemar 精确双侧 p = 2·P(X≤6), X~Bin(80, 0.5) < 0.0001；
- EM 差 = +13.28 pp（Wald 95% CI [+8.9%, +17.7%]）；
- **判定：PASS**（Step300 EM 22.46% > Base 9.18%，p < 0.05）。

**discordant pairs 分源**（0→1）：nq 18、hotpotqa 15、triviaqa 14、
2wikimultihopqa 12、popqa 11、musique 2、bamboogle 2（全部 7 源均有净增益）；
1→0 仅 6 题（triviaqa 4、popqa 1、bamboogle 1）。

分源 EM（Base / Step300 / SearchR1）：nq 9/**27**/16、hotpotqa 7/**22**/16、
triviaqa 19/**29**/27、2wikimultihopqa 4/**16**/13、popqa 3/**13**/7、
bamboogle 5/6/7、musique 0/**2**/2。**Step300 在全部 7 源上 ≥ Base。**

### 3.3 机制指标（探索性，dated addendum 口径）

| 指标 | Base | Step 300 | 官方 Search-R1 |
|---|---|---|---|
| search→correct | 0 | 0 | 0 |
| search→wrong | 242 | 1 | 93 |
| no-search→correct | 47 | **115** | 88 |
| 每题搜索次数 | 0.473 | **0.002** | 0.182 |
| 一步完成率 | 52.7% | **99.8%** | 81.8% |
| 格式合规 | 52.7% | **99.8%** | 81.8% |

**判定：EM 提升不是来自学会搜索。** 证据链（与 Step 100 开发集结论一致）：

1. **零搜索依赖**：Step300 全部 115 个正确答案**零搜索依赖**；其唯一 1 次搜索
   （status=success 的真正检索）对应的题目仍答错（reward 0.1）。
2. **被纠正题目无搜索参与**：74 个 0→1 题中 Step300 侧搜索 0 次、全部直接作答。
3. **搜索被压缩而非增强**：242→1 次，且唯一一次是 success 检索（成功检索仍答错）。
4. **官方 Search-R1 对照（描述性）**：官方 checkpoint 做了 90 次真正成功检索，
   **search→correct = 0**——在该评测集/评测语义（official-loose：invalid 不惩罚、
   error 仅重试）下，搜索从未带来任何 EM 收益；RL 在 3B 规模自然收敛到
   「不搜、直接答、格式全合规」策略。
5. **增益结构**：reward 0.0 从 242→1（几乎每题输出格式正确的 `<answer>`），
   compliance 52.7%→99.8%，EM（1.0）+68。主要增益落在「完全不得分 →
   格式合规」路径，短事实题（nq/triviaqa/popqa/hotpotqa）靠参数记忆直答受益。

**含义**：Step 300 模型是「参数记忆 + 高格式合规」模型，不是「搜索 agent」。
这与 256 题开发集在 512 题盲测集上完全复现。探索性指标解释结果，不改变
主判据 PASS 的结论。

---

## 4. 边界与后续

- 本段完成训练、合并、盲测、判定全部授权动作；**不自动启动奖励改进训练**。
- 全程未人工打开 final-confirm512 题目；三段冻结配置未修改；未重新预注册；
  dated addendum（`P3_FINAL_EVAL_ADDENDUM_2026-08-17.md`）声明主判据唯一性，
  官方 Search-R1 仅描述性参考。
- 验收中唯一偏离预期值的是 optimizer step（477 vs 期望 600），已完整解释为
  ppo_epochs=1 下的 minibatch 计数（§1.5），三 checkpoint 轨迹自洽，非异常。
- 监控 cron 已停止；评测 tmux/GPU 已清理；训练 run 的 gs150/200/250/300
  checkpoint、rollouts、日志全部保留。
