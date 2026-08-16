# P3 Checkpoint 合并与评测链路门禁（2026-08-16）

**授权范围**：仅本门禁（checkpoint 合并 + 合并产物验证 + GPU1 smoke-16 评测链路）。
**不授权**：正式 Step 0–50 训练（冻结设计审阅通过但暂缓，另行单独批准）。
**输入**：已成功的 resume smoke `global_step_2` FSDP checkpoint；全程未触碰、未修改
正式训练目录（`runs/p3-formal-segment-*`）；三段冻结配置与预注册未修改。

## 1. 门禁目的

正式训练 Step 50/100/300 的 FSDP 分片 checkpoint 需转换为 HF 权重后评测。
本门禁用 resume smoke global_step_2 提前打通「FSDP checkpoint → verl model_merger →
HF 模型 → Transformers/vLLM 加载 → 官方宽松评测入口 smoke-16」全链路，
消除"训完却无法评测"的工程风险。**不以 EM 高低判定**；smoke-16 仅验证
加载、生成、搜索协议与评测链路。

## 2. 合并命令（verl 官方工具，纯 CPU，未用 GPU）

```sh
cd /home/imc/yzy/agent/project3-search-agent-rl
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/vendor/verl-agent:$PWD" \
  /media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124/bin/python \
  vendor/verl-agent/scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir  /media/imc/data/project3-search-agent-rl/runs/p3-official-offload-resume-smoke-fsdp6-b66-n5-s0-20260816b/checkpoints/global_step_2/actor \
  --target_dir /media/imc/data/project3-search-agent-rl/models/p3-official-smoke-gs2-merged-20260816a
```

- 输入：`p3-official-offload-resume-smoke-fsdp6-b66-n5-s0-20260816b`（resume smoke，
  exit_code=0）`checkpoints/global_step_2/actor/`，6-rank FSDP 分片
  （`model_world_size_6_rank_{0..5}.pt` + optimizer/extra_state + config/tokenizer）。
- 上游 pin `20bd331b…`（vendor/verl-agent），patches 0001–0006 已应用。
- 输出（全新目录，合并前不存在）：
  `/media/imc/data/project3-search-agent-rl/models/p3-official-smoke-gs2-merged-20260816a/`
- 日志：device mesh `[0..5]` (fsdp)，6 shards 加载合并，
  `MERGE_EXIT=0`（完整输出见本 gate 归档 / run 日志）。

## 3. 源 checkpoint 字节不变证明（合并前后全量 SHA256）

- 合并前：`find checkpoints/global_step_2 -type f | xargs sha256sum` → 27 文件 / 36 GiB，
  manifest `/tmp/gs2_before.sha256`。
- 合并后：`sha256sum -c /tmp/gs2_before.sha256` → **27/27 OK（CHECK_EXIT=0）**。
- 结论：merger 对源目录只读，合并前后源 checkpoint 字节完全一致
  （actor 分片、optimizer、extra_state、data.pt 均未变）。

## 4. 合并产物 manifest（`verify_p3_merged_model.py`，VERIFY_MERGED: PASS）

产物目录：`/media/imc/data/project3-search-agent-rl/models/p3-official-smoke-gs2-merged-20260816a/`
完整报告：`/media/imc/data/project3-search-agent-rl/gates/verify-merge-gs2-20260816a.json`

| 项 | 值 |
|---|---|
| 文件 | config.json / generation_config.json / model.safetensors.index.json / tokenizer 全套（tokenizer.json、tokenizer_config.json、vocab.json、merges.txt、special_tokens_map.json、added_tokens.json）/ 权重分片 model-00001-of-00002 + model-00002-of-00002.safetensors —— 全部齐全 |
| 参数量 | 3,397,103,616 = Base 实参 3,085,938,688 + 151,936×2,048（独立 lm_head.weight 副本；**与 embed_tokens.weight 逐字节相等**，tie 语义保持——verl FSDP 保存 tied 模型（tie_word_embeddings=true）的自然格式） |
| dtype | bfloat16（权重与加载后均 bf16） |
| NaN/Inf | 0 张量含 NaN、0 张量含 Inf |
| missing / unexpected keys | 均空（safetensors keys vs Transformers 加载后模型 keys） |
| 总大小 | 6,810,172,509 bytes（6.34 GiB） |
| config | Qwen2ForCausalLM，hidden 2048 / 36 层 / 16 头，tie_word_embeddings=true |
| tokenizer | vocab 151,665，eos/pad=`<|endoftext|>` |

逐文件 SHA256：

```
added_tokens.json       58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b
config.json             e99bfd96fb50a908fcf32707c561b4a66f4b07fbce8a569a6066ccbcfd08d778
generation_config.json  bd03c5abee61c7bba94010a6885933427b45575109c54c7d988299536038cff8
merges.txt              8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5
model-00001-of-00002.safetensors  84651ddfeb04da0e1a22f768d8d2311e9f31153df5d8ef8336131a5e062bbb2c
model-00002-of-00002.safetensors  56fa7332d750b13ab9baa140b3fb1a33ea924ffac63fdd3ca3a44c08e79fe438
model.safetensors.index.json      eac9151a4a19af8a4e06ddfd346bc18e61650485a2df5f548afddda961aea17c
special_tokens_map.json 6676f091c8bc4d1b50146427cfde92073402866b87b6e39223227931b70083e9
tokenizer.json          9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa
tokenizer_config.json   a5dae0102f342a3b6bc2a4d03f430ed83666dcab8879c9ee44c08a3d9095132d
vocab.json              ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
```

## 5. 工具 test 子命令适配性

**不适用，已记录原因**：verl `model_merger.py test` 的 FSDP 分支
（`_test_state_dict`）断言合并结果与 `--test_hf_dir` 参考 HF 模型**逐张量相等**
（atol/rtol=1e-6），仅适用于「未训练、权重仍等于参考模型」的初始 checkpoint。
本输入为已训练 2 步的模型，不存在合法 reference，逐张量等值断言必然失败。
替代验证已覆盖合并正确性：tie 内容逐字节一致、NaN/Inf=0、missing/unexpected=空、
参数量精确匹配、以及下述 vLLM 加载 + smoke-16 端到端链路。

## 6. GPU1 受管 smoke-16 评测（官方宽松入口）

**Run ID**：`p3-eval-vllm-official-smoke-gs2merged-20260816a`
**日志**：`/media/imc/data/project3-search-agent-rl/runs/p3-eval-vllm-official-smoke-gs2merged-20260816a/`
（stdout.log / stderr.log / results.json / episodes.jsonl / cleanup.log / metadata.env）
**启动命令**：

```sh
PROJECT3_DATA_ROOT=/media/imc/data bash scripts/start_tmux_run.sh \
  p3-eval-vllm-official-smoke-gs2merged-20260816a 1 -- \
  bash -c 'export PROJECT3_EVAL_DATA=smoke; \
  export PROJECT3_EVAL_MODEL=/media/imc/data/project3-search-agent-rl/models/p3-official-smoke-gs2-merged-20260816a; \
  bash scripts/run_p3_eval_vllm_official.sh'
```

- 受管运行（run_managed.sh 门禁：flock、GPU1 空闲检查、磁盘、原子产物）；
  wrapper 硬门禁：`CUDA_VISIBLE_DEVICES=="1"`（GPU1 唯一，未用 GPU0/5）、
  Retriever health（`127.0.0.1:18080` ready，vectors=21,015,324）、patch 0001–0004
  已应用、data SHA 对照 manifest、leakage=0、`PROJECT3_RUN_ID/RUN_DIR` 必需。
- 评测数据：`datasets/searchr1-smoke/test.parquet`（16 条）——**非 final-confirm512**，
  本门禁未查看、未评测 final512。
- tokenizer：固定 Qwen2.5-3B Base。
- 引擎：vLLM 0.8.5.post1（VLLM_USE_V1=0，V0 引擎）、bf16、
  gpu_memory_utilization 0.6、tensor_parallel 1、max_model_len 2304（与训练 rollout 同路径）。

**结果**（exit_code=0，38 秒，20:55:35→20:56:13）：

| 项 | 值 |
|---|---|
| episodes | **16/16 完成**（episodes.jsonl 16 行） |
| vLLM 加载 | 成功（引擎构建无错误，加载即评测首步） |
| 搜索协议 | 24 总步；8 error_observation → tool exception → retry（官方宽松语义：无 query → 错误观测 → 重试，无惩罚） |
| EM | 1/16 = 6.25%（2 步训练模型正常水平；**门禁不以此判定**） |
| 泄漏 | 16 eval ∩ 8 reference = 0 |
| peak GPU | allocated 13.78 GiB / reserved 13.29 GiB（24.5 GiB 卡，无 OOM） |
| 失败模式 | 无 NaN、无 OOM、无模型加载错误 |
| decoding | vllm-native-greedy，seed=0，max_steps=2 / history_length=2 / topk=3 / timeout=180 |

## 7. PASS 判定对照

| PASS 标准 | 结果 |
|---|---|
| merge exit 0 | ✓ MERGE_EXIT=0 |
| vLLM 加载 + smoke-16 exit 0 | ✓ run exit_code=0（评测脚本内 vLLM 引擎加载成功） |
| 16/16 样本完成，无 NaN/OOM/模型加载错误 | ✓ 16/16 episodes；NaN=0、peak 13.78GiB<24.5GiB |
| checkpoint 源 SHA 前后一致 | ✓ 27/27 文件 sha256sum -c 通过（合并前后字节不变） |
| 退出后 GPU/进程清理干净 | ✓ GPU1 回基线 18 MiB/0%；cleanup.log `compute_processes=none`；无残留 vLLM/Ray 进程 |

## 8. 边界声明

- 本门禁只验证合并与评测链路；smoke-16 EM 不作任何质量/效果声明。
- 未启动任何正式训练（Step 0–50 待单独批准）；三段冻结配置
  （`configs/p3_formal_segments_2026-08-16.json`，invariant SHA
  `2cc743a3…`）未修改；未重新预注册。
- 本门禁**补充了已预注册的 checkpoint 转换执行细节**（FSDP → model_merger → HF
  → 受管评测），不改变预注册判定规则与实验有效性。
- 合并产物 `p3-official-smoke-gs2-merged-20260816a` 为门禁专用，不接入正式训练
  （正式 Step 0 仍从 Qwen2.5-3B Base 重启）。
- 正式段 Step 50/100/300 checkpoint 评测时复用本门禁同一 merge 命令与评测入口；
  正式段 run 由 run_managed 生成，per-run config_fp 记录于各 run 日志。
