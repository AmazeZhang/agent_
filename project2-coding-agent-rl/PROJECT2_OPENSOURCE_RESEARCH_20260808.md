# 项目 2 开源调研报告：阶段 1 数据/框架/配方选型 — 2026-08-08

> 结论先行：**SWE-Master（RUCAIBox）在 2026-02 开源了完整训练代码**——
> 正是训出我们手中 SWE-Master-4B-RL 的项目。其 pipeline = **rLLM（RL/GRPO）+
> OpenRLHF（SFT）+ 自研数据过滤**，配方公开可照抄。配套数据资产
> （SWE-smith 52k 任务 + 26k 轨迹）已实测下载、格式验证兼容。
> 推荐：训练框架 rLLM，SFT 数据直接复用 SWE-Master 过滤链路，GRPO 参数照抄官方 4B 脚本。

## 1. 调研背景

阶段 0 结论（门禁① FAIL）后，阶段 1 需要：100+ 任务数据池、teacher 轨迹、
SFT/GRPO 训练框架。本次调研验证三件事：
1. 有哪些可直接部署的训练框架（支持 agent 多轮 RL + 我们已有协议）？
2. 有哪些可直接下载的数据集（任务/轨迹/格式兼容）？
3. 与我们的场景最接近的公开配方是什么（可照抄的参数）？

## 2. 训练框架对比

| 框架 | 规模/许可 | 活跃度 | 说明 | 结论 |
|---|---|---|---|---|
| **rLLM**（rllm-org/rllm） | 5,770★ / Apache-2.0 | 2026-08-07 仍在更新 | 内置 mini-swe-agent harness；SFT/GRPO/REINFORCE/RLOO；tinker 单机 + verl 多卡后端；训练时 base_url 指向 gateway 自动捕获 logprobs，agent 代码 eval/train 同一份；60+ benchmark（含 SWE-bench） | **首选** |
| verl（verl-project/verl） | 22,867★ / Apache-2.0 | 2026-08-08 活跃 | rLLM 的底层后端，DAPO 等算法生态 | 作为 rLLM 后端使用 |
| THUDM/AgentRL | 337★ / MIT | 2026-01 | 多轮多任务 GRPO（Ray placement groups） | 参考，不主用 |
| NVIDIA Polar（arXiv 2605.24220） | 论文+开源 | 2026-05 | API 网关捕获轨迹做 GRPO，agent 黑盒零改动（Codex 3.8%→26.4% Verified） | 我们的架构已类似，对照验证 |
| OpenRLHF（OpenRLHF/OpenRLHF） | — | 活跃 | 仅做 SFT 阶段（SWE-Master 的选择） | SFT 用 |

**验证**：SWE-Master repo 内嵌 rllm（`DeepSWE_RL/rllm`），RL 入口
`python3 -m rllm.trainer.verl.train_agent_ppo`——官方生产验证过 rLLM 路线，可直接对齐。

## 3. 数据资产（已实测）

| 数据集 | 规模 | 许可 | 状态 | 用途 |
|---|---|---|---|---|
| SWE-bench/SWE-smith（52k 任务） | 52k 实例，11 parquet ~250MB | MIT | 样本已下载验证（train-00002: 3,696 行） | 任务池来源 |
| SWE-bench/SWE-smith-trajectories（26k 轨迹） | 8 parquet ~1.2GB | MIT | 样本已下载验证（ticks-00000: 3,229 行） | SFT 正例（resolved 标签现成） |
| SWE-Gym（SWE-Gym/SWE-Gym） | 2,438 实例 + Lite 234 | Apache-2.0 | 2025-07 后停更，数据在 HF/Docker Hub | 任务池备选 |
| nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1 | 6,436 样本 RLVR 格式 | CC-BY-4.0 | 2026-06 更新 | RL 格式参考 |
| SWE-Master 配套（repo 内） | 过滤脚本/配方 | MIT | 可用 | 复用 |

**格式验证结果**：
- 轨迹 parquet 列：`messages`（SWE-agent 风格完整对话）/ `instance_id` / `resolved`（bool）/
  `patch`（gold diff）/ `model` / `traj_id`。首条样本即 boltons 任务（lm_rewrite 生成），
  与 phase0 任务同源 → **与我们 mini-swe-agent 协议天然兼容**。
- 任务 parquet 列：`instance_id` / `patch` / `element`（测试规格）/ `image_name` /
  `repo` / `problem_statement`——SWE-bench 标准结构。

## 4. SWE-Master 官方配方（可照抄部分）

### 4.1 整体 pipeline（来自 repo 结构与训练脚本）

```
teacher rollout（每 issue N 次）→ resolve-rate 难度双峰过滤（剔除 trivial/intractable）
→ long-horizon SFT（OpenRLHF multiturn）→ RLVR+GRPO（rllm train_agent_ppo）
→ TTS（LLM 模拟验证，SWE-World）
```

### 4.2 4B-RL 官方训练参数（`examples/swe/swe_rl_*_4b.sh`，即我们 4B-RL 模型的训练配置）

| 参数 | 官方值 | 备注 |
|---|---|---|
| 算法入口 | `rllm.trainer.verl.train_agent_ppo` | rLLM + verl 后端 |
| adv_estimator | `loop`（按回合） | 多轮轨迹 |
| 底座 | Qwen3-4B-Instruct SFT 版（bon 过滤后） | 我们：Qwen2.5-Coder-7B-Instruct |
| rollout | vLLM async、temperature 1.0、n=4、TP=2 | group-relative advantage |
| lr | 1e-6 | |
| clip_ratio_high | 0.28（非对称） | |
| KL | 0（use_kl_loss=False, kl_coef=0.0） | 纯执行反馈 |
| loss 聚合 | seq-mean-token-sum | |
| max_response_length | 122,880 | **80K+ 长轨迹；需适配** |
| gpu_memory_utilization | 0.70 | 与我们 vLLM 一致 |
| 数据 | SWE-Gym+SWE-rebench 混合 verl.parquet | |

### 4.3 SFT 参数（`qwen_25_coder_32B_new_remove_01_not_dedep.sh`）

`deepspeed --module openrlhf.cli.train_sft`：`--multiturn --packing_samples
--max_len 81920 --zero_stage 3 --lr 5e-5 --bf16 --flash_attn --gradient_checkpointing
--apply_chat_template`；数据格式 `{"input": [{"role":..., "content":...}]}` 多轮 JSONL。

### 4.4 可复用的过滤脚本（`OpenRLHF_SFT/SFT_data_pre_process/`、`data_preparation/`）

- `bon_filter/0_bon_pass_rate_init.py` → `1_bon_pass_rate_select.py` → `2_get_corr_data.py`：
  先算每条轨迹 pass rate，再选择中间难度，再取对应数据
- `difficulty_score_add.py` / `difficulty_score_filter.py`：难度分数计算与过滤
- `prepare_swe_data_json_jsonl.py`（rllm/examples/swe/）：→ verl.parquet 转换

## 5. 与我们的差异与适配点

| 差异 | 官方（H800×8, 80GB/卡） | 我们（GPU 1-7, 24GB 级/卡） | 适配 |
|---|---|---|---|
| 上下文 | 128K 上限，response 122,880 | 7B+24GB 只能 ~32K 上下文 | 训练任务选短轨迹（响应 ≤24-32K） |
| 环境后端 | Docker + swebench_fork wheel | 本地 venv Phase0Env（已验证） | 数据对接 `make_test_spec_for_rl.py` 格式，执行层用 phase0 |
| reward | swebench 测试规格 | 规则评估器（门禁②全 PASS） | 直接接 `@rllm.evaluator` |
| 训练规模 | 32B/128K 大配比 | 7B 小配比 | batch/mini-batch 缩小，其余照抄 |

## 6. 落地路径

1. 全量下载数据（26k 轨迹 ~1.2GB + 52k 任务 ~250MB，clash 代理，<30 分钟）
2. 复用 SWE-Master 过滤链路 → 100-200 中间难度任务池
3. SFT：OpenRLHF multiturn（配方 4.3，max_len 缩至 32K）
4. RL：rLLM `train_agent_ppo`（配方 4.2，上下文缩至 24-32K）
5. 评测：phase0 协议零改动，三臂对比（zero-shot / SFT / SFT+RL）

## 7. 可追溯性

- 样本数据：`/media/imc/data/yzy/agent/project2/datasets/`
  （swe-smith-trajectories/ticks-00000-of-00008.parquet、swe-smith-tasks/train-00002-of-00011.parquet）
- 官方 repo：https://github.com/RUCAIBox/SWE-Master（MIT，2026-02-24 开源训练代码）
- rLLM：https://github.com/rllm-org/rllm（Apache-2.0）
- 数据集：https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories、
  https://huggingface.co/datasets/SWE-bench/SWE-smith
