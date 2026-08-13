# P3-A veRL GRPO单步配置审计

- 日期：2026-08-13
- 状态：配置翻译与静态门禁通过，尚未启动训练
- 目标：物理GPU1上完成Qwen2.5-1.5B LoRA一次真实参数更新
- 固定上游：`verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`

## 1. 当前硬件与软件基线

2026-08-13只读核查：

```text
GPU：8 × NVIDIA GeForce RTX 4090 D，单卡24564 MiB
Driver：595.45.04
本地nvcc：CUDA 12.4.131
GPU0：GNOME Remote Desktop，354 MiB计算进程，禁止训练
GPU1：18 MiB基础占用，无计算进程，作为单步门禁卡
GPU5：仍按项目规则默认排除
```

训练环境：

```text
Python       3.10.16
PyTorch      2.6.0+cu124
vLLM         0.8.5.post1
FlashAttn    2.7.4.post1
Transformers 4.51.1
Ray          2.43.0
PEFT         0.15.2
Hydra Core   1.3.2
```

环境路径为`/media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124`。正式命令固定
`PYTHONNOUSERSITE=1`，不接受用户级Python包混入。依赖版本未在本阶段安装或修改。

## 2. 为什么不能直接执行上游run_search.sh

上游`examples/gigpo_trainer/run_search.sh`是8卡、7B、256问题、Group 5的GiGPO配置，直接运行
不满足本机首步安全边界：

1. `trainer.n_gpus_per_node=8`会把桌面GPU0纳入Ray资源池；
2. 模型为Qwen2.5-7B，首次真实Backward成本过大；
3. `trainer.total_epochs=1`不是“一次更新”，会遍历整个Dataloader；
4. 数据路径依赖`$HOME`，不绑定当前已验哈希的数据；
5. 默认启用W&B，产生本阶段不需要的外部写入；
6. Retriever上游启动器使用GPU FAISS并可占用全部可见卡；
7. 50步才保存Checkpoint，不满足单步恢复门禁。

这不是说上游代码不能使用，而是必须把上游配置翻译为当前机器可验证的小规模实验。训练入口
仍是`python -m verl.trainer.main_ppo`，算法、环境、Rollout、Reward和Actor更新均来自固定veRL
仓库。

## 3. 单步配置

版本化启动器：`scripts/run_p3_grpo_one_step.sh`。

| 模块 | 固定值 | 理由 |
|---|---|---|
| 算法 | Outcome-only GRPO | 首个基线，不提前混入GiGPO改进 |
| 模型 | 本地Qwen2.5-1.5B-Instruct | P2已验权重，降低首次Backward风险 |
| 参数更新 | LoRA rank/alpha 32/32 | 单卡24GB可控，仍是真实非零参数更新 |
| 问题/Group | 8问题，Group 2 | 使用全部8条训练Smoke，得到16条轨迹 |
| 环境步数 | 最多2步 | 最多32次动作，限制首次运行成本 |
| Token | Prompt 2048，Response 256 | 与规划Profile一致 |
| Actor | mini-batch 8，micro-batch 1 | 单卡保守显存配置 |
| Offload | Actor参数/优化器/Reference均开启 | 优先避免OOM |
| Rollout | vLLM V0、TP=1、显存比例0.60 | 使用上游基线值，只映射逻辑GPU0=物理GPU1 |
| Retriever | CPU Wiki-18 Top-3 | 不用Fixture、不占GPU |
| Retriever timeout | 180秒 | 全库Flat串行检索队列可能超过默认60秒 |
| 训练长度 | `total_training_steps=1` | 明确只做一次更新 |
| 保存 | `save_freq=1` | 最后一步强制保存Checkpoint |
| 日志 | Console、本地Rollout | 不启用W&B或外部服务 |
| Ray | 32 CPU | 防止Ray默认占用全部96逻辑CPU |

`actor_rollout_ref.rollout.n`保持1，因为该fork明确要求环境任务的GRPO分组由`env.rollout.n=2`
实现。GRPO按`uid`和`traj_uid`分组，不依赖错误的Rollout重复数。

## 4. Loss Mask语义

本仓库Search环境采用“每个环境Step生成一个Action样本”的旧agent-system路径：

1. 当前Observation（可能包含上一步检索文档）进入Prompt；
2. vLLM只生成本步Action/Answer作为Response；
3. Retriever返回内容进入下一步Prompt，不拼进当前Response；
4. PPO Actor默认仅对Response区域使用`attention_mask[:, -response_length:]`。

因此这个路径不应打开新式`actor_rollout_ref.rollout.multi_turn`工具配置；强行打开会要求另一套
SGLang Tool Config。静态代码表明Retrieved Token不在Policy Response中，但P3真实运行仍必须
保存Token级Trace，实测断言Observation Token的Loss Mask为0，不能用代码阅读替代退出门禁。

## 5. 防误启动与清理边界

启动器在进入veRL前强制检查：

- submodule必须是固定SHA；
- 模型、train/test parquet和Python必须存在；
- Retriever必须是`http://127.0.0.1:<port>/retrieve`；
- 正式运行必须存在`PROJECT3_RUN_ID/PROJECT3_RUN_DIR`，即由`run_managed.sh`创建；
- `CUDA_VISIBLE_DEVICES`必须精确等于物理GPU1；
- Retrieval可观测性Patch必须已经应用；
- `/health`必须为ready且向量数为21,015,324。

测试结果：

```text
非受管启动：exit 13
错误GPU0映射：exit 14
未应用Patch：exit 15
Hydra完整解析：通过
关键配置断言：32项通过，0 mismatch
最大轨迹：16
最大环境动作：32
```

`scripts/run_managed.sh`继续提供唯一Run ID、data盘空间门禁、GPU文件锁、独立进程组和定向
TERM/KILL清理。禁止使用`pkill python`、`ray stop --force`或`tmux kill-server`作为常规停止。

## 6. 下阶段执行顺序

训练尚未启动。P3-B严格按以下顺序执行：

1. 再次确认GPU1无计算进程、GPU0只有桌面进程；
2. 应用仓库内可审计的Retrieval状态Patch；
3. 在CPU-only tmux启动localhost Retriever并等待health ready；
4. 创建唯一训练Run ID，通过`start_tmux_run.sh`只映射物理GPU1；
5. 观察模型/vLLM/FSDP初始化和第一批Rollout，不扩大资源；
6. 完成一次Backward、Optimizer step和Checkpoint保存；
7. 定向停止Retriever，确认训练tmux状态、端口、Ray/Python和GPU进程；
8. 审计Reward→Advantage→Loss、梯度和Mask后写完成报告；
9. 在Checkpoint实际恢复一个Step之前，不进入5/20步。

预定训练命令结构：

```bash
cd /home/imc/yzy/agent/project3-search-agent-rl
export PROJECT3_DATA_ROOT=/media/imc/data
bash scripts/apply_project_patches.sh

bash scripts/start_tmux_run.sh <unique-run-id> 1 -- \
  bash scripts/run_p3_grpo_one_step.sh
```

用户观察：

```bash
tmux attach -t p3-<unique-run-id>
# 安全脱离但不停止：Ctrl-b，然后按d
tail -f /media/imc/data/project3-search-agent-rl/runs/<unique-run-id>/stdout.log
tail -f /media/imc/data/project3-search-agent-rl/runs/<unique-run-id>/stderr.log
```

P3-B开始时会给出实际Run ID和Retriever tmux会话名，不让用户猜测占位符。
