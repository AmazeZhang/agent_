# 项目三：Search Agent RL

本项目研究多轮搜索智能体中稀疏奖励如何经由 Advantage Estimation 和 Policy Loss 更新
模型策略。主框架使用 `verl-agent` 集成的 Search-R1 环境，算法主线为：

```text
Reward → Return → GRPO/GiGPO Advantage → PPO-style Loss → Policy Update
```

## 上游源码

```text
vendor/verl-agent/
```

该目录由根仓库作为 Git submodule 管理，固定上游
[`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent)。它已经包含Search-R1环境、
本地Retriever、GRPO、GiGPO、PPO、DAPO、GSPO和RLOO。原始
[`PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1)只作为论文、
模型和实验日志对照，不再引入第二套旧版verl源码。

## 当前基线

- 开发模型：`Qwen/Qwen2.5-1.5B-Instruct`；
- 正式模型：`Qwen/Qwen2.5-3B-Instruct`；
- 基线算法：Outcome-only GRPO；
- 改进算法：Similarity-based GiGPO；
- 候选创新：结构化Anchor State、Dynamic Sampling或Reward/Advantage可靠性加权；
- 核心观测：Reward、Return、Advantage均值/方差、有效Group比例、KL、Entropy、Clip
  Fraction、Gradient Norm、搜索次数、答案准确率和跨数据集泛化。

## 开发依据与当前阶段

项目按以下两份文件逐阶段实施：

- [`docs/DEVELOPMENT_SPEC_2026-08-11.md`](docs/DEVELOPMENT_SPEC_2026-08-11.md)：时间表、模块接口、验收门槛和服务器实验流程；
- [`configs/experiment_profiles.yaml`](configs/experiment_profiles.yaml)：Smoke、开发、主实验和完整实验的初始资源配置。

当前状态为`P3-0 / 源码与Spec冻结`。配置文件中的数值是租机和首轮实验的起点，不是已经
验证过的最佳参数；必须经过单卡Smoke和20 Step吞吐测量后才能升级配置。

服务器准备完成后先运行只读预检，不直接启动训练：

```bash
bash scripts/preflight.sh 1
# 当前8卡服务器默认使用六张稳定计算卡；GPU 0禁用，GPU 5默认排除
bash scripts/preflight.sh 1,2,3,4,6,7
# 仅在明确接受掉卡风险并有人值守时临时启用GPU 5
ALLOW_UNSTABLE_GPU5=1 bash scripts/preflight.sh 1,2,3,4,5,6,7
```

## 资源结论

- 12GB：只做推理和Reward链路，不作为RL训练配置；
- 1×24GB：最小训练Smoke，限定1.5B LoRA、短上下文、Group 2和CPU/小型Retriever；
- 2×24GB或1×48GB：推荐开发下限；
- 4×24GB：3B简历级主实验下限；
- 7×4090：完整主实验推荐配置。

完整E5 Flat Index约64.6GB，不能在单张24GB上与训练模型共存。单卡阶段必须使用CPU
BM25、CPU ANN、在线Retriever或裁剪Corpus。详细资源预算和租机门禁见
[`docs/RESOURCE_PLAN.md`](docs/RESOURCE_PLAN.md)。

## 阅读顺序

1. `agent_system/environments/env_package/search/`：环境、工具和最终答案Reward；
2. `verl/trainer/ppo/ray_trainer.py`：Reward进入Advantage Estimator；
3. `gigpo/core_gigpo.py`：Episode/Step Advantage与State Grouping；
4. `verl/trainer/ppo/core_algos.py`：Clipped Policy Loss和KL；
5. `verl/workers/actor/dp_actor.py`：Entropy、KL、Loss聚合和反向传播；
6. `examples/gigpo_trainer/run_search.sh`：官方Search GiGPO配置。

所有上游修改必须形成独立Commit或保存在`patches/`，实验结果放在`experiments/`，不得把
模型、Corpus、FAISS Index、Checkpoint或密钥提交到Git。
