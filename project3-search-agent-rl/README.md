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

- [`docs/SEARCH_R1_REPRODUCTION_PLAN_2026-08-12.md`](docs/SEARCH_R1_REPRODUCTION_PLAN_2026-08-12.md)：复现口径、逐级门槛与秋招交付主线；
- [`docs/P1_SEARCH_PIPELINE_AUDIT_2026-08-12.md`](docs/P1_SEARCH_PIPELINE_AUDIT_2026-08-12.md)：数据、Retriever、Reward语义、泄漏与故障归因审计；
- [`docs/PROGRESS_SYNC_2026-08-12.md`](docs/PROGRESS_SYNC_2026-08-12.md)：当前完成项、哈希、复现命令和下一阶段同步入口；
- [`docs/DEVELOPMENT_SPEC_2026-08-11.md`](docs/DEVELOPMENT_SPEC_2026-08-11.md)：时间表、模块接口、验收门槛和服务器实验流程；
- [`configs/experiment_profiles.yaml`](configs/experiment_profiles.yaml)：Smoke、开发、主实验和完整实验的初始资源配置。

当前状态为`P1完成 / P2待执行`：隔离环境、版本锁、安全门禁以及数据→Fixture Retriever→
SearchEnv→严格EM的模型无关CPU闭环已通过；尚未下载模型，尚未完成模型驱动的Search-R1
功能复现，也未启动训练。配置文件中的数值只是首轮实验起点，必须经过单卡Smoke和20 Step
吞吐测量后才能升级配置。

服务器准备完成后先运行只读预检，不直接启动训练：

```bash
bash scripts/preflight.sh 1
# 当前8卡服务器默认使用六张稳定计算卡；GPU 0禁用，GPU 5默认排除
bash scripts/preflight.sh 1,2,3,4,6,7
# 仅在明确接受掉卡风险并有人值守时临时启用GPU 5
ALLOW_UNSTABLE_GPU5=1 bash scripts/preflight.sh 1,2,3,4,5,6,7
```

预检和所有GPU实验都必须将`PROJECT3_DATA_ROOT`指向已挂载的大容量数据盘。GPU任务通过
`scripts/run_managed.sh`启动、通过`scripts/stop_managed.sh`定向停止；禁止直接执行上游
8卡脚本。完整生命周期规则见[`docs/EXPERIMENT_SAFETY.md`](docs/EXPERIMENT_SAFETY.md)。
当前服务器使用`export PROJECT3_DATA_ROOT=/media/imc/data`。
固定的软件与硬件基线见[`docs/ENVIRONMENT_BASELINE_2026-08-12.md`](docs/ENVIRONMENT_BASELINE_2026-08-12.md)。

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
