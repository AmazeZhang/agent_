# P3五步受控晋级计划（2026-08-13）

## 1. 目标和边界

本阶段从Attempt G的`global_step_2`恢复，在相同veRL、模型、数据、Retriever和物理GPU1上运行到
`global_step_5`，新增三次真实GRPO更新。目标是验证短程连续训练、逐步Checkpoint、证据文件和
Actor生命周期在多步场景仍稳定。

本阶段不扩大到全量数据、不启用多卡、不执行正式质量评测。即使五步全部成功，也只能表述为
“五步工程晋级通过”，不能表述为Search-R1完整复现、收敛或质量提升。

## 2. 固定输入

```text
veRL commit       20bd331bdbc9026a5668e11362178e10ab7400c8
source checkpoint /media/imc/data/project3-search-agent-rl/runs/
                  p3-grpo-shutdown-gate-qwen15b-s0-20260813g/checkpoints/global_step_2
model             Qwen2.5-1.5B-Instruct
adaptation        LoRA rank 32 / alpha 32 / all-linear
dataset           searchr1-smoke: train 8 / val 16
physical GPU      1 only
retriever         CPU Wiki-18 IndexFlatIP, 21,015,324 vectors
seed              0
target step       5
total epochs      5
```

数据集每个epoch只有一个训练batch。从Step 2恢复后，epoch 3、4、5分别提供Step 3、4、5，不会
重跑Step 1/2，也不需要把全部开源数据加入当前工程门禁。

## 3. 保持不变的训练参数

- train batch 8，PPO mini batch 8，micro batch/GPU 1；
- rollout group size 2，environment max steps 2；
- prompt/response上限2048/256；
- GRPO advantage normalization，KL loss系数0.001；
- learning rate `3e-6`，parameter/optimizer/reference offload；
- vLLM V0、同步rollout、eager模式、GPU memory utilization 0.6；
- `save_freq=1`、`test_freq=-1`、无held-out validation。

## 4. 非破坏性资源策略

1. `run_managed.sh`只暴露物理GPU1；GPU0继续禁止，GPU5不使用；
2. Retriever使用`CUDA_VISIBLE_DEVICES=''`，只监听`127.0.0.1:18080`；
3. 新Run ID和新目录，拒绝复用或覆盖Attempt G；
4. 每步普通Rollout和audit均独占partial写入、fsync、原子rename；
5. 退出顺序固定为RegisterCenter→GPU Worker→TaskRunner→Driver/Ray；
6. 不使用全局`pkill`、`ray stop`、`tmux kill-server`；
7. Retriever只通过本轮精确tmux pane的Ctrl-C停止；
8. 数据盘预留门禁150GiB，预计新增三个Checkpoint约23GiB。

## 5. 启动与查看

计划Run ID：`p3-grpo-resume-step5-qwen15b-s0-20260813h`。

训练必须通过tmux：

```bash
tmux attach -t p3-p3-grpo-resume-step5-qwen15b-s0-20260813h
```

分离但不中断：按`Ctrl-b`，再按`d`。

无需attach也可查看：

```bash
tmux capture-pane -pt p3-p3-grpo-resume-step5-qwen15b-s0-20260813h:0 -S -100
tail -f /media/imc/data/project3-search-agent-rl/runs/p3-grpo-resume-step5-qwen15b-s0-20260813h/stdout.log
```

## 6. 晋级通过条件

### 配置和恢复

- resolved config为`resume_path`，来源精确指向Attempt G Step 2；
- `total_training_steps=5`、`total_epochs=5`；
- 日志实际加载模型、Optimizer、Extra State和Data State。

### Step 3–5

- 每步均出现有限的global step、grad norm、loss、reward和throughput；
- 无NaN、Inf、OOM或Traceback；
- 每步普通/audit JSONL各自存在且无partial；
- Prompt policy-loss token保持0；
- Retriever成功状态包含真实numeric Wiki ID，失败状态保持类型化。

### Checkpoint和退出

- `global_step_3`、`global_step_4`、`global_step_5`结构完整；
- `latest_checkpointed_iteration.txt=5`；
- LoRA/Optimizer/Scheduler从Step 2继续变化；
- RegisterCenter、GPU Worker和TaskRunner均`INTENDED_USER_EXIT`并进入DEAD；
- Actor/训练Worker无SYSTEM_ERROR、RAY_WORKER_FAILURE、unexpected failure、SIGTERM或段错误；
- Ray基础设施的EXPECTED_TERMINATION单独记录，不误判为训练Worker失败；
- GPU1、训练/Ray PID、Retriever PID和18080端口全部释放。

## 7. 时间与磁盘估计

Attempt G单步耗时87.623秒，初始化与恢复约40–60秒，每步保存约11秒。新增三步预计训练主体约
263秒，总墙钟约5–7分钟。三个完整Checkpoint预计约23GiB，写入3TiB空闲的数据盘，不占用仅
约95GiB空闲的系统盘。

## 8. 完成后的审计

运行结束后重新执行独立`experiment-audit`。审计必须继续把“五步工程稳定性”与“完整复现/质量
提升”分开；任何退出、证据或范围问题均写入完成报告后再推送远端。

## 9. 启动前配置审计

新增`scripts/run_p3_grpo_resume_step5.sh`和验证器`resume-step5`模式。实际解析Hydra配置后得到：

```json
{
  "status": "pass",
  "checks": 33,
  "mismatches": {},
  "derived": {
    "questions_per_step": 8,
    "trajectories_per_question": 2,
    "maximum_trajectories": 16,
    "maximum_environment_actions": 32
  }
}
```

同时通过8项训练证据/生命周期单元测试、0003补丁逆向检查、Shell语法和Git whitespace检查。
配置级门禁已通过，物理资源和Retriever健康门禁仍须在tmux启动前实时执行。
