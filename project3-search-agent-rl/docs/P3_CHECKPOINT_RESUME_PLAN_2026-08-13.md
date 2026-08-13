# P3-C Checkpoint恢复与运行时证据计划

## 目标与边界

本阶段只从已验收的`global_step_1`恢复，并执行一个新的Actor Update到
`global_step_2`。它用于验证Checkpoint可恢复、实际训练Token Mask、结构化Retriever状态和
退出清理；不是完整Search-R1复现，也不用于宣称质量提升。

固定输入：

```text
源Checkpoint  /media/imc/data/project3-search-agent-rl/runs/p3-grpo-1step-qwen15b-s0-20260813c/checkpoints/global_step_1
模型            Qwen2.5-1.5B-Instruct
算法            GRPO + LoRA rank 32
数据            searchr1-smoke/train.parquet（8题）
轨迹            每题2条，最多2个环境步骤
训练GPU         仅物理GPU1
Retriever       localhost CPU，Wiki-18完整IndexFlatIP
```

本阶段不扩展到全量训练集。8题Smoke足以验证状态恢复和计算链路；全量数据会放大资源消耗，
但不会提高这个工程门禁的判别力。

## 新增可审计证据

1. Retriever每次工具调用保存`status`、错误、结果数和`document_ids`，不复制完整文档正文；
2. 每条训练记录保存完整padded `input_ids`、`attention_mask`和实际`policy_loss_mask`；
3. 审计器断言Prompt区Loss Mask全零，且Response Mask不能覆盖Padding；
4. JSONL先以独占`.partial`写入，`flush + fsync`后原子改名，拒绝覆盖旧证据；
5. `0001`与`0002`补丁已经在干净上游副本顺序应用、反向应用并与当前vendor逐文件比较一致；
6. Mask单元测试2项及Search联合回归14项通过，覆盖单轮Response Attention Mask、多轮显式
   Loss Mask、原子写入拒绝覆盖和端到端Retriever Document ID。

## Checkpoint恢复细节

veRL的`resume_path`会恢复Actor、Optimizer、Extra State和`data.pt`，并把Global Step设为1。
Smoke训练集只有一个Batch，而保存的DataLoader已经yield 8个样本，因此恢复配置设
`total_epochs=2`：第一个循环完成已耗尽的Epoch，第二个Epoch提供唯一Batch，执行
`global_step_2`。如果日志中没有同时出现以下证据，则恢复门禁失败：

```text
Load from checkpoint folder: .../global_step_1
Setting global step to 1
training/global_step:2
.../checkpoints/global_step_2
.../rollouts/2.audit.jsonl
```

新Checkpoint写入新的Run目录，源`global_step_1`只读使用，不覆盖、不移动、不删除。

## Ray退出规避与判定

Attempt C的栈明确位于Ray 2.43.0：

```text
TaskEventBufferImpl::FlushEvents
TaskInfoAccessor::AsyncAddTaskEventData
```

不是Forward、Backward或Checkpoint保存崩溃。Ray二进制同时包含
`RAY_task_events_report_interval_ms should be > 0 to use TaskEventBuffer`，所以本次启动前固定
`RAY_task_events_report_interval_ms=0`，只禁用Ray任务事件缓冲，不修改训练算法。CPU-only本地
Ray actor创建、kill和`ray.shutdown()`探针已正常退出。

训练完成仍必须检查stderr中无`SIGSEGV`、`Fatal Python error`和unexpected worker death，不能仅
依赖顶层exit code。

## 非破坏性门禁

- 启动前确认GPU0只有桌面进程，训练只映射物理GPU1；
- Retriever设置`CUDA_VISIBLE_DEVICES=''`；
- 使用唯一Run ID和新的Run目录，拒绝覆盖；
- 通过`run_managed.sh`记录进程组，只定向清理本Run；
- 禁止`pkill python`、`ray stop --force`、`tmux kill-server`；
- 结束后复核GPU1、Ray/Python进程、Retriever端口和两个tmux会话；
- 保存源/目标Checkpoint哈希、Optimizer/Scheduler连续性、Mask和Retriever状态摘要。

## 启动与查看

计划Run ID：`p3-grpo-resume-step2-qwen15b-s0-20260813d`。

训练启动后：

```bash
tmux attach -t p3-p3-grpo-resume-step2-qwen15b-s0-20260813d
# 退出查看但不中止：Ctrl-b，然后按d
tmux capture-pane -pt p3-p3-grpo-resume-step2-qwen15b-s0-20260813d:0 -S -100
tail -f /media/imc/data/project3-search-agent-rl/runs/p3-grpo-resume-step2-qwen15b-s0-20260813d/stdout.log
```

`Pane is dead (status 0)`只表示tmux中的受管命令已经正常返回；最终结果以Checkpoint、审计
JSONL、stderr和资源清理的联合验收为准。
