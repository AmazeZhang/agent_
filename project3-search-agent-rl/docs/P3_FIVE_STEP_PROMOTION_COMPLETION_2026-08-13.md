# P3五步工程晋级完成报告

## 结论

Run `p3-grpo-resume-step5-qwen15b-s0-20260813h` 已从Attempt G的Global Step 2准确恢复，
连续完成Global Step 3、4、5并以退出码0结束。三个Checkpoint、六个Rollout证据文件、
Optimizer/Scheduler连续状态和Actor主动退出均通过核验。此次可以判定“五步短程工程闭环通过”。

该结论仅表示恢复、真实优化、持久化、检索和生命周期链路可运行，不表示模型质量提升。
本次未执行held-out validation，且Step 4与Step 5的task reward和success均为0。独立实验审计
因此给出`WARN`（same-family provisional）。

## 运行范围

- 时间：2026-08-13 21:23:28至21:29:24（约5分56秒）；
- 设备：仅物理GPU1；GPU0保留给Linux图形界面，GPU5排除；
- 数据：Search-R1 smoke train 8题、test 16题；训练使用dataset-provided NQ/HotpotQA target；
- Retriever：CPU Wiki-18，21,015,324条向量/语料；
- 模型与算法：Qwen2.5-1.5B-Instruct、LoRA、veRL GRPO；
- 恢复源：Attempt G `global_step_2`；目标：`total_training_steps=5`；
- seed：0；单卡；每步16条trajectory；未运行validation；
- veRL基线提交：`20bd331bdbc9026a5668e11362178e10ab7400c8`。

## 恢复与训练证据

stdout明确加载Step 2的model、optimizer、extra state和data state，并从`global_step=2`继续。
Step 3、4、5均完成Rollout、Backward/Optimizer更新、指标记录和Checkpoint保存。

| Step | Action数 | grad norm | entropy | raw score mean/max/min | episode reward | success | valid action | step time | throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 21 | 0.295 | 1.207 | 0.052 / 1 / -0.1 | 0.125 | 0.125 | 0.571 | 91.161s | 97.103 token/s |
| 4 | 22 | 0.415 | 1.208 | -0.032 / 0 / -0.1 | 0 | 0 | 0.682 | 87.383s | 97.570 token/s |
| 5 | 24 | 0.368 | 1.004 | -0.046 / 0 / -0.1 | 0 | 0 | 0.542 | 100.109s | 104.307 token/s |

Raw action score分布为：Step 3 `{1: 2, 0: 10, -0.1: 9}`，Step 4
`{0: 15, -0.1: 7}`，Step 5 `{0: 13, -0.1: 11}`。GRPO的组内advantage标准化仅用于
训练，不作为外部评价分数。

## 连续状态与Checkpoint

- `latest_checkpointed_iteration.txt`为`5`；
- `global_step_3/4/5`均包含约7.26GB model state、296MB optimizer、148MB LoRA adapter、
  extra state和data state；无`.partial`文件；
- 392个LoRA张量在2→3、3→4、4→5每段均发生变化且全部有限；
- LoRA元素变化量约为36,928,795 / 36,928,375 / 36,928,267，delta L2分别为
  0.026777 / 0.021472 / 0.019240；
- Adam状态计数为6→9→12→15；421个Optimizer state有限；
- Scheduler `last_epoch`为2→3→4→5，`_step_count`为3→4→5→6。

以上证明三次优化更新连续、Checkpoint不是重复拷贝，但不能由参数变化推导质量改善。

## Loss Mask与检索证据

三个audit JSONL分别为21、22、24条，共67条Action。全部记录
`prompt_policy_loss_tokens=0`，policy token总数为3017、3031、2876；当前单轮配置使用
`response_attention_mask`，Prompt未进入policy loss。

Retriever累计10次成功请求，30个document ID均为非空、唯一的真实数字Wiki ID；
typed status分别为：Step 3 success=3/invalid_query=2，Step 4 success=2/invalid_query=4，
Step 5 success=5/invalid_query=3。Retriever日志中的10次请求均首次尝试成功。

## 退出与资源安全

- RegisterCenter、GPU Worker、TaskRunner依次`INTENDED_USER_EXIT`并被观察到`DEAD`；
- Actor/训练Worker日志中无`SYSTEM_ERROR`、`RAY_WORKER_FAILURE`、unexpected failure、
  异常SIGTERM或Segmentation fault；
- Raylet、GCS、Dashboard在`ray.shutdown()`时存在基础设施级SIGTERM，日志标记为
  `EXPECTED_TERMINATION`，这是正常关闭，不等同于训练Actor失败；
- 顶层退出码0，GPU1恢复约18MiB，无训练/Ray计算进程残留；
- Retriever使用精确tmux会话Ctrl-C停止，端口18080无监听，未使用全局`pkill`或`ray stop`；
- GPU0仅保留GNOME图形进程，实验未占用GPU0。

## 核心文件SHA256

```text
metadata.env     b002b5e82270b1c4308d50752a8321ed2e0b796fa3aa28fb9b08e41b64d868a2
Hydra config     a4e50d372964468a56d42cc661c5d34cf37ae0abd01965fa13e70852e0363498
stdout.log       c620e9e39292d737b1e3762d6632b836a10ae2c0fb45fe381fc85406e7728fd5
stderr.log       529fe8ce06ac7f01160a8912ab22ce5eb0c67cc1930c7225fc844206d46ec655
Step3 normal     8910d748491a8f7320929d295665bd7e0732cc0b3c6812fc577ed12e9a802f44
Step3 audit      00a8d0c50c682d78dd819645e9dfd9d4711e118afa10703950f47a33699c43b4
Step4 normal     485e4149c3413aac254639b41f4d21ff7f259d239defef21c93d0897f5b14c29
Step4 audit      1a9eb2166a100b678c36e91beaaac33a7f82dbfc102018e750aadb138b90a4c7
Step5 normal     f9755f840c7b6cab1100af1b0370c6494994587b77125883d24bfcc95930faec
Step5 audit      4cc4d4a07f60072d43374a9a7e6f302e1011dda47b63eb21a9318df9dc950b11
Step3 adapter    0377f58312ce3e90ceeff9b112f5ddf7b1a5e2f11f8ab6faf0e1e589771621c6
Step4 adapter    7d47d805c7ce3f66bc14d16f5f8ec918a6830f06502cc602624cc99dac94f084
Step5 adapter    85ee90752a55d0fb5330ddd82a59f406fbfaeb1ee696dc4f519f45ae6388a24c
```

## 审计结论与下一步

独立审计结果：GT provenance PASS、score normalization PASS、runtime/dead-code PASS、
evaluation type PASS；result/tracker和scope为WARN。完成本文及同步文件后，tracker过时问题已修正，
但实验范围问题仍然存在。

下一阶段不应直接以20步训练替代评价。应先建立独立held-out评测，固定推理和Retriever配置，
对Step 2与Step 5做同条件比较；随后增加多seed和未训练baseline，再决定是否扩大训练步数或数据量。
在这些证据完成前，禁止声称质量提升、收敛、泛化或“完整Search-R1复现”。
