# P3-C Checkpoint恢复与运行时证据完成报告

## 1. 结论

- Run ID：`p3-grpo-resume-step2-qwen15b-s0-20260813d`
- 时间：2026-08-13 18:38:59至18:41:37（Asia/Shanghai）
- 顶层退出码：0
- 工程门禁：**WARN**

已证实：veRL从`global_step_1`恢复模型、Optimizer、Scheduler/RNG和DataLoader状态，执行了
第二个Global Actor Update，并保存`global_step_2`；运行时Loss Mask和真实Wiki-18检索状态
均已结构化落盘。

未通过项：Ray 2.43.0不再出现上次的`TaskEventBuffer` Segmentation fault，但关闭GPU Worker时
仍以SIGTERM退出，并被Ray记录为`SYSTEM_ERROR`和unexpected worker failure。因此只能说
“训练结果完成且资源已释放”，不能说“基础设施干净退出”。

本阶段仍是8题Smoke、单Seed、无Held-out Validation，不支持完整Search-R1复现或质量提升结论。

## 2. 恢复与第二次更新

日志直接记录：

```text
Load from checkpoint folder: .../global_step_1
Setting global step to 1
Loading from ... model ... optim ... extra_state
training/global_step:2.000
training/epoch:1.000
local_global_step_folder: .../global_step_2
```

关键训练指标：

```text
grad_norm                    0.283
pg_loss                      0.000（日志三位小数）
actor KL loss                0.001
entropy                      1.273
reward mean/max/min          0.125 / 1.0 / 0.0
valid_action_ratio           0.667
step time                    88.931 s
peak allocated/reserved GPU  27.205 / 27.402 GiB
throughput                   95.175 token/s
```

Checkpoint连续性：

- `latest_checkpointed_iteration.txt=2`；
- Scheduler由`last_epoch=1, _step_count=2`推进到`last_epoch=2, _step_count=3`；
- Step 1/2 Optimizer均有421个State Entry，其中392个参数状态对应LoRA；
- Adapter的392/392个张量全部发生变化；
- LoRA-B的20,643,840/20,643,840个元素均非零；
- Step 1→2最大绝对参数差为`9.092113e-06`；
- Adapter SHA256由`d84d48...2186`变为`2de259...c6d2`。

这支持“第二个Global Actor Update真实发生”，不等价于模型质量提升。

## 3. Token级Policy Loss Mask

`rollouts/2.audit.jsonl`共21条Action记录：

```text
问题组UID                 8
独立Trajectory UID       16
Env Step 0 / 1           16 / 5
Mask Source              response_attention_mask（21/21）
Policy Loss Token总数    2629
Active Response Token    2629
Prompt Policy Loss Token 0
```

逐条校验通过：

1. `input_ids`、`attention_mask`和`policy_loss_mask`宽度一致；
2. 2048宽Prompt区域的Policy Loss Mask全部为0；
3. Policy Mask不覆盖Response Padding；
4. 本Run的`multi_turn.enable=false`，`dp_actor.py`实际使用
   `attention_mask[:, -response_length:]`，与审计文件的Mask Source相同；
5. 5条包含检索Observation的后续Prompt同样满足Prompt Mask为0。

因此“检索Observation进入后续Prompt，但Prompt Token不参与Actor Policy Loss”在本Run得到运行时支持。

## 4. 结构化Retriever证据

真实CPU Retriever健康门禁：

```text
IndexFlatIP(d=768)
vectors=21,015,324
corpus_rows=21,015,324
```

21条Action中的工具状态：

```text
未执行Search Tool        16
成功检索                  3
invalid_query             2
其他Retriever失败         0
```

3次成功调用均保存3个数字Wiki Corpus ID，共9个ID；2次空查询保存
`status=invalid_query`、`api_request_error="query is None"`、`document_ids=[]`和
`retrieval_failed=true`。模型非法/空工具调用与Retriever服务故障没有混淆。

## 5. 产物与哈希

```text
Step 2 model state       7,256,716,618 bytes
Step 2 optimizer           295,770,858 bytes
Step 2 LoRA adapter        147,770,464 bytes
2.jsonl                         38,696 bytes，21行
2.audit.jsonl                  522,081 bytes，21行
```

```text
optimizer     7880692ffa31889fe2c5f09e88d189fa732f4b4acf3d645934e1c2012448e24c
extra state   7aad6ff8284b939446308459d8f353c335619886b662a7b9678bf928fd32d972
data state    70ce1753432b6e2d5a7152a8955835de8be2a883815d6c7e7c772f4508bdfb35
lora adapter  2de259fd86610d4e4cba1c40bc10147d07470e1641e573dfefd6f21e2b98c6d2
rollout       4510f7ca21fb0a1d617a657aeba17b740b8f6760cbe1efeb3ae9934d315d999d
audit rollout 0425d3903f27d2832133c8399cdd06395c7d3136b76edd32fa81d180f675c9f2
stdout        e9479a92c6dda3a32d2506272ce9cb116bb464571e96f5d93eaf776d6029cb27
stderr        82279465f530bab3984cc14ce268a9b30351078401e7c805648ef908c45df1b2
metadata      a73fae264bad609c7380cf3bba31d4ea9aa9426b9bb92e0d0eb002cea5d09f1a
```

7.26GB完整模型文件存在且时间/大小正确，本次未把其SHA作为新增门禁；Adapter、Optimizer、
Scheduler和日志的联合证据已经证明连续更新。

## 6. 退出与资源清理

改进：未再出现`SIGSEGV`、`Fatal Python error`或`AsyncAddTaskEventData`栈。

残留问题：保存完成后Ray日志仍报告：

```text
Worker exit type: SYSTEM_ERROR
The process receives a SIGTERM
```

所以退出门禁仍为WARN/FAIL，顶层`exit_code=0`不能单独代表干净退出。

资源清理通过：

- GPU1回到18MiB且无Compute Process；
- GPU0仅保留GNOME Remote Desktop（354MiB）；
- 无Raylet、GCS、训练Python残留；
- CPU Retriever收到唯一目标tmux的Ctrl-C后端口关闭、进程回收、tmux `status 0`；
- 未使用`pkill`、`ray stop --force`或`tmux kill-server`。

## 7. 独立完整性审计

fresh same-family审计结论为**WARN（provisional）**：

- Ground Truth：PASS；
- Score Normalization：PASS；
- Result Existence/Numeric Fidelity：WARN（结果成立，但退出状态矛盾）；
- Runtime Observability：WARN（核心证据成立，普通Generation Dump仍有覆盖窗口）；
- Scope：WARN；
- Evaluation Type：`real_gt_training_reward_no_heldout_evaluation`。

下一门禁是修复Ray Worker的退出语义，并让普通`N.jsonl`也采用非覆盖原子写入。之后才能考虑
5步训练；在Held-out验证和多Seed之前，仍不报告质量改进或完整复现。
