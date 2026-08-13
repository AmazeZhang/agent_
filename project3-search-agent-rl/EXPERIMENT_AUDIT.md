# P3 One-Step Experiment Audit

- Date: 2026-08-13
- Auditor: fresh Codex reviewer（exact submodel unavailable）
- Review independence: same-family
- Acceptance status: provisional
- Overall verdict: **WARN**
- Integrity status: **warn**

未发现伪造Ground Truth、用模型自身最大值归一化成绩或虚构Optimizer Step。本次证据支持
“真实veRL GRPO单步参数更新完成”，但只是一轮Smoke，不支持完整复现或质量提升结论。

## A. Ground Truth Provenance：PASS

- Ground Truth来自Search-R1数据记录，经Search环境传入规则Reward，不由模型输出生成；
- Reward使用最终`<answer>`与数据集target的严格归一化Exact Match；
- 实际Run调用真实Wiki-18服务，未引用Ground-Truth Fixture；
- Retriever日志记录3次成功Top-3请求。

证据：`search_r1_like_qa_em.py:66-86,96-128`、`env.py:29-56`、
`scripts/run_p3_grpo_one_step.sh:143-155`和Run的`stderr.log:54-62`。

## B. Score Normalization：PASS

- 原始Terminal Reward为0/1，非法动作另加-0.1；
- GRPO组内均值/标准差是算法的Advantage计算，不是对外质量指标归一化；
- Run保存原始Reward统计：均值0.125、最大1、最小0；含惩罚的Score范围为-0.1到1；
- 未发现用模型自身最大值作为指标分母。

## C. Result Existence and Numeric Fidelity：WARN

已核验：

- `metadata.env`为`exit_code=0`，总训练步数为1；
- Actor Update耗时25.397秒，`grad_norm=0.300`、`pg_loss=-0.001`、`global_step=1`；
- LoRA-B共20,643,840个元素全部非零；
- Scheduler为`last_epoch=1`、`_step_count=2`、最后LR为`3e-6`；
- 模型、Optimizer、Extra State、Dataloader State和LoRA Adapter均存在；
- 21条Action记录对应8个问题、Group 2的16条轨迹，不是21条独立轨迹。

警告：Checkpoint保存和Step Metrics完成后，Ray Worker收到SIGTERM，并在关闭阶段发生
Segmentation fault。顶层exit 0表示训练主流程返回成功，不代表基础设施完全干净。崩溃发生在
结果落盘后，不否定已经保存的更新，但Checkpoint仍须实际恢复验证。

## D. Dead Code and Observability：WARN

Retrieval Patch在内存中生成`retrieval/retrieval_failed`信息，但当前Rollout Dump只持久化
`input/output/score/step`，本次`reward_extra_infos_dict`为空。因此：

- Retriever成功有服务日志证据；
- 没有结构化的逐轨迹Retriever状态证据；
- 静态代码表明检索Observation进入下一步Prompt、Actor Loss只作用于Response；
- 本次没有保存Token ID、边界和Loss Mask，不能宣称Token Mask已实测通过。

## E. Scope：WARN

- 8个训练问题、16条轨迹、21条Action、最多2个环境Step；
- 单Seed、单次Optimizer Update；
- `test_freq=-1`且最终Validation为None；
- 只支持训练闭环Smoke，不支持收敛、泛化、鲁棒性或质量提升结论。

## F. Evaluation Type：PASS

分类为`real_gt`训练Reward，但没有执行Held-out Evaluation。环境是程序化交互，目标答案来自
数据集而非模型或人工本轮生成。

## Claim Impact

| Claim | Verdict |
|---|---|
| C1：真实veRL GRPO单步Optimizer Update完成 | Supported，必须注明退出段Worker崩溃 |
| C2：Search-R1训练复现完整完成 | Unsupported |
| C3：Checkpoint可恢复 | 尚未验证，必须实际Resume |
| C4：Retrieved Observation Token不参与Loss | 静态支持，Run证据不足 |
| C5：本次训练带来质量提升 | Unsupported |
| C6：基础设施干净退出 | Contradicted；GPU清理成功但Worker关闭崩溃 |

## Action Items

1. 在下一Run保存Token级Prompt/Response/Loss Mask审计Trace；
2. 把Retrieval状态写入结构化Rollout Evidence；
3. 使用Checkpoint实际恢复一个Step；
4. 定位Ray/vLLM Worker关闭时的SIGTERM Segfault；
5. 上述三项通过前不进入5/20 Step，也不报告质量改进。
