# P3 Checkpoint Resume Experiment Audit

- Date: 2026-08-13
- Auditor: fresh Codex reviewer（exact submodel unavailable）
- Review independence: same-family
- Acceptance status: provisional
- Overall verdict: **WARN**
- Integrity status: **warn**

未发现伪造Ground Truth、用模型预测统计自归一化成绩、虚构Checkpoint或虚构Optimizer Update。
本次证据支持Checkpoint恢复、第二个Global Actor Update、运行时Loss Mask和真实Wiki-18结构化
检索证据；不支持“干净退出”“完整Search-R1复现”或“质量提升”。

## A. Ground Truth Provenance：PASS

- 8条训练记录来自NQ/HotpotQA数据，Target位于数据集`reward_model/env_kwargs`字段；
- `SearchEnv`使用外部数据Target，规则Reward对最终`<answer>`做归一化Exact Match；
- Run实际使用21,015,324向量的Wiki-18 Retriever，Document ID为数字Corpus ID；
- 已知Ground-Truth-derived Fixture只用于P1测试，未进入本Run。

## B. Score Normalization：PASS

- Reward为原始规则分数，Action Score分布为2个`1.0`、12个`0.0`、7个`-0.1`；
- 21条Action的均值为0.0619047614，与日志`critic/score/mean=0.062`一致；
- Episode Reward均值0.125使用16条Trajectory作为分母，标签明确；
- GRPO组内Advantage标准化是算法步骤，不是对外质量指标自归一化。

## C. Result Existence and Numeric Fidelity：WARN

已核验：

- Hydra配置为`resume_path=.../global_step_1`、总Step 2、总Epoch 2；
- 日志明确加载Step 1模型、Optimizer和Extra State，并恢复DataLoader；
- Step 2为`grad_norm=0.283`，保存完整Checkpoint且Tracker内容为2；
- Scheduler由`last_epoch=1/_step_count=2`推进至`2/3`；
- Step 1/2 Optimizer均有421个State Entry，392个LoRA参数状态的内部Adam计数3→6；
- 392/392个Adapter张量变化，LoRA-B共20,643,840个元素全部非零；
- Step 1/2 Adapter SHA256不同，Step 2为`2de259...c6d2`；
- 两份Step 2 Rollout均存在且各有21行。

警告：`metadata.env`记录顶层exit 0，但Ray在结果保存后将GPU Worker的SIGTERM退出记录为
`SYSTEM_ERROR`和unexpected worker death，不能把exit 0解释为基础设施干净退出。

## D. Dead Code and Observability：WARN

运行时证据通过：

- 21条记录、8个Group UID、16个Trajectory UID、Env Step为16/5；
- 21/21的2048 Token Prompt Policy Mask全零；
- Policy Loss Token总数2629，等于Active Response Token总数；
- 本Run`multi_turn=false`，审计与`dp_actor`实际使用同一Response Attention Mask；
- 3次成功检索保存9个数字Document ID；
- 2次空查询保存`invalid_query`、错误文本、空ID和`retrieval_failed=true`。

剩余警告：审计JSONL采用独占、fsync和原子改名，但普通Generation Dump仍用覆盖写；同Step重跑
理论上可能先覆盖普通Dump，再被审计Dump拒绝。没有证据表明本Run发生过覆盖。

## E. Scope：WARN

- 8个训练问题、16条Trajectory、21条Action、最多2个环境步骤；
- 单Seed，仅新增一个Global Actor Update；
- `val_before_train=false`、`test_freq=-1`，Final Validation为None；
- 只支持恢复和训练闭环工程证据，不支持收敛、泛化、鲁棒性、质量提升或完整复现。

## F. Evaluation Type：PASS

分类为`real_gt_training_reward_no_heldout_evaluation`。Target来自数据集，环境为程序化交互，
没有执行Held-out评估。

## Claim Impact

| Claim | Verdict |
|---|---|
| C1：从Global Step 1恢复到Global Step 2完成 | Supported |
| C2：第二个Global Actor Update真实发生且Adapter变化 | Supported；须说明内部含3次Mini-batch Adam Step |
| C3：检索Observation/Prompt Token不参与Policy Loss | Supported for this run |
| C4：真实Wiki状态/Document ID持久化且失败可区分 | Supported |
| C5：基础设施干净退出 | Contradicted / FAIL |
| C6：Search-R1完整复现或质量提升 | Unsupported / FAIL |

## Action Items

1. 修复Ray Worker退出，要求不再出现`SYSTEM_ERROR`、unexpected worker death或强制SIGTERM；
2. 普通Generation Dump也改为独占、fsync、原子改名；
3. 保持“第二个Global Actor Update”的精确表述，不把内部3个Adam Step说成3个Global Step；
4. 完成Held-out Evaluation、多Seed和更长训练后，才讨论质量或完整复现。
