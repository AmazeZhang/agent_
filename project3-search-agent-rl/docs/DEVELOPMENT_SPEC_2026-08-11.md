# Search Agent RL开发与实验Spec

- 制定日期：2026-08-11
- 计划周期：2026-08-12至2026-09-27
- 状态：执行基线v0.1，服务器验证后滚动修订
- 主框架：`langfengQ/verl-agent`，固定submodule commit `20bd331`
- 主环境：verl-agent集成的Search-R1
- 开发模型：Qwen2.5-1.5B-Instruct
- 正式模型：Qwen2.5-3B-Instruct

## 1. 项目目标

完成一个可复现、可修改、可解释的多轮Search Agent强化学习项目，核心回答：

> 稀疏的最终答案Reward如何经过Return和Advantage Estimation进入Policy Loss，并且GRPO、
> GiGPO、KL和Entropy如何影响搜索策略、训练稳定性与泛化？

最终项目必须同时具有：

1. 可运行系统：Retriever、Environment、Rollout、Reward、Advantage、Loss和Checkpoint闭环；
2. 算法理解：能够从公式和代码解释Reward到梯度更新的每个环节；
3. 个人贡献：至少一项不是官方配置切换的算法或状态表示改进；
4. 实验证据：固定数据、基线、消融、多Seed、失败分析和成本报告；
5. 求职产物：中文技术报告、英文README、图表、Demo和可核验简历Bullet。

## 2. 非目标

- 不从零重写verl、FSDP或vLLM；
- 不以复现论文全部7B/14B结果为首轮目标；
- 不把调用在线搜索API作为唯一可复现Retriever；
- 不把官方论文数字写成个人实验结果；
- 不在未完成Smoke和监控前直接运行1000 Step；
- 不同时推进过多GRPO变体，首轮只保证GRPO与GiGPO闭环。

## 3. 完成标准

### 3.1 最小闭环完成

- Search环境能完成至少一次工具调用和最终答案提交；
- 最终答案规则Reward可由单元测试覆盖；
- 至少完成一个非零梯度更新；
- Checkpoint可以保存、重新加载并继续训练；
- Retrieved/Observation Token不参与Policy Loss；
- 日志包含Reward、Advantage、KL、Entropy、Clip Fraction和Gradient Norm。

### 3.2 简历级完成

- Base、GRPO和GiGPO使用相同模型、任务切分与评测预算；
- 至少实现一个个人改进并提供关闭该改进的严格对照；
- 完成至少一个关键超参消融和两个以上Seed；
- 报告准确率、搜索次数、Token、延迟、GPU Hours和失败类型；
- 所有表格和图能够从原始日志自动生成；
- 明确上游代码、适配代码和个人算法代码的边界。

### 3.3 理想完成

- 在NQ开发集完成训练，在至少一个多跳或跨域数据集冻结评测；
- 个人改进在主要效果、样本效率或稳定性中至少一项稳定优于GRPO/GiGPO基线；
- 完成三Seed均值、标准差或置信区间；
- 将有效算法迁移到ALFWorld或Coding Agent的小规模环境，验证跨环境可用性。

## 4. 开发排期

日期以2026年为准。服务器不可用时优先完成CPU代码、测试、数据和分析工具，不等待GPU。

| 阶段 | 日期 | 主要任务 | 阶段交付物 | 退出门槛 |
|---|---|---|---|---|
| P3-0 规格冻结 | 08-12～08-14 | 固定上游Commit、代码地图、配置矩阵和实验ID | Spec v0.1、配置清单、风险清单 | 所有入口和依赖版本可定位 |
| P3-1 环境与数据 | 08-15～08-18 | 安装verl/vLLM、准备Search数据、CPU BM25或小语料Retriever | 安装记录、数据Manifest、Inference Smoke | 一条轨迹完成搜索和规则Reward |
| P3-2 更新链路仪表化 | 08-19～08-23 | Reward、Return、Advantage、Loss Mask和梯度日志；添加单测 | Telemetry模块、单测、样例Trace | 一个Batch可追踪Reward到Loss |
| P3-3 GRPO基线 | 08-24～08-30 | 1.5B LoRA、20 Step Smoke、再扩到50～100 Step | S1曲线、Checkpoint、吞吐报告 | 无OOM/NaN且存在有效组内Reward方差 |
| P3-4 GiGPO复现 | 08-31～09-06 | Exact/Similarity GiGPO、Step Advantage和Grouping统计 | S2/S3对比、状态组审计 | GiGPO结果可重跑且Step Advantage非退化 |
| P3-5 个人改进 | 09-07～09-13 | 结构化Anchor State为主；Dynamic Sampling为备选 | 算法代码、设计说明、关闭改进的对照 | 改进可单独开关且有单元测试 |
| P3-6 正式实验 | 09-14～09-20 | 3B、100～300 Step、KL/Threshold/Weight消融和多Seed | 主结果表、消融图、失败分析 | 至少两Seed且指标可自动复算 |
| P3-7 求职交付 | 09-21～09-27 | 冻结评测、成本统计、README、报告、Demo和简历证据 | Final Report、Demo、Resume Evidence | 所有简历数字绑定日志或报告 |

第一求职可用里程碑定在2026-08-30：完成1.5B GRPO真实参数更新和一份可信基线报告。
完整简历级里程碑定在2026-09-27。若P3-3未通过，不因日期压力跳过门槛。

## 5. 预期代码结构

```text
project3-search-agent-rl/
├── vendor/verl-agent/          # 固定上游，不直接堆积个人改动
├── configs/
│   └── experiment_profiles.yaml
├── search_rl/
│   ├── reward_pipeline.py      # Reward分量、Hard Gate和审计
│   ├── advantage_audit.py      # Group/Step Advantage统计
│   ├── anchor_state.py         # 结构化Anchor State与相似度
│   ├── telemetry.py            # Reward→Loss指标记录
│   └── schemas.py              # Trace和实验数据结构
├── tests/
│   ├── test_reward_pipeline.py
│   ├── test_advantage_audit.py
│   ├── test_anchor_state.py
│   └── test_loss_mask.py
├── scripts/
│   ├── preflight.sh
│   ├── prepare_data.sh
│   ├── start_retriever.sh
│   ├── run_smoke.sh
│   ├── run_train.sh
│   └── summarize_run.py
├── experiments/
├── patches/
└── docs/
```

目录中的未创建文件代表后续阶段的明确交付物，不要求P3-0一次性生成空实现。

## 6. 功能Spec

### 6.1 Retriever与环境

- Retriever通过HTTP接口独立运行，训练进程只依赖稳定的`search(query, topk)`协议；
- Smoke默认CPU BM25或裁剪Corpus，主实验再决定CPU ANN或完整E5 Dense Index；
- 每次请求记录Query Hash、Top-k文档ID、耗时、错误和重试次数；
- 环境明确区分`search`、`answer`、格式错误、超时和最大步数终止；
- Retriever错误不能被当作模型错误Reward，必须独立标记。

### 6.2 Reward

首轮保持简单、可验证：

```text
R_outcome = ExactMatch(final_answer, ground_truth)
R_total = R_outcome
```

后续Reward实验必须分量化记录：

```text
R_total = R_outcome
        + w_format * R_format
        + w_retrieval * R_retrieval
        - w_invalid * C_invalid
        - w_cost * C_cost
```

约束：

- Outcome Reward始终是主要目标；
- Format Reward不能让格式正确但答案错误的轨迹获得接近成功的分数；
- Retrieval Reward只能使用训练期允许的信息，冻结评测不能泄漏答案；
- 每个Reward分量都保存原始值和加权值；
- 新Reward必须完成Reward Hacking案例测试。

### 6.3 Advantage

GRPO基线：

```text
A_episode(i) = normalize_within_group(R_i)
```

GiGPO：

```text
G(i,t) = r(i,t) + gamma * G(i,t+1)
A(i,t) = A_episode(i) + omega * A_step(anchor_state(i,t))
```

必须记录：

- Group Reward均值、标准差和零方差比例；
- Episode Advantage与Step Advantage各自分布；
- Anchor Group数量、大小、单元素组比例和Reward冲突率；
- Similarity Threshold改变后的合并率；
- Advantage极值、NaN、Inf和裁剪前后统计。

### 6.4 Policy Loss

主实验使用PPO-style Clipped Policy Loss：

```text
ratio = exp(current_log_prob - old_log_prob)
L_pg = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
L_total = L_pg + beta * KL(current || reference) - alpha * Entropy
```

约束：

- 只对模型生成的Reasoning、Query和Answer Token计算Loss；
- Search返回的Observation/Retrieved Token必须Mask；
- 区分PPO Approx KL和Reference KL；
- Entropy即使系数为0也必须记录；
- 保存Clip Fraction、Ratio分位数、Gradient Norm和学习率。

## 7. 个人改进Spec

### 7.1 主方案：结构化Anchor State

当前GiGPO Search示例使用原始文本相似度进行近似State Grouping。计划构造：

```text
anchor_state = {
  normalized_query,
  topk_document_ids,
  evidence_signature,
  turn_index,
  previous_action_type
}
```

候选相似度：

```text
similarity = a * query_similarity
           + b * document_jaccard
           + c * evidence_overlap
           + d * action_type_match
```

验收指标：

- 有效Step Group比例提高；
- 单元素组比例下降；
- 人工抽样的错误合并率不显著上升；
- Step Advantage方差和训练稳定性可解释；
- 最终准确率、样本效率或搜索成本至少一项优于原始Similarity GiGPO。

### 7.2 备选方案：有效Group动态采样

若结构化State不能形成足够重复组，则转向Dynamic Sampling：优先采样组内成功率处于中间
区间的任务，同时保留困难任务探索配额。切换主方案必须在实验记录中说明失败证据，不能
为了结果好看同时无控制地叠加多个改进。

## 8. 实验Spec

### 8.1 核心实验

| ID | 算法 | 目的 |
|---|---|---|
| S0 | Base/Instruct | 固定原始搜索能力 |
| S1 | Outcome-only GRPO | 主基线 |
| S2 | Exact-state GiGPO | 验证严格Step Grouping |
| S3 | Similarity GiGPO | 官方可复现改进基线 |
| S4 | Structured-state GiGPO | 个人主改进 |
| S5 | Dynamic Sampling + GiGPO | 仅在主方案受阻时作为备选 |

### 8.2 消融优先级

按优先级执行，算力不足时从后向前删除：

1. Structured State On/Off；
2. Similarity Threshold：0.85、0.90、0.95；
3. Step Advantage Weight：0、0.5、1.0、2.0；
4. KL Coefficient：0、0.001、0.01；
5. Entropy Coefficient：0与一个小正值；
6. Reward分量消融。

开发阶段固定Seed 0；正式结论至少使用Seed 0和1，关键主结果争取Seed 0/1/2。不能以更换
Seed寻找最佳值替代稳定性分析。

### 8.3 主要指标

- 任务：Exact Match、Pass@1、跨数据集性能；
- 行为：平均搜索次数、无效动作率、重复Query率、答案前有效证据覆盖；
- Advantage：有效Group率、零方差Group率、Episode/Step Advantage分布；
- 优化：Policy Loss、Reference KL、Approx KL、Entropy、Clip Fraction、Gradient Norm；
- 系统：Tokens/s、Rollout/s、Retriever P50/P95、GPU峰值显存、GPU Hours；
- 可靠性：环境错误率、Checkpoint恢复、Seed方差和失败类型。

## 9. 资源Profile与预算门禁

配置初值见`configs/experiment_profiles.yaml`。

| Profile | 用途 | 单次预算上限 | 超限处理 |
|---|---|---:|---|
| smoke_1x24 | 环境和真实Backward | 12 GPU Hours | 停止并检查显存/依赖，不续跑 |
| dev_2x24 | 1.5B 20～100 Step | 96 GPU Hours | 用20 Step吞吐重新估算 |
| main_4x24 | 3B单组主实验 | 288 GPU Hours | 72小时未完成则停止并分析瓶颈 |
| full_7x24 | Dense Retriever与正式消融 | 每轮504 GPU Hours | 先验证单轮收益，再批准下一轮 |

GPU Hours是预算门禁而非预期必然消耗。任何扩容都必须附前20 Step的实测吞吐、峰值显存和
预计总成本。

## 10. 服务器实验SOP

1. 执行`nvidia-smi`，确认物理GPU 1、2、3、4、6、7空闲；物理GPU 0禁止使用，物理GPU 5因历史掉卡默认禁用；
2. 使用`bash scripts/preflight.sh <gpu_ids>`记录GPU、CUDA、PyTorch、磁盘和可见卡；
3. 保存代码Commit、submodule Commit、环境Lock和数据Manifest；
4. Retriever与训练分别在tmux会话启动，并记录PID和日志路径；
5. 先执行Inference Smoke，再执行1 Step、5 Step和20 Step训练；
6. 检查OOM、NaN、Reward方差、Loss Mask、KL、Entropy和Checkpoint；
7. 根据20 Step实测生成资源外推报告；
8. 用户确认后再启动100～300 Step正式实验；
9. 训练结束后运行统一汇总脚本，禁止手工抄写指标；
10. 保存失败实验，不删除或只保留最佳Run。

## 11. 暂停与降级条件

出现以下任一情况立即暂停扩容：

- Reward全为0或全为1，连续多个Batch无组内方差；
- Loss或Gradient出现NaN/Inf；
- Retrieved Token未正确Mask；
- Retriever错误率超过5%或P95延迟严重拖慢Rollout；
- Checkpoint不能恢复；
- GiGPO Step Group几乎全部为单元素；
- 训练准确率提升但冻结评测显著下降；
- 单次实验预计超过预算且没有早期收益证据。

降级顺序：缩短上下文与Step → 降Group/Batch → 1.5B LoRA → CPU Retriever/小Corpus →
仅运行诊断。禁止通过删除Reference KL、关闭评测或降低验证标准来掩盖资源问题。

## 12. 状态更新规则

每完成一个阶段，在本文顶部状态和下表更新，不以“代码写完”代替“验收通过”。

| 阶段 | 当前状态 | 证据位置 |
|---|---|---|
| P3-0 规格冻结 | 进行中 | 本文、README、资源计划、固定submodule |
| P3-1 环境与数据 | 未开始 | 待补充 |
| P3-2 更新链路仪表化 | 未开始 | 待补充 |
| P3-3 GRPO基线 | 未开始 | 待补充 |
| P3-4 GiGPO复现 | 未开始 | 待补充 |
| P3-5 个人改进 | 未开始 | 待补充 |
| P3-6 正式实验 | 未开始 | 待补充 |
| P3-7 求职交付 | 未开始 | 待补充 |

计划变更必须记录日期、原因、影响和新的验收条件。未经实验验证的设想只能标为“候选”，
不能写入最终简历成果。
