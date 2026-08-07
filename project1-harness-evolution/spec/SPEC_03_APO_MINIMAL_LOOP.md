# SPEC 03 — M3：APO 最小闭环

- 版本：v0.1（草稿）
- 日期：2026-08-07
- 前置：M0（APO 示例跑通）、M1（基线）、M2（划分/指标/资源/过滤）

## 1. 闭环定义（对应调研报告 1 节目标闭环）

```text
Agent 执行(dev 集) → 发现失败 → AgentRx 诊断根因
→ 反馈适配器: 诊断 → APO 文本批评/梯度
→ APO 生成候选资源（beam search）
→ 候选过滤（M2 规则）
→ 验证集重跑
→ 收益与回归检查（M2 指标）
→ 接受/拒绝 + 资源版本更新
→（回到第一行，下一轮）
```

## 2. 关键实现：AgentRx 诊断 → APO 反馈适配器

对应调研报告 2.3 第 2 项"AgentRx 诊断到优化器反馈的结构化转换"。

### 2.1 输入（诊断输出，已有）

AgentRx judge 产物：失败类别（10 类）、失败步骤、证据（violation 日志）、（可选的）修复建议。

### 2.2 转换目标

- APO 原生用 `TraceToMessages` 把 rollout 轨迹转成消息喂给优化 LLM（文本批评）。
- 适配器 `optimizers/diagnosis_to_feedback.py` 在 trace 消息之上**注入诊断段落**，格式：

```text
[DIAGNOSIS] 类别: <category>
失败步骤: <step>
证据: <evidence 摘要，≤N tokens>
建议: <judge 建议，若有>
```

- 实现两种模式（供 M5 消融）：
  - `feedback=diagnosis`：注入诊断段落（本项目方案）
  - `feedback=plain`：仅原始轨迹（等价纯 APO，对照臂）

### 2.3 接入点

- 继承/包装官方 `TraceToMessages`（vendor 不改），在消息拼装后追加诊断段。
- APO 的 `gradient_batch_size` 从失败样本轨迹中采样（失败样本 = 高信息量，参考 RethinkSkill 结论）。

## 3. Trainer 配置（参考官方 room_selector_apo.py）

```python
algo = APO[Task](
    deepseek_async_client,
    val_batch_size=...,      # 按 val 集规模
    gradient_batch_size=..., # 失败轨迹数
    beam_width=2, branch_factor=2, beam_rounds=2,  # 首轮保守
)
trainer = Trainer(
    algorithm=algo,
    n_runners=N,             # CPU 并行 rollout，N 按机器核数
    initial_resources={"system_prompt": baseline_prompt()},
    adapter=DiagnosisAwareAdapter(),   # 2.3
)
trainer.fit(agent=tau2_agent, train_dataset=dev, val_dataset=val)
```

- `n_runners` 决定并行度：无 GPU 时 CPU 并行，先小（2–4）再调。
- 训练/评测任务**只从 dev/val 加载**；holdout id 断言在 dataset 层强制。

## 4. 回归门控与版本更新

- `evaluation/gate.py`：每轮 APO 输出历史最优候选 → 在 **val 集** 重跑 → 与基线比较（M2 2.3 无回归判据）。
- 接受：更新 `resources/versions/v{N}/` 下的资源文件 + `resources/versions/CHANGELOG.md`（版本、候选来源、val 指标、成本、决策原因）。
- 拒绝：记录拒绝理由（指标回退/成本超限/非法），保留证据。
- 门控后的资源版本 = "Harness 版本"，对应调研报告闭环最后一环"更新 Harness 版本"。

## 5. 对照臂（本轮就位，M5 出数）

| 臂 | 反馈 | 目的 |
|---|---|---|
| baseline | 无优化（M1 基线 prompt） | 零点 |
| apo-plain | 纯 APO（原始轨迹反馈） | 隔离"诊断"的贡献 |
| apo-diagnosis | APO + AgentRx 诊断反馈 | 本项目方案 |

## 6. 成本与运行控制

- 每臂 APO 预算上限：beam_rounds ≤ 3、每轮 val 重跑 ≤ 2 次/候选、per-instance cost 上限沿用 $0.05。
- 全程 tmux 运行；日志与中间产物落 `runs/loop-apo-<臂名>/`。
- 每臂结束后先 `evaluation/metrics.py` 复核，再决定是否继续轮次。

## 7. M3 验收

- [ ] 适配器两种模式输出正确（单元测试：给定诊断 JSON → 消息含/不含 DIAGNOSIS 段）
- [ ] APO 完成 ≥1 轮完整闭环（候选生成 → 过滤 → val 重跑 → 门控决策）
- [ ] 三个对照臂至少各 1 轮，数字落盘且可由 metrics.py 重算
- [ ] holdout 断言在真实运行中触发过 0 次（可 grep 日志证明）
- [ ] CHANGELOG.md 有 ≥1 条真实版本记录
