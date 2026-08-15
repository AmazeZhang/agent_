# Experiment Audit Report

**Date**: 2026-08-15

**Auditor**: fresh Codex reviewer（same-family，provisional，只读）

**Project**: Project 3 / Search-R1 vLLM heldout-32 evaluation

## Overall Verdict: WARN

原因码：`real_small_sample_signal_but_not_significant_reused_devset_and_runtime_code_uncommitted`

原始episodes重算支持：vLLM下Base为3/32，Step5old和train64nqh8均为5/32；训练Adapter确实
改变输出，Base→train64有2道题从错变对、0道从对变错。这是初步正向信号，但精确配对McNemar
`p=0.5`，Wilson区间高度重叠，不能称为统计显著提升、泛化改善或完整Search-R1复现。

## Checks

### A. Ground Truth Provenance: PASS（有限定）

- heldout-32来自上游test，确定性分源抽样并排除上游train、smoke train/test问题；
- manifest记录上游train overlap=0，三个Run使用相同heldout SHA和真实21,015,324-vector Wiki Retriever；
- train64-nqh来自上游train，因此没有问题级训练泄漏；
- 该32题已被多轮调参反复查看，后续应定位为`dev32`，不能再作为最终独立确认集。

### B. Score Normalization: PASS

- EM按真实dataset target和严格评分，`reward >= 1.0`才计正确；
- 0.1 format reward没有混入EM，也没有用模型自身统计量归一化；
- GRPO训练归一化未被表述为评测成绩。

### C. Result Existence and Numeric Fidelity: PASS

| 模型 | EM | 搜索episode | invalid action |
|---|---:|---:|---:|
| Base | 3/32 | 6 | 8/38 |
| Step5old | 5/32 | 6 | 8/38 |
| train64nqh8 | 5/32 | 5 | 6/37 |

Base→train64配对为`0→1=2, 1→0=0, 1→1=3, 0→0=27`。13/32题原始动作变化，两个EM
flip均为正向。Step5old与train64恰好答对同一5题。Adapter文件存在且结果内hash匹配。

### D. Runtime Path and Reproducibility: FAIL/WARN

- HF与vLLM正式结果和最小诊断共同证明生成backend会显著改变输出；使用与训练同版本、同类
  配置的vLLM作为正式工程评测backend是合理的；
- 不能把差异单独归因于FlashAttention，也不能称独立harness为完整veRL `val_only`同一路径；
- 五个成功Run实际使用当前未提交的`scripts/run_p3_eval_vllm.py`；`22df3fe`版本仍有ragged
  batch、`LoRARequest`参数和`LLM.shutdown()`问题；
- `bdedc18`只提交文档与诊断脚本，没有提交四项运行修复，Run也未记录运行时代码SHA/diff；
- 所以“bdedc18包含全部运行修复并可直接复现”不成立。

### E. Scope Assessment: WARN

- 单seed、32题、greedy、同一个反复用于决策的开发集；
- Base 3/32 = 9.38%，Wilson 95% CI `[3.24%, 24.22%]`；
- train64 5/32 = 15.63%，Wilson 95% CI `[6.86%, 31.75%]`；
- 精确双侧McNemar `p=0.5`，不显著；
- 搜索episode从6降到5，因此不支持“搜索策略改善”。

### F. Evaluation Type: PASS/WARN

分类：`real_gt_real_wiki_retriever_reused_dev32_single_seed_greedy_fork_semantics`。

这是严格fork投影/format语义下的真实GT开发集评测，不是完整上游test、官方宽松Search-R1语义、
多seed确认实验或完整Search-R1复现。

## GPU and Exit Gate

五个Run均exit code 0，只暴露物理GPU1，cleanup均为`compute_processes=none`。stderr存在
`destroy_process_group() was not called`警告，但未观察到残留；后续应显式清理process group。

## Claim Impact

- 工程评测链路有效——**supported**；
- LoRA训练参数影响vLLM输出——**supported**；
- 当前dev32上`5/32 vs 3/32`初步正向信号——**supported with qualifier**；
- 统计显著提升、泛化改善、搜索策略改善——**unsupported**；
- 完整Search-R1复现——**unsupported**；
- `bdedc18`包含全部运行修复——**unsupported**。

## Required Actions

1. 单独审阅并提交当前vLLM脚本修复，后续结果记录脚本SHA和准确Git状态；
2. 将heldout-32定位为`dev32`；
3. 从未被调参查看的数据构建至少128–256题确认集并预注册配对比较；
4. 运行一次vLLM确认评测；
5. 另建官方宽松动作语义基线，严格fork结果不直接对照论文数字。

完整审计trace：`.aris/traces/experiment-audit/2026-08-15_run05/`。
