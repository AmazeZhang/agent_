# P3 Held-out 评测对比报告

| Run | 模型 | run_id |
|---|---|---|
| [base](/media/imc/data/project3-search-agent-rl/runs/p3-eval-heldout32-base-s0-20260814d) | p3-eval-heldout32-base-s0-20260814d | `p3-eval-heldout32-base-s0-20260814d` |
| [step2](/media/imc/data/project3-search-agent-rl/runs/p3-eval-heldout32-step2-s0-20260814e) | p3-eval-heldout32-step2-s0-20260814e | `p3-eval-heldout32-step2-s0-20260814e` |
| [step5](/media/imc/data/project3-search-agent-rl/runs/p3-eval-heldout32-step5-s0-20260814f) | p3-eval-heldout32-step5-s0-20260814f | `p3-eval-heldout32-step5-s0-20260814f` |

- 评测题数：32（同一批问题逐题配对）
- 参数一致性：一致 ✓
- 解码后端：['hf-transformers-greedy']；HF 与 vLLM 一致：是
- 解码说明：decoding_backend=hf-transformers-greedy (temperature 0). Same across all three models, but not byte-identical to the vLLM rollout path used in training; re-verify with verl/vLLM native eval before claiming clear improvement.

## 总体指标（含 Wilson 95% 区间）

| 模型 | EM | success | answer 合规率 | 无效动作率 | invalid query 率 | api error 率 |
|---|---|---|---|---|---|---|
| base | 0/32 (0.0%) [95% CI 0.0–10.7] | 0/32 (0.0%) [95% CI 0.0–10.7] | 96.9% | 8.3% | 25.0% | 0.0% |
| step2 | 1/32 (3.1%) [95% CI 0.6–15.7] | 1/32 (3.1%) [95% CI 0.6–15.7] | 96.9% | 23.1% | 42.9% | 0.0% |
| step5 | 0/32 (0.0%) [95% CI 0.0–10.7] | 0/32 (0.0%) [95% CI 0.0–10.7] | 100.0% | 18.4% | 33.3% | 0.0% |

## 分源 EM

| 来源 | base (n, EM率) | step2 (n, EM率) | step5 (n, EM率) |
|---|---|---|---|
| 2wikimultihopqa | 4条 0 (0.0%) | 4条 0 (0.0%) | 4条 0 (0.0%) |
| bamboogle | 2条 0 (0.0%) | 2条 0 (0.0%) | 2条 0 (0.0%) |
| hotpotqa | 8条 0 (0.0%) | 8条 0 (0.0%) | 8条 0 (0.0%) |
| musique | 2条 0 (0.0%) | 2条 0 (0.0%) | 2条 0 (0.0%) |
| nq | 8条 0 (0.0%) | 8条 1 (12.5%) | 8条 0 (0.0%) |
| popqa | 4条 0 (0.0%) | 4条 0 (0.0%) | 4条 0 (0.0%) |
| triviaqa | 4条 0 (0.0%) | 4条 0 (0.0%) | 4条 0 (0.0%) |

## 配对 McNemar（同一问题逐题对比，两尾精确检验）

| 对比 | 都错 | 仅前者对 | 仅后者对 | 都对 | 不一致对 | p 值 |
|---|---|---|---|---|---|---|
| base ↔ step2 | 31 | 0 | 1 | 0 | 1 | 1.0000 |
| base ↔ step5 | 32 | 0 | 0 | 0 | 0 | 1.0000 |
| step2 ↔ step5 | 31 | 1 | 0 | 0 | 1 | 1.0000 |

## 失败案例分类（EM=0 的 episodes）

| 类别 | base | step2 | step5 |
|---|---|---|---|
| answer_format（答案格式（无 <answer> 或无法提取）） | 1 | 1 | 0 |
| invalid_action（无效动作（投影失败 / 混合 / 重复标签）） | 1 | 5 | 5 |
| retrieval_failure（检索失败（invalid_query / api_error / no_results）） | 0 | 0 | 0 |
| retrieved_but_wrong（检索成功但答错） | 2 | 1 | 1 |
| no_search（未搜索（直接作答）） | 28 | 24 | 26 |

## 证据与 hash 归档

| Run | results.json SHA256 | episodes.jsonl SHA256 | 数据文件 SHA256 | 核对 |
|---|---|---|---|---|
| base | `02d59b4d3ff07d20c361917fa15b6b959151c7aa6c6f313620ad8dce3b4c2543` | `65ca088eaa34a62b077fd9c72844303cbb4138101f1945598269e0d2845b011e` | `1f8caca3255928baeac2aafb1b1c25445533426664a9d85c5519a4d6fab6d62f` | True |
| step2 | `45c2484963f255e74a05e4b239f4d8345d329f4912c66906efee8b436ecf4542` | `fc70769c1ce1adc534997e5203599d5f10f2163c94f8489f7f12540e797f9ecf` | `1f8caca3255928baeac2aafb1b1c25445533426664a9d85c5519a4d6fab6d62f` | True |
| step5 | `bdcf8b1be4cbb8bdd7c6a456a67a6fe8af3560ce01a05d37387f8719c416ff6c` | `4fc94c4bd0e7a66a03661cfc31c40fac0d1c6b0fbff93935491236e2b410f172` | `1f8caca3255928baeac2aafb1b1c25445533426664a9d85c5519a4d6fab6d62f` | True |

## 声明边界

- 本报告只对评测 run 做机械汇总；样本量（n≤32）只能作为小样本初步证据。
- 解码为 HF transformers 贪心（temperature 0），与训练期 vLLM rollout 存在 backend 差异；若出现明确提升，必须先用 verl/vLLM 原生评测复核关键结论。
- smoke-16 结果仅作管线门禁，不用于任何质量声明。