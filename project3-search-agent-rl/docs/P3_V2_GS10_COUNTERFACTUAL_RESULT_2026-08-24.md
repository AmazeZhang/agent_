# P3 fresh v2 gs10 反事实检索评测结果（2026-08-24）

## 结论

fresh Search-aware v2 gs10 的正确回答依赖真实检索证据，预注册机制门禁通过。
同一 merged 模型、同一 confirm256 数据、同一 greedy 解码，仅改变证据内容：

| 条件 | EM | 搜索题 | 搜索且正确 | 未搜索且正确 | 作答合规 |
|---|---:|---:|---:|---:|---:|
| real | 73/256 | 239 | 65 | 8 | 247 |
| shuffled | 10/256 | 239 | 2 | 8 | 222 |
| no-evidence | 18/256 | 239 | 10 | 8 | 219 |

- real vs shuffled：real-only 63，shuffled-only 0，McNemar 精确双侧 `p<1e-8`；
  real 的 65 个搜索且正确中 63 个翻转。
- real vs no-evidence：real-only 62，no-evidence-only 7，McNemar 精确双侧
  `p<1e-8`；real 的 65 个搜索且正确中 62 个翻转。
- 三条件均搜索 239/256；未搜索且正确均为 8，构成内部对照。
- shuffled 698/698 次固定映射服务、0 fallback；no-evidence 669/669 次中性证据；
  两个 run 均 0 API error，数据 SHA、模型、prompt、解码和步数预算一致。

这支持“fresh gs10 的大多数搜索正确答案依赖真实证据”，但不证明 v2 的模型质量
优于 clean GRPO10；模型间提升仍需多 seed 对照。

## 运行与安全验收

- shuffled：`p3-eval-v2-tenstep-gs10-confirm256-shuffled-20260824a`，exit 0。
- no-evidence：`p3-eval-v2-tenstep-gs10-confirm256-noevidence-20260824a`，exit 0。
- 两个 run 均仅使用物理 GPU1；GPU0/5未使用；结束后 GPU1回到 18 MiB。
- Retriever health 始终为 ready，vectors=21,015,324。
- 无 OOM、NaN/Inf、Xid、Traceback、worker loss 或残留评测进程；无 `.partial`。
- vLLM 退出均记录 `destroy_process_group() was not called` warning，但无进程或
  显存残留；作为工程 warning 保留，不解释为 Actor/NCCL 故障。
- 两个精确的已完成 tmux 会话在验收后移除，其他旧会话未触碰。

## 证据

- 预注册：`docs/P3_V2_GS10_COUNTERFACTUAL_PREREG_2026-08-24.md`
- 统计：`gates/p3_v2_gs10_counterfactual_stats_20260824.json`
- 复算：`scripts/p3_v2_counterfactual_stats.py` 的显式 Run ID 参数
- 原始产物：对应 run 目录的 `results.json`、`episodes.jsonl`、
  `retrieval_condition_preregistration.json`、日志和 GPU peak 文件
