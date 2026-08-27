# P3 Aware 既有 query：top-3 + 两篇候选回放（2026-08-27）

## 结论

只回放 seed2026 Aware 已生成的 query，不运行 Policy、不生成新答案、不训练、不使用 GPU。
历史 top-3 的题级 evidence-hit 为 `127/234（54.3%）`；增加两篇候选文档后为
`140/234（59.8%）`，净增13题、`+5.56pp`。

在先前定位的100个“已经搜索、top-3未命中证据且最终答错”问题中，增加两篇候选只补回10题，
仍有90题没有命中。因此：

- top-k 截断确实造成一小部分漏召回；
- 但把 `k=3` 简单扩大到5不足以解决主要问题；
- 当前更应优化 query 表达，而不是立即替换 Retriever、增加搜索奖励或启动长训练。

本实验只衡量 evidence proxy recall，**没有让模型读取新增文档，不能推断 top-5 EM 会提升**。

## 结果

| 指标 | 历史 top-3 | top-3 + 两篇候选 | 差值 |
|---|---:|---:|---:|
| evidence-hit search calls | 147 | 169 | +22 |
| evidence-hit questions | 127 | 140 | +13 |
| 题级 hit rate（给定搜索） | 54.27% | 59.83% | +5.56pp |
| 错误且未命中的100题中补回 | — | 10 | 10% |
| 仍未命中 | 100 | 90 | −10 |

新增题级命中 qid：`38,47,61,81,107,116,121,153,165,180,181,185,207`。

## 协议与完整性

- 输入：Aware seed2026 confirm256 的既有328次成功搜索；其中历史未命中的181次调用、
  158个唯一 query 需要访问 Retriever。
- Retriever：既有共享 PID 1355816，CPU Wiki-18 E5 `IndexFlatIP`，21,015,324 vectors；
  8并发，低于服务上限64；API failures=0；本轮未创建、未停止该服务。
- 历史 top-3 锚：直接使用 `episodes.jsonl` 中保存的3篇文档正文，精确复现147次/127题命中。
- 新增候选：请求当前 top-5，排除历史 top-3 ID 后按当前排名取前两篇，避免当前临界排名漂移
  反向改写历史基线。
- episodes SHA256：
  `fd055ff85c848f25200de1596526379baf0fca310927fc250518ca40576b5756`；
  响应集合 SHA256：
  `57da7c5ecea338ea1207ae89aad917014788e4dae12e79eef45da2cc4b3c30e3`。

## 首次回放失败记录

首次方案要求当前 top-3 与8月24日历史 top-3 逐篇一致。q213 的第3名从历史 `7009180`
漂移为当前 `16864745`，使当前基线多1次/1题命中，因此门禁拒绝输出结论。失败证据保存在
`gates/p3_aware_topk3_topk5_replay_20260827.json.failure.json`。最终协议改为历史 top-3 锚定，
不隐藏或删除该失败。

## 产物

- `scripts/replay_p3_aware_queries_topk.py`
- `gates/p3_aware_topk3_topk5_replay_20260827.json`
- `gates/p3_aware_topk3_topk5_replay_20260827.json.failure.json`
