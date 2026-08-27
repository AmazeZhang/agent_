# P3 检索到答案漏斗诊断（2026-08-27）

## 结论

只复算 seed2026 Clean/Aware 的既有 `episodes.jsonl`，不调用模型、Retriever 或 GPU。
结果表明：Aware-v2 已解决“是否搜索”和大部分“是否作答”问题，当前第一瓶颈是
**query/检索召回没有命中答案证据**；第二瓶颈才是命中证据后的抽取与综合。

不建议因此直接更换 Retriever 或启动长训练。下一个最小有用实验应只比较同一批既有 query
在 `top-k=3` 与一个候选检索设置下的 evidence-hit recall，先确认是截断深度/排序问题，还是
query 本身没有表达对检索意图。

## Aware-v2 题级漏斗（n=256）

| 阶段 | 答对 | 已答但错 | 未作答 | 合计 |
|---|---:|---:|---:|---:|
| 未搜索 | 7 | 15 | 0 | 22 |
| 搜索、未命中证据 | 7 | 94 | 6 | 107 |
| 搜索、命中证据 | 64 | 62 | 1 | 127 |
| 合计 | 78 | 171 | 7 | 256 |

关键转化率：

- 搜索题的题级 evidence-hit：`127/234 = 54.3%`；
- 命中证据后的正确率：`64/127 = 50.4%`；
- 未命中证据时的正确率：`7/107 = 6.5%`；
- 命中证据后的作答率：`126/127 = 99.2%`。

Aware 的178个错误题按首要可观测缺口分为：

- 搜索但未命中证据：100题（56.2%）；
- 命中证据但仍错误：63题（35.4%）；
- 未搜索：15题（8.4%）。

因此优先顺序是：

1. query quality / retrieval recall；
2. evidence-to-answer 抽取与综合；
3. 不再优先增加搜索奖励或训练轮次。

## 与 Clean 的对照

| 指标 | Clean | Aware-v2 |
|---|---:|---:|
| 搜索题 | 182 | 234 |
| 命中证据题 | 95 | 127 |
| 题级 hit rate（给定搜索） | 52.2% | 54.3% |
| hit→correct | 51.6% | 50.4% |
| 未命中→correct | 8.0% | 6.5% |
| 未作答 | 35 | 7 |

Aware 多搜索了52题，命中证据题增加32、命中且答对增加15，但 hit→correct 基本没有变化。
在逐题配对的“Clean未搜索→Aware搜索”56题中，Aware有34题命中证据、20题答对；相对 Clean
新增答对12、丢失7，净+5。说明新增搜索并非无效，但整体收益仍受未命中和证据转答案限制。

## 口径与声明边界

- evidence-hit 复用冻结的 v2 规则：只在真实 Retriever 返回的 document body 中匹配
  ground-truth alias 的连续完整 token phrase；不检查 query、错误文本或模型输出。
- 这是确定性、可复算的召回 proxy；alias 出现不等于证据语义充分。
- `未作答=7` 使用 offline answer-compliance 口径；`max_steps_exhausted=16` 使用真实环境提交
  口径，二者差9题来自有答案草稿但没有完成环境提交，不能混为一谈。

## 复算入口

- 脚本：`scripts/analyze_p3_retrieval_answer_funnel.py`
- 结果：`gates/p3_retrieval_answer_funnel_20260827.json`
- 输入 episodes SHA256：Clean
  `72c5b24ff1cd83c5f930b00cd4d222c9695b955837e6389c4874ba4d1732ad58`；Aware
  `fd055ff85c848f25200de1596526379baf0fca310927fc250518ca40576b5756`。
