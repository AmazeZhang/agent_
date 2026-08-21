# Local OpenSearch-VL 面试项目范围

> 日期：2026-08-22
> 定位：复现 OpenSearch-VL 的关键方法与工程闭环，不追求原论文集群规模或榜单数值等价。

## 1. 项目目标

实现一个可本地复现、可训练、可演示的多轮视觉检索 Agent：

```text
图片 + 知识型问题
  -> 模型识别信息缺口
  -> crop / OCR / image_search / text_lookup
  -> 根据 observation 继续多轮决策
  -> 输出有本地证据支持的最终答案
  -> Agentic SFT 冷启动
  -> GRPO 与 fatal-aware GRPO 优化
```

项目成功的判断依据是完整流程、可执行轨迹、固定评测和 RL 增益，不是复制原作者的硬件规模或
第三方搜索服务。

## 2. 保留的原方法核心

1. **Wikipedia 多跳数据思想**：问题从实体关系路径生成，源实体图片负责视觉锚定，答案位于后续
   实体或事实节点；避免“查询图片就是答案实体”的单步捷径。
2. **统一工具环境**：SFT、RL rollout、评测和演示共用相同工具名称、参数、observation 与错误语义。
3. **Agentic SFT**：监督完整的 reasoning/tool-call/observation/final 多轮轨迹，而非只监督答案。
4. **环境内 RL rollout**：策略自行决定调用什么工具、查询内容、是否继续以及何时结束。
5. **组合奖励**：保留最终答案 `r_acc`、查询/证据质量 `r_query` 和乘法格式门 `r_fmt`；过程奖励
   不得压过最终任务奖励。
6. **Fatal-aware GRPO**：连续工具失败达到阈值后截断策略梯度；fatal 轨迹对 advantage 做单侧
   clamp，保留失败前有效推理的正向信号。

## 3. 明确替换的部分

| 原项目 | 本项目 |
|---|---|
| 开放 Web image/text search | 固定 revision 的 WIT/Wikimedia 图像与 Wikipedia 文本索引 |
| Serper/Jina/外部视觉搜索 | 本地 `image_search` 与 `text_lookup`，无 API key |
| 原始 RL-8K 全量开放环境 | 可执行原始子集 + 有 provenance 的 `offline-agentic-rl` 派生集 |
| 36K 全参数 LLM/ViT/projector SFT | 先 LoRA/QLoRA 缩放复现；资源验证后再讨论更大设置 |
| 大集群异步 Megatron 训练 | 当前主机可承载的有界 GRPO；必要时保持 rLLM/verl 接口但缩小并行度 |
| GPT-4o 榜单 judge | gold answer、实体证据、规则指标和人工轨迹案例 |

原始 `Search-VL-RL-8K` 始终保持不变。任何筛选、实体 ID、evidence ID、检索结果或合成任务均进入
独立派生数据及 run 日志，不能回写原数据。

## 4. 本地数据与任务设计

首个候选库不下载 308 GB WIT 全量，只使用一个约 0.93 GB shard 验证 schema、特征空间与索引。
必要时扩展到 2–4 shards，形成约数万张图像的演示级候选库。

任务至少覆盖：

- 图片实体识别后查证事实；
- 裁剪局部后检索；
- 图像候选冲突，需要第二轮检索或文本查证；
- Wikipedia 两跳或多跳事实问题；
- no-match、错误 crop、工具暂时失败和恢复；
- 低质量图像的 OCR/增强工具选择。

训练、验证和测试按实体划分，避免相同实体或派生图跨 split 泄漏。模型 prompt 不得包含
`entity_id`、`evidence_ids` 或 gold trajectory；这些字段只用于环境执行和 reward。

## 5. 固定评测与 RL 有效性

必须在相同 held-out 集合比较：

| 阶段 | 作用 |
|---|---|
| Base | 未训练基线 |
| SFT | 工具协议与多轮冷启动 |
| SFT + vanilla GRPO | 验证环境 RL 的基础增益 |
| SFT + fatal masking | fatal 机制第一项消融 |
| SFT + fatal masking + clamp | 完整方法 |

核心指标：最终答案准确率、格式成功率、多轮任务完成率、无效/重复调用率、平均工具步数、
no-match 幻觉率、fatal 比例，以及按任务类型分层的 reward。不能只用训练 reward 声称效果。

首轮只做 1–5 条 rollout 和不超过 20 个 optimizer steps 的工程 smoke。扩大数据、步数或多卡配置
前必须先审查固定评测、reward hacking 风险与资源预算。

## 6. GPU 与进程安全

- 数据下载、数据审计、索引 schema 检查和 CPU 单测不得占用 GPU。
- 物理 GPU0 永久排除；GPU5 默认排除，只有项目安全文件允许且显式设置恢复开关时才使用。
- 所有 GPU Run 必须使用项目 `start_tmux_run.sh` / `run_managed.sh`，具备命名 tmux、独立进程组、
  PID/token 日志和精确 stop；禁止 `pkill`、`killall` 或按模糊命令名终止进程。
- 启动前检查目标卡已有 compute process、显存和磁盘；Run 结束后检查目标卡没有遗留进程。
- 大模型、数据、索引、checkpoint 和日志写入项目四数据盘；不占用根盘，不使用 Clash 大流量端口。

## 7. 实施门禁

1. 完成 RL 图片资产校验、安全解压和 7,992 条引用审计；
2. 完成 200 条分层清单；
3. 下载一个 WIT shard 并固定真实 schema；
4. 完成本地 `image_search`/`text_lookup` 与多跳数据 pilot；**已完成**；
5. 运行 Base 固定评测；**评测协议 v4 已校准，待最终 Base 复评**；
6. 运行真实数据小规模 SFT 并复评；**旧协议 1→5 step 工程门禁已完成，不作为效果证据**；
7. 运行 vanilla/fatal-aware GRPO smoke 与消融；
8. 只有观察到可信增益且审计无数据泄漏后，才准备扩大训练。
