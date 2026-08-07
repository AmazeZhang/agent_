# 双项目小规模 Pilot 报告

日期：2026-08-06

## 结论

两个项目均已通过下一层工程验证：项目一完成 DeepSeek 驱动的 AgentRx 六阶段诊断，项目二在 5 个受控 coding 任务上取得 5/5 独立验证成功。当前适合扩大到更真实的轨迹采集，不适合立刻宣称训练收益，也不建议跳过数据阶段直接启动 SFT/GRPO。

## 项目一：AgentRx 完整诊断

新增项目级运行时适配器：

- `project1-harness-evolution/scripts/agentrx_deepseek_cli.py`
- `project1-harness-evolution/scripts/run_agentrx_deepseek.sh`

适配器不修改 vendor 源码，将 AgentRx 的静态 invariant、one-shot 动态 invariant、checker 和 judge 接到 DeepSeek OpenAI 兼容接口，并修复模型 JSON schema shape drift。

有效复跑结果：

- 六阶段 `ir → static → dynamic → check → judge → report` 全部完成。
- IR：1 条轨迹、25 个步骤。
- checker：加载 1 条静态和 1 条动态 invariant。
- 动态规则成功核验 assistant 声称的 10 个可用 T-shirt 选项与工具结果一致。
- violations：0。
- judge：`INCONCLUSIVE`，原因是该轨迹没有发生失败，与 tau2 Reward 1.0 一致。
- 动态生成：14,244 输入 token、729 输出 token。
- judge：10,717 输入 token、119 输出 token。

结果目录：`project1-harness-evolution/diagnosis/runs/retail-task2-full-deepseek-pilot-v2/`。

### 自然轨迹与真实失败诊断基准

- retail 自然任务共运行 20 条（基础 10 + 较复杂 10），结果均为 Reward 1.0、DB match 通过；这批数据适合做成功轨迹，但不能提供失败诊断监督。
- 随后改用 AgentRx 上游自带的 7 条真实失败轨迹及 `tau_retail.json` 根因标签，不合成或伪造失败标签。
- DeepSeek 零样本 judge 的严格类别准确率为 1/7（14.3%），精确步骤为 1/7，步骤 ±1 为 3/7，±2 为 4/7；步骤接近但类别错误不计为正确诊断。
- 上游安装中缺少 `few_shot_examples` 目录；随后从有人工根因标签的 Magentic-One 轨迹构建 5 个跨域真实示例，且与评测任务无 ID 重叠。
- 标签版 few-shot 得到类别 2/7、精确步骤 3/7、步骤 ±2 为 3/7，共 73,240 tokens；加入真实轨迹片段的 v2 得到类别 1/7、精确步骤 2/7、步骤 ±2 为 4/7，共 105,588 tokens。
- 三种配置均为每题单次 judge，小样本波动明显，没有跨指标稳定提升。为避免对同一组 7 条测试轨迹过拟合，当前停止继续调提示，后续需先建立独立开发集和留出评测集。

严格汇总：`project1-harness-evolution/diagnosis/benchmarks/upstream-tau-failure-benchmark/summary.json`。

三配置对照：`project1-harness-evolution/diagnosis/benchmarks/upstream-tau-failure-comparison.json`。

## 项目二：SWE-agent 五任务 Pilot

任务覆盖五种基础缺陷：算术运算符、字符串规范化、聚合公式、分页偏移和有序去重。所有原始仓库均能稳定复现测试失败。

| 任务 | API 调用 | 输入 token | 输出 token | 记录成本（USD） | 独立复验 |
|---|---:|---:|---:|---:|---|
| calculator | 12 | 14,245 | 389 | 0.0008293 | 3/3 pass |
| slug | 13 | 16,076 | 457 | 0.0009657 | 3/3 pass |
| inventory | 13 | 15,426 | 397 | 0.0007056 | 3/3 pass |
| pagination | 14 | 25,299 | 708 | 0.0009749 | 3/3 pass |
| dedupe | 16 | 28,960 | 676 | 0.0010627 | 3/3 pass |
| **合计** | **68** | **100,006** | **2,627** | **0.0045382** | **15/15 pass** |

所有补丁均满足：

1. SWE-agent 状态为 `submitted`；
2. `git apply --check` 通过；
3. 在原始仓库的只读挂载副本中重新应用；
4. 在 `--network none`、CPU-only 容器中执行完整 `unittest`；
5. 测试退出码为 0。

保留的独立评测容器：

- `agent-p2-patch-eval-20260806`
- `agent-p2-pilot-eval-slug-20260806`
- `agent-p2-pilot-eval-inventory-20260806`
- `agent-p2-pilot-eval-pagination-20260806`
- `agent-p2-pilot-eval-dedupe-20260806`

轨迹中共发生 8 次 `FunctionCallingFormatError` 自动重试，主要是模型生成自然语言总结而没有直接调用 `submit`，以及一次同时发出多个工具调用。它们没有导致任务失败，但属于后续轨迹清洗和 SFT 行为格式优化的明确对象。

可使用 `project2-coding-agent-rl/scripts/summarize_pilot.py` 从落盘轨迹重新汇总 token 与成本。

### 轨迹清洗导出

新增 `scripts/export_verified_rollouts.py`，要求轨迹已提交、补丁与 submission 一致、未修改测试、历史完整，并可选实时核验离线 Docker 容器状态。当前 5 条全部通过，0 条拒绝，导出为：

- `project2-coding-agent-rl/datasets/verified-pilot5/verified_rollouts.jsonl`
- `project2-coding-agent-rl/datasets/verified-pilot5/manifest.json`

### 首条真实 SWE-bench 探针

在 `pydicom__pydicom-1458` 指定提交上完成了真实仓库探针：

- 修复了 SWE-agent 对浅克隆无条件执行完整 `git fetch` 的行为，改为目标提交不存在时才做定向浅 fetch。
- 20 次调用上限会在模型动手前终止；提高为 40 后，模型生成了非空补丁，41 次调用，记录成本 `$0.005179804`。
- 两个 Float/Double Float 直接复现通过，但候选遗漏官方测试要求的 `BitsStored` 和 `PlanarConfiguration` 条件。
- 候选补丁在离线 CPU 容器的官方测试派生检查中失败（退出码 1）；相同检查使用 gold patch 为 4/4 通过（退出码 0）。因此该真实 rollout 的 reward 诚实记为 0，不进入成功 SFT 数据。
- 真实任务代码快照、gold/test patch 和 NumPy wheel 放在 `/media/imc/data/yzy/agent/project2/real-probe/`。

这条失败证明真实仓库采集、补丁导出和严格 reward 判定链路可行，也证明受控任务 5/5 不能外推为真实任务成功率。

### SWE-smith 扩量启动

- 已从官方 SWE-smith train split 固定 20 条候选，覆盖 10 个轻量 Python 仓库，每库 2 条；完整元数据放在数据盘。
- 两条 OAuthlib 任务的基线均严格复现为 4/4 FAIL_TO_PASS 失败。
- 第一条使用 25 次调用、319,616 输入 token、669 输出 token，成本 `$0.004479104`；第二条使用 41 次调用、427,444 输入 token、797 输出 token，成本 `$0.0052416896`，触发调用上限后自动提交。
- 两个补丁均只修改 1 个非测试文件，并分别在独立评测 checkout 上通过完整测试：673 passed、2 skipped、退出码 0；两条 reward 均为 1。
- agent 分支由上游移除了 FAIL_TO_PASS 测试，评测 checkout 恢复完整测试，因此模型没有接触隐藏测试。
- 当前 2/2 来自同一仓库，只证明真实 SWE-smith 采集与 reward 链路可行，不能作为跨仓库成功率估计。
- 第三条扩到 Pygments：模型经 16 次调用正确修复 options 转发，但遗漏隐藏测试覆盖的 builtin 映射和 `is_in` 损坏；完整测试为 5114 passed、2 failed、16 skipped，reward 0。由于问题描述未覆盖 bug patch 中这些额外损坏，该条另标记任务描述不完整，而不是只归因于模型失败。
- 当前累计 3 条、2 个仓库：reward 1 为 2 条，reward 0 为 1 条。样本仍小，不报告稳定成功率。
- 完整性回查发现，第 4–6 次本地上传保留父提交；第 5、6 条模型直接读取了 `Bug Patch` 与原始实现，第 4 条虽未读取但答案可访问。三条全部标记 invalid 并从指标排除。
- 当前严格可信结果退回 3 条、2 个仓库：reward 1 为 2 条，reward 0 为 1 条。旧轨迹保留作完整性证据，不作为训练或成功率数据。
- 本地实例入口现强制检查 `HEAD^` 不可访问；受影响任务已生成单提交净化快照，后续必须重跑。
- Funcy curry/compose 净化重跑无法读取父历史；模型修复 curry 但遗漏空 compose/rcompose identity，完整测试为 201 passed、2 failed，reward 0。
- 当前严格可信结果为 4 条、3 个仓库：reward 1 为 2 条，reward 0 为 2 条；另有 3 条泄漏运行永久排除。

## SFT/GRPO 进入条件判断

当前结论：**训练工程方向可行，但数据与存储门槛尚未满足，不应立即启动正式 SFT/GRPO。**

原因：

- 5 个任务都是受控小仓库、局部缺陷，无法代表真实跨仓库泛化。
- 当前只有成功轨迹，缺少多 rollout、失败轨迹、组内 reward 方差和困难度分层，无法形成有意义的 GRPO 批次。
- 手册规划的首版 SFT 数据是 500–2,000 条验证成功轨迹，目前只有 5 条。
- `rllm[verl]` 完整训练栈和本地可训练基座模型尚未安装；API 模型本身不能作为本地权重训练对象。
- 工作区所在根分区只余约 137GB（92% 已用）。主机另有 `/media/imc/data`，约 3.4TB 可用；用户现已授权将模型、数据、checkpoint 和大缓存放到该数据盘，代码仍保留在当前工作区。

## 推荐下一门槛

1. 先将真实任务配置固定为 40 次调用并加入任务特定离线评测，再扩到 20 个 SWE-smith 或小型真实仓库任务，每题采集 2–4 条 rollout。
2. 同时保留成功与失败轨迹，记录 reward、测试差异、调用成本和格式错误。
3. 建立 SFT 清洗器：只保留可独立复验成功、未改测试、补丁最小且工具协议合规的轨迹。
4. 在已授权的 `/media/imc/data/yzy/agent/` 命名空间安装大体积训练依赖、下载 1.5B/3B 基座并做小型 LoRA SFT smoke。
5. 训练阶段继续执行：启动前检查 GPU 1–7 状态、禁止 GPU 0、只在 tmux 中运行。
