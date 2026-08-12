# 项目一：Trace 驱动 Agent 自进化

## 当前状态（2026-08-10 核对）

M1–M5 工程里程碑与 r3 协议修正已完成。APO/GEPA × plain/diagnosis 四臂在 val8
独立重跑 3 次、多数票计分后均为 0.875，与同尺度基线 v0 持平；gate 全部拒绝，
版本保持 v0。当前诚实结论是：**自进化工程闭环完整跑通，但方法尚未产生收益**。

`scripts/run_loop.sh` 已提供已有产物上的统一编排入口，但诊断产物缺失时仍需人工生成，
且报告不会自动重建，因此尚未完成 clean-room 一命令全链复现验证。正式范围要求的
holdout 根因/步骤/token/成本/耗时统一比较也尚未完成，所以不能标记为 2.2 全量验收。
项目一完整现状、
缺失项和下一步选项见工作区根目录
`docs/PROJECT_STATUS_2026-08-10.md`；实验细节见
`reports/ablation_2026-08-08.md` 的 r3 追加章节。

## 上游源码

```text
vendor/agent-lightning/
vendor/AgentRx/
vendor/tau2-bench/
```

## 隔离环境

```bash
source .venvs/agent-lightning/bin/activate
source .venvs/agentrx/bin/activate
source .venvs/tau2/bin/activate
```

三个环境不能直接合并。当前阶段先跑 tau2 mock/text 基线，再接入 Agent Lightning Trace，最后用 AgentRx 做轨迹诊断。

真实任务运行前需要在项目级配置文件中提供模型 endpoint/key；不要修改或提交 vendor 仓库内的 `.env`。

## DeepSeek Smoke Test

密钥由工作区根目录的 `.secrets/deepseek.env` 提供，不写入命令行或上游仓库：

```bash
bash scripts/run_tau2_smoke.sh
```

已验证模型 `deepseek-v4-flash` 支持 OpenAI Chat Completions、内部推理内容及工具调用。输出预算不能设置得过小，否则内部推理可能耗尽预算而最终回答为空。

`scripts/tau2_deepseek_cli.py` 在运行时完成两项兼容配置，不修改 vendor 源码：将 NL assertion judge 指向 DeepSeek，并按官方缓存未命中价格注册 LiteLLM 成本映射。

## 可行性验证结果

- 真实 retail 任务 0、1、2 均完成，最终 Reward 均为 1.0。
- 任务 2 的已适配运行记录 agent cost `$0.002031344`、user cost `$0.0003169432`。
- `tracing/tau2_to_agentrx.py` 可将 tau2 批次结果转换为 AgentRx wrapper 输入。
- AgentRx IR 产物：`diagnosis/runs/retail-task2-ir/trajectory_ir.json`，共 25 个非空步骤。

AgentRx 默认只支持其内置 endpoint；这一限制现已由项目级运行时适配器补齐，vendor 源码保持不变。执行完整诊断：

```bash
bash scripts/run_agentrx_deepseek.sh <trajectory.json> \
  --domain tau --endpoint azure --dynamic-mode oneshot \
  --run-dir <new-output-directory>
```

有效 pilot 结果位于 `diagnosis/runs/retail-task2-full-deepseek-pilot-v2/`：六阶段均完成，checker 实际加载 1 条静态和 1 条动态 invariant，0 violations，judge 判断该 Reward 1.0 轨迹没有发生失败。

## 首批诊断基准

- `retail-pilot10` 与 `retail-hard10` 共 20 条自然 rollout，全部 Reward 1.0；成功和动作路径偏差分别记录，不把最终成功误标成失败。
- `diagnosis/runs/upstream-tau-failure-benchmark/` 使用上游 7 条真实失败轨迹和真实根因标签。
- `diagnosis/summarize_agentrx_benchmark.py` 的严格结果为类别 1/7、精确步骤 1/7、步骤 ±2 为 4/7。
- 上游未提供 few-shot 目录，因此另从有人工根因标签的 Magentic-One 轨迹构建了 5 个跨域真实示例，并确认与 7 条 tau-retail 评测任务无 ID 重叠。
- 标签版 few-shot 得到类别 2/7、精确步骤 3/7；加入根因附近原始轨迹片段后得到类别 1/7、精确步骤 2/7、步骤 ±2 为 4/7，且 token 从零样本 65,114 增至 105,588。
- 三种配置均只运行一次，未显示跨指标稳定提升。为避免对 7 条评测轨迹过拟合，已停止在该集合上调提示；下一次诊断优化必须使用独立开发集和留出评测集。

对照汇总：`diagnosis/benchmarks/upstream-tau-failure-comparison.json`。
