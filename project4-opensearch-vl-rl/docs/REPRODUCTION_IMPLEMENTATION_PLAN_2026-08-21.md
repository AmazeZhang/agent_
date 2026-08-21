# OpenSearch-VL 7×4090 受控复现实施计划

> 日期：2026-08-21  
> 状态：规划完成，尚未启动环境安装、模型下载或 GPU 实验  
> 硬件边界：物理 GPU0 永久禁用；GPU5 仅按项目四安全规范在专项门禁通过后使用  
> 目标：完成资源受限的 `Agentic SFT → Agentic RL → 消融` 双阶段复现，并严格区分上游结果、
> 本地工程闭环和本地效果证据。

## 1. 总体判断

7 张可用 RTX 4090 D 共约 168 GiB 显存，足以完成 8B 模型的推理、LoRA/QLoRA SFT 和缩小版
fatal-aware RL，但不足以原样执行论文的 32k 全参数 SFT、8×80GB 级 RL 和 70k response 大批量
rollout。

本项目采用“算法和数据流忠实、规模受控”的复现口径：

```text
官方 checkpoint 推理闭环
→ 小规模 Agentic LoRA SFT
→ 从 SFT checkpoint 启动在线 RL
→ fatal-aware GRPO 消融
→ held-out 与多 seed 验证
```

首轮不追求论文绝对分数，优先证明阶段衔接、工具闭环、loss mask、fatal token mask、advantage
clamp、checkpoint resume 和退出清理真实有效。

## 2. 当前上游断点与计划改动

| 上游现状 | 计划改动 | 验收证据 |
|---|---|---|
| RL `MODEL_PATH` 指向原始 Qwen3-VL | 改为显式接收 SFT checkpoint，禁止静默回退 | 启动元数据、模型 SHA、加载日志 |
| 默认 `adv_estimator=rloo` | baseline 保留 RLOO；论文链路显式使用 GRPO | 冻结配置与日志 |
| `visit` 文件存在但未注册 | 统一 10 工具注册和协议名 | 工具注册单测 |
| `PythonInterpreter` 同进程执行 | 默认移除；未来仅接真沙箱 | 工具列表与安全测试 |
| RL `set -x` 后加载 `.env` | 关闭 xtrace 或在密钥加载前后显式禁用 | 日志中无 key，secret scan |
| RL 用 `SERP_API_KEY`，推理用 `SERPER_API_KEY` | 建立项目四私有配置适配层 | 配置单测 |
| URL 下载无 SSRF/大小门禁 | 增加 URL、DNS、重定向、大小、像素和超时限制 | 恶意 URL/大响应测试 |
| 动态导入 `upload.py` | 默认禁用，固定路径和 SHA 后才允许 | fail-closed 测试 |
| `trust_remote_code=true` | 默认关闭；仅审核本地快照例外 | 配置和加载日志 |
| 上游写仓库相对路径和 `/tmp/vdr_tools` | 全部改为 Run 隔离的数据盘目录 | Run 目录清单 |
| 单机脚本硬编码 8 GPU | 参数化物理卡白名单、逻辑卡数和并行度 | preflight 与 smoke |
| H100/RDMA/bond1 假设 | 单机 PCIe 配置，`NCCL_IB_DISABLE=1` | NCCL smoke |
| 无项目四受管生命周期 | 实现 preflight/tmux/run/stop/cleanup 门禁 | CPU 测试与假进程测试 |

## 3. 资源和目录规划

项目代码和小型报告继续保存在：

```text
/home/imc/yzy/agent/project4-opensearch-vl-rl/
```

所有大文件放在：

```text
/media/imc/data/yzy/agent/project4-opensearch-vl-rl/
├── envs/{infer,sft,rl}/
├── models/{base,official,sft,rl}/
├── datasets/{raw,processed,manifests}/
├── hf-cache/
├── tool-cache/
├── checkpoints/
├── runs/<run-id>/
└── secrets/
```

当前根分区只剩约 91 GiB，禁止把 Hugging Face cache、模型、数据和 checkpoint 留在根分区。
数据盘当前约有 2.2 TiB 可用。首轮实验至少保留 300 GiB 空间；进入全参尝试或较长 RL 前重新
估算。Checkpoint 只规划保留“最近 2 个 + 最佳 1 个”，但实际删除必须另行获得确认。

## 4. GPU 和服务布局

### 4.1 默认安全布局

GPU5 未通过专项门禁前：

```text
稳定候选：1,2,3,4,6,7
禁用：0,5
```

用于单卡推理和早期 smoke 时，从稳定候选中实时选择一张空闲卡，不固定假设 GPU1 永远可用。

### 4.2 GPU5 恢复门禁

按以下顺序逐级测试，任一级失败即停止：

1. NVML/PCIe/Xid/温度/空闲检查；
2. GPU5 单卡 CUDA 分配、矩阵乘和短时显存压力测试；
3. GPU4+5 和 GPU5+6 的 NCCL P2P/all-reduce smoke；
4. 包含 GPU5 的 6 卡 NCCL smoke；
5. 项目四 1-step 受管训练且有人值守；
6. 退出后 GPU5 回到基线且无新 Xid。

GPU5 使用时显式设置 `ALLOW_UNSTABLE_GPU5=1`，并把授权、温度、UUID、PCI bus、测试日志和
退出状态写入对应 Run。

### 4.3 RL 推荐 6+1 布局

GPU5 健康时：

```text
训练：物理 1,2,4,5,6,7
本地 judge/搜索摘要：物理 3
保留桌面：物理 0
```

训练进程的可见顺序固定为 `1,2,4,5,6,7`，内部逻辑卡为 `0..5`。初始并行方案：

```text
train_tp=2
train_pp=1
train_cp=1
data_parallel=3
gen_tp=2
trainer.n_gpus_per_node=6
```

三个 TP pair 为 `(1,2)`、`(4,5)`、`(6,7)`。本地 judge 优先选择 7B 级量化模型；上游默认
Qwen3-32B 摘要模型不适合单张 24 GiB 卡。若使用外部 judge API，GPU3 可作为推理/故障余量。

## 5. API 与离线边界

| 能力 | 是否必须联网 | 首轮策略 |
|---|---:|---|
| SFT | 否 | 使用官方成品专家轨迹离线训练 |
| crop/sharpen/perspective | 否 | 先启用 |
| OCR/layout parsing | 可选 | 先关闭或本地部署，再接远端服务 |
| text/image search | 在线 RL 需要 | Serper/Jina 或统一网关，强制缓存与预算 |
| query-quality reward | 完整 RL 需要 | 先小模型/低成本 judge，再正式 judge |
| 最终 GPT-4o 式评测 | 正式对比需要 | 只对晋级后的少量 held-out 调用 |

最小在线 RL 配置需要搜索、网页读取和 judge。没有 key 时可以完成 SFT、离线工具单测、fatal
mask 单测和冻结缓存 rollout，但不能声称完成真实在线 Agentic RL。

## 6. 分阶段实现计划

### P0：安全门禁与基线冻结

目标：任何训练前先具备可审计、可精确停止的运行壳。

任务：

- 实现项目四 `gpu_guard/preflight/run_managed/start_tmux_run/stop_managed`。
- GPU0 硬拒绝；GPU5 默认拒绝且只接受 `ALLOW_UNSTABLE_GPU5=1`。
- 校验 GPU UUID、空闲、磁盘、Run ID、Git/submodule commit、配置 SHA 和端口。
- 新 Run 目录拒绝覆盖；独立进程组和 Ray 临时目录；TERM→等待→精确 KILL。
- 修复 `.env` xtrace 泄密；实现日志 secret scan。
- CPU 假进程测试正常结束、异常结束、超时、子进程残留和精确停止。

验收：不使用真实模型即可证明 GPU0/已有 Run/未知占用均 fail-closed，且不触碰其他进程。

### P1：环境和供应链冻结

目标：建立互不污染的 infer、SFT、RL 环境。

任务：

- 在数据盘建立三个 Python 环境，不使用系统 Python site-packages。
- 环境安装默认清空 Clash 7890 代理变量，从可直连的 PyPI/GitHub 获取；安装日志记录实际 index、
  direct URL 和 hash。不得因 tmux server 继承旧环境而静默走 7890。
- 确认 Python、CUDA 12.4、PyTorch、FlashAttention、DeepSpeed、Ray、SGLang、veRL、
  Megatron-LM、Transformer Engine 的兼容组合。
- 保存 lock/freeze、安装来源、wheel/hash、驱动和硬件快照。
- 仅运行 import/CPU 测试；随后在 tmux 受管 Run 中做单卡 CUDA smoke。

验收：三个环境可独立导入，GPU smoke 退出后无计算进程残留。

### P2：数据、模型与证据清单

目标：下载固定资产且不污染根分区。

任务：

- 固定 Qwen3-VL-8B base、官方 OpenSearch-VL-8B、SFT/RL 数据 revision。
- 权重和数据优先从可直连的 ModelScope 或 `hf-mirror.com` 获取，不使用 Clash 7890 批量下载；
  7891 属于同一 Clash 进程，未经再次确认也不用于大文件。
- 下载到数据盘，记录文件清单、大小、SHA256/revision 和许可证。
- 统计 SFT/RL 条数、字段、图片可读率、重复项和缺失项。
- 建立 10/50/100/500 条递增 smoke 子集，不修改原始数据。

验收：资产可离线重新定位；没有模型或数据写入 Git/根分区。

### P3：安全推理闭环

目标：先区分模型问题和工具环境问题。

顺序：

1. 官方 8B checkpoint，仅本地视觉工具，1 条样本；
2. 10 条样本；
3. 开启缓存后的 text search；
4. 开启 image search/layout parsing；
5. 50 条小评测。

任务：

- 默认移除 PythonInterpreter 和动态 upload.py。
- 实现 URL/下载/图片门禁与不可信 observation 标记。
- 记录工具成功率、超时率、平均轮数、fatal 比例、缓存命中和 API 成本。

验收：轨迹可审计、工具失败类型可区分、相同缓存输入可重复。

### P4：Agentic LoRA SFT

目标：获得能进入 RL 的本地 cold-start checkpoint。

首轮配置：

```text
model=Qwen3-VL-8B-Instruct
method=LoRA；必要时 QLoRA
physical_gpus=稳定卡起步，GPU5 通过门禁后再单独扩容
cutoff_len=4096，稳定后单独尝试 8192
per_device_batch=1
gradient_checkpointing=true
bf16=true
vision_tower=首轮冻结
projector=首轮冻结，后续作为独立变量
samples=100 → 500 → 更大子集
epochs=1 → 3
```

验证：

- observation token 不进入生成 loss；
- think/tool_call/response token mask 正确；
- SFT 前后固定 prompt 的格式合法率、工具选择和最终回答对比；
- 1 step → resume → 5 step → 20 step；
- checkpoint 可加载，optimizer/scheduler 连续。

全参数 ZeRO-3 不是首轮目标。只有 LoRA 链路完成后，才单独评估 CPU optimizer offload、2k～4k
上下文和 100～500 条数据的全参工程 smoke，且不能称作论文配置复现。

### P5：SFT→RL 接通与 rollout-only

目标：修复上游阶段断点，不更新权重先验证在线环境。

任务：

- RL 必须显式加载 P4 的 SFT checkpoint 和 SHA；禁止回退到 base。
- 统一工具名并注册 `visit`；保持 PythonInterpreter 禁用。
- 先 `rollout-only`，验证 group、reward、fatal 元数据、response mask 和缓存。
- 冻结一批 rollout 作为后续算法对照输入。

验收：日志明确显示 SFT checkpoint；无训练更新时也能完整生成和评分轨迹。

### P6：缩小版 Agentic RL

首轮配置建议：

```text
physical_training_gpus=1,2,4,5,6,7（仅在 GPU5 门禁通过后）
train_tp=2, pp=1, cp=1, dp=3
gen_tp=2
prompt_batch=6，稳定后 12
n_responses=2，稳定后 4
mini_batch=6
max_prompt_length=2048
max_response_length=2048，稳定后 4096/8192
n_parallel_tasks=6～12
n_parallel_tools=16～32
steps=1 → resume → 5 → 20
```

顺序先用 RLOO 做工程 baseline，再显式切 GRPO。每级检查 actor/reference/rollout 显存、KL、reward、
fatal ratio、有效 token 数、参数变化、checkpoint 和进程清理。不得直接尝试 70k response。

### P7：消融与效果验证

只有 P6 通过后进行：

| 实验 | 初始化 | 算法 |
|---|---|---|
| A | SFT checkpoint | 不做 RL |
| B | 同一 SFT checkpoint | vanilla GRPO |
| C | 同一 SFT checkpoint | GRPO + fatal mask |
| D | 同一 SFT checkpoint | GRPO + fatal mask + one-sided clamp |

控制变量：相同数据、seed、工具缓存、API 配置、prompt、训练步数、batch、checkpoint 和评测协议。
先做单 seed 诊断，确认指标非退化后再做至少 3 seeds。

指标至少包括：

- held-out 最终准确率；
- 格式合法率；
- 平均工具轮数和有效工具轮数；
- 工具/API 失败率；
- fatal 比例和 fatal step 分布；
- 被 mask token 比例；
- 正/零/负 advantage 分布及 clamp 数量；
- 缓存命中、延迟、API 调用和成本；
- OOM/Xid/NCCL/Ray 故障和清理完整性。

### P8：结果审计与交付

- 区分作者论文结果、本地复现结果和个人安全/工程改造结果。
- 数字只从原始 Run 和评测文件生成，保留配置 SHA、commit、seed 和数据 manifest。
- 至少包含失败 Run 和未支持的 claim。
- 在有同条件 baseline、held-out 和多 seed 证据前，不声称 fatal-aware 方法带来质量提升。

## 7. 近期执行顺序

下一轮建议只进入 P0，不下载模型、不安装大环境：

1. 实现项目四受管运行脚本；
2. 为 GPU0 拒绝、GPU5 授权、已有 Run、未知 GPU 进程、精确停止写测试；
3. 增加网络工具安全配置和 PythonInterpreter 默认禁用补丁；
4. 完成 CPU 测试和脚本审查；
5. 再向用户报告环境安装清单、预计下载量和首个 GPU smoke 命令，获得确认后进入 P1。

## 8. 完成定义

“双阶段复现完成”至少要求：

- 本地 SFT checkpoint 真实生成并通过固定推理对照；
- RL 明确从该 SFT checkpoint 初始化；
- 至少完成 1 step、resume、5/20 step，参数和 optimizer state 连续；
- fatal mask 与 clamp 有 token/advantage 级证据；
- 退出后无残留进程、端口和 GPU 显存；
- 有同条件 held-out baseline。

“消融完成”还要求 A/B/C/D 使用冻结条件运行，并至少报告重复实验的不确定性。达成工程闭环但没有
显著提升时，应结论为“资源受限复现链路成立，效果证据不足”，不能包装成论文结果复现成功。
