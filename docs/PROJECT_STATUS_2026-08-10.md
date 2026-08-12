# 双项目接手状态（2026-08-10）

- 初次核对：2026-08-10 20:25 CST（UTC+08:00）
- 最后更新：2026-08-12（纳入 GPU5 隔离、四卡 NCCL 与 ZeRO-3 smoke 结果）
- 仓库分支：`main`，HEAD `418c75c`，与 `origin/main` 同步
- 本文用途：作为 AI 或开发者接手时的第一入口；历史报告保留，不覆盖旧结论
- 事实来源：仓库代码与原始结果、数据盘产物、训练日志、宿主机 tmux/进程/GPU 快照

## 1. 总体结论

| 项目 | 当前阶段 | 已完成 | 当前阻塞/边界 |
|---|---|---|---|
| 项目一：Trace 驱动 Agent 自进化 | M1–M5 工程里程碑、r3 协议修正完成；正式交付仍有缺口 | 轨迹→诊断→候选→独立重跑→gate 闭环；APO/GEPA × diagnosis/plain 四臂均有结果 | 四臂 r3 均与同尺度基线持平；holdout 最终比较、成本同口径和 clean-room 一键复现未完成 |
| 项目二：Coding Agentic RL | Phase 1a 完成，Phase 1b 单卡及隔离四卡链路已验证 | 数据/G1 完成；fused CE 通过 CPU/GPU、真实单步及四卡 ZeRO-3 对照；GPU2/4/6/7 容器内 NCCL PASS | 正式 238 条 SFT 未完成；gradient checkpoint + ZeRO-3 兼容性待修；RL 和三臂评测尚未开始 |

项目二已经在不重启、不恢复 GPU5 的条件下恢复了可实验的四卡环境：容器按 UUID 仅
注入物理 GPU2/4/6/7，NCCL all-reduce 与 7B 四卡 ZeRO-3 fused/reference smoke 均通过。
当前优先级是把这条隔离路径固化为正式训练入口，并解决长序列训练所需的 gradient
checkpoint + ZeRO-3 兼容性，再启动 238 条正式 SFT；
项目一是否继续扩大实验，需要先决定是否投入新的评测预算，而不是直接追加同配置轮次。

## 2. 项目一：当前事实

### 2.1 已完成

- M1：tau2 retail 40 任务基线，成功率 0.900；固定 dev/val/holdout = 24/8/8。
- M2：tau2 轨迹转换、AgentRx 失败诊断、诊断反馈适配。
- M3/M4：APO 与 GEPA 各有 plain/diagnosis 两臂，候选生成和 val 评测真实执行。
- M5 工程产物：四臂消融、独立重跑、回归 gate、原始结果和报告落盘。
- r3 修正了三个关键协议问题：
  1. gate 改用 val8 上的同尺度基线 0.875，而不是 retail40 的 0.900；
  2. 每个任务独立重跑 3 次并按多数票计分；
  3. 候选持平也拒绝，避免 seed 复制品产生伪版本。
- `scripts/run_loop.sh` 已存在，可编排环境检查、基线/划分检查、val 基线和四臂运行。

### 2.2 r3 最终结果

| 臂 | 内部 val | 独立重跑 ×3 多数票 | gate | 版本 |
|---|---:|---:|---|---|
| baseline v0 | — | 0.875 | — | v0 |
| APO plain | 1.000 | 0.875 | reject：持平/无提升 | v0 |
| APO diagnosis | 1.000 | 0.875 | reject：持平/无提升 | v0 |
| GEPA plain | 0.875 | 0.875 | reject：持平/无提升 | v0 |
| GEPA diagnosis | 0.875 | 0.875 | reject：持平/无提升 | v0 |

APO 内部的 1.000 是单次评测噪声造成的假信号；独立三次重跑显示 task 27 稳定失败，
四臂真实水平都没有超过基线。诊断反馈有“防止 GEPA 候选身份退化”的定性价值，
但没有转化为成功率提升，不能宣称方法有效。

### 2.3 仍缺失或需要补强

1. **方法收益缺失**：当前最重要的研究结论是负结果，不应包装成提升。
2. **统计证据不足**：val 只有 8 个任务；三次重跑降低随机噪声，但不是多 seed、跨任务域统计验证。
3. **正式 holdout 交付未完成**：holdout 未用于调优是正确的，但 DEVELOPMENT_SCOPE 2.2
   同时要求在留出集统一比较根因、失败步骤、token、成本和耗时；当前没有这组最终数字。
   若因为 val 无胜出而决定不解封 holdout，应明确调整验收范围，不能同时宣称 2.2 全部完成。
4. **诊断器质量有限**：既有上游失败基准只有 7 条，严格类别准确 1–2/7，尚不足以证明诊断反馈可靠。
5. **成本 gate 不完整**：r3 候选 `candidate_cost` 仍为 `null`，当前 gate 实际主要依据成功率。
6. **一键复现未完全闭合**：`run_loop.sh` 在诊断产物缺失时要求人工先生成，也不会自动重写消融报告；因此是“已有产物上的编排入口”，尚不是经验证的 clean-room 一命令重建。
7. **历史记录需按轮次解释**：r1/r2 使用过不同 gate 尺度与语义，最终结论必须以 r3 为准，不能混合比较。

### 2.4 项目一接手入口

- 消融结论：`project1-harness-evolution/reports/ablation_2026-08-08.md`（含 r3 追加）
- 原始 r3：`project1-harness-evolution/runs/loop-*/round3.json`
- 同尺度基线：`project1-harness-evolution/runs/baseline_val_rerun.json`
- 编排入口：`project1-harness-evolution/scripts/run_loop.sh`
- gate：`project1-harness-evolution/evaluation/gate.py`
- 版本历史：`project1-harness-evolution/resources/versions/CHANGELOG.md`

## 3. 项目二：当前事实

### 3.1 已完成的旧闭环与边界

此前 3B smoke 已完成数据、SFT、GRPO、checkpoint 导出和同协议 WP7 评测的工程闭环，
但 GRPO 四次全零 reward，WP7 为 0/15 提交。该阶段证明链路能运行，也实证了
3B 能力、数据格式和奖励动作空间错配问题；不能作为训练有效性的正结果。

### 3.2 新 Phase 1a 已完成

| 资产 | 当前规模/状态 |
|---|---|
| SWE-smith task pool | train 148 + eval 10 |
| SFT 数据 | 287 条 resolved multiturn 轨迹 |
| 24K 预过滤数据 | 238 条，避免截断丢失末尾修复动作 |
| eval gold 复核 | 10/10 最终任务通过三条件复核 |
| 7B 底座 | Qwen2.5-Coder-7B-Instruct，约 15GB，已落数据盘 |
| 训练环境 | `.venvs/phase1-openrlhf` 可导入 OpenRLHF、Torch、Transformers、DeepSpeed |

数据审计的两个关键结论必须在后续保持：

- SWE-smith 的 task `patch` 是引入 bug 的 patch；环境初态必须是 commit + apply patch，修复方向是反向变换。
- trajectory 的 `patch` 列在池内约 71% 错位；SFT 只使用已抽查匹配的 `messages`，gold patch 必须来自 tasks parquet。

### 3.3 Phase 1b 本轮进展与当前阻塞

- 原普通 CE 路径：7B LoRA、ZeRO-3、24K 多卡训练至少两次在第一个 backward OOM；
  正式输出目录 `phase1/checkpoints/sft-7b/` 仍为空。
- `fused_ce.py` 已修复梯度符号和 temperature backward，冻结 lm_head 时跳过约
  2.03 GiB 的 fp32 weight gradient，并通过补丁接入 OpenRLHF 0.10.4 Actor。
- 接入是可观测、fail-closed 的：输出激活 marker、记录调用数，不支持的模式不静默退回
  普通 logits 路径；干净环境可用 installer + 版本锁定 patch 重建。
- CPU float32 与 GPU2 bf16 小张量测试均通过输出、hidden/weight gradients、temperature
  和 token mask 对照。
- 完整 Qwen2.5-Coder-7B Actor 单卡对照通过：loss 差 0、per-token log-prob 最大绝对差
  `1.907e-06`、LoRA 全局 gradient relative L2 `1.906e-02`。
- 一条真实 Phase 1 多轮轨迹在物理 GPU2 完成一步：1819 tokens、523 个 assistant loss
  tokens、loss `0.92989695`、grad norm `0.53340524`、LoRA 参数发生更新，adapter 已保存到
  `/media/imc/data/yzy/agent/project2/phase1/checkpoints/sft-single-step-smoke-20260810/`。
- 该 adapter 是 smoke 产物，不是正式训练 checkpoint，不能用于宣称 G2 完成或模型提升。

物理 GPU5 的 PCIe 链路故障仍存在。内核日志显示 2026-08-09 04:07:14
`Slot(106): Link Down / Card not present`，随后 GPU5 报 Xid 79（fallen off the bus），
驱动对全节点记录 Xid 154 `Node Reboot Required`，并解绑 GPU5、移除其 DRM device。
此后 modeset 每 5 秒持续等待 GPU5 progress，宿主机直接启动 NCCL 即使不选择 GPU5
也会在 NVML 枚举时失败。未执行 PCI reset、GPU reset、驱动重载或重启。

GPU5 **不是训练必需卡**。NVIDIA container runtime 的 UUID 设备注入已验证成功：
容器内只看见物理 GPU2/4/6/7，四 rank NCCL all-reduce 全部 PASS。随后 7B、LoRA、
ZeRO-3 的 128-token fused/reference 对照也通过：四 rank loss 均为 `13.240282`，
每 rank 捕获 392 个梯度；loss 差 0、gradient global relative L2 `1.416e-02`、cosine
`0.99992771`、max absolute diff `5.859e-03`。证据日志分别为
`runs/phase1/nccl_container_g2467.log` 与
`runs/phase1/z3_container_g2467_seqlen128_r13.log`（runs 默认不入 Git）。

该 ZeRO-3 smoke 为隔离验证而使用 client-created Torch AdamW，避免依赖 DeepSpeed
FusedAdam JIT；短序列还关闭了 gradient checkpoint。启用非重入 checkpoint 时，
PyTorch 重计算会把 ZeRO-3 已重新分片的参数视为 shape `[0]` 并触发 CheckpointError，
这是正式长序列训练前仍需修复的独立软件问题，不能把 smoke PASS 等同于 G2 完成。

详细实现、数值和安全记录见
`project2-coding-agent-rl/PROJECT2_PHASE1B_SMOKE_REPORT_20260810.md`。

### 3.4 RL 与最终评测尚未开始

- Phase 1 新路线尚无 SFT 产物，所以 GRPO 不能进入正式训练。
- zero-shot / SFT / SFT+RL × 15 任务三臂评测尚无新数字。
- G1 已满足；G2、G3、G4 均未满足。G2 的单卡和隔离四卡最小链路证据已补齐，但“正式训练完成
  且 eval pass@1 > 0”两个必要条件仍缺失。

### 3.5 运行资源快照（易变信息）

2026-08-10 21:26 的最后一次 GPU 实验状态（资源占用属于易变信息）：

- 四卡 ZeRO-3 smoke 已正常退出，GPU2/4/6/7 均释放；没有遗留训练进程。
- 宿主机 `nvidia-smi` 正常显示 GPU0–4、6、7；GPU1/3 的旧 vLLM 分别约占
  17.6/13.8 GiB，GPU2/4/6/7 基本空闲。GPU0 只有桌面系统占用，项目禁止使用。
- 只有 GPU5 无 device handle；其 PCI function 仍可被 `lspci` 看见，但没有
  `Kernel driver in use`，PCIe current link speed/width 无法读取。
- 普通命令沙箱隐藏 `/dev/nvidia*`，会让 `nvidia-smi` 误报整机驱动失联；GPU 状态必须
  由宿主机上下文核验。本节已按宿主机结果纠正。

安全记录：曾有一次将 `CUDA_VISIBLE_DEVICES=2,4` 与 DeepSpeed `--num_gpus 2` 组合，
DeepSpeed 启动日志显示它忽略过滤并映射到物理 GPU0/1；发现后立即终止，未进入模型训练，
GPU0 返回系统基线。所有 Phase 1 多卡入口现已改为显式 `--include` 并加入拒绝 GPU0 的预检。

这些服务和 GPU 状态可能随时变化；每次启动任务前仍必须重新检查，且禁止使用物理 GPU 0。

### 3.6 项目二接手入口

- 新路线规格：`project2-coding-agent-rl/SPEC_PHASE1_20260808.md`
- Phase 1a 交付：`project2-coding-agent-rl/PROJECT2_PHASE1A_REPORT_20260808.md`
- 旧 3B 闭环：`project2-coding-agent-rl/PROJECT2_STATUS_20260808.md`
- Phase 0 验证器：`project2-coding-agent-rl/PHASE0_VALIDATOR_REPORT_20260808.md`
- Phase 1 脚本：`project2-coding-agent-rl/scripts/phase1/`
- Phase 1b smoke 报告：`project2-coding-agent-rl/PROJECT2_PHASE1B_SMOKE_REPORT_20260810.md`
- 大资产根目录：`/media/imc/data/yzy/agent/project2/phase1/`

## 4. 工作区与可复现性状态

本节所列开发内容已在 2026-08-12 纳入待提交集合；vendor 子模块仍会因已应用的可复现
patch 显示 modified，这是预期状态：

- `project1-harness-evolution/resources/versions/CHANGELOG.md`：APO r3 两条拒绝记录未提交。
- `project1-harness-evolution/vendor/agent-lightning`：AgentOps message role/tool-call 兼容补丁。
- `project2-coding-agent-rl/vendor/SWE-agent`：浅 clone reset 补丁；仓库已有对应 patch 文件说明。
- `project2-coding-agent-rl/scripts/phase1/` 与 `patches/`：本轮 Phase 1b fused、测试、
  GPU/NCCL 预检和容器隔离入口。

仓库自带 `shared/scripts/cpu_smoke.sh` 于 2026-08-10 通过。新增 fused CE CPU/GPU 数值
测试、真实单卡一步 SFT、隔离四卡 NCCL 和 ZeRO-3 对照均通过；正式数据集训练仍未
启动，算法收益仍无实验结论。

## 5. 建议的下一步顺序（尚待对齐）

1. GPU5 不进入训练计划；继续使用 UUID 白名单容器，仅注入物理 GPU2/4/6/7，每次启动前逐卡检查。
2. 将 client AdamW/带编译器镜像方案整理为正式训练入口，避免 runtime-only 镜像的 FusedAdam/Triton 编译失败。
3. 修复或绕开非重入 gradient checkpoint 与 ZeRO-3 参数重分片冲突，并增加带 checkpoint 的短序列回归测试。
4. 做 24K 单步显存验证；通过后才启动 238 条正式 SFT。GPU5 可保持故障状态，不把恢复它作为实验前置条件。
5. 正式 adapter 保存/加载并完成 2–3 个 eval 快速验证后，才进入 RL。
6. 项目一在继续实验前选择目标：工程收尾、扩大统计验证，或研究方法迭代。三者成本和产出不同，不应默认同时推进。
