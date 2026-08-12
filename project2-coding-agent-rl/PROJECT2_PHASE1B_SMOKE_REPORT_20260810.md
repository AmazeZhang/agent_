# 项目二 Phase 1b：fused CE 与真实一步 SFT 验证（2026-08-10）

## 结论

Phase 1b 已从“未接入的显存优化原型”推进到**可重装、可观测、通过数值验证，且在
一条真实多轮轨迹上完成一次 LoRA optimizer step 并保存 adapter**。这证明了
数据 → assistant-token mask → 7B fused loss → backward → 参数更新 → 保存的单卡链路。

这不是正式 SFT 完成：没有可用于正式评测的训练 checkpoint。物理 GPU5 仍有 PCIe
link-down/Xid 79，但已通过 NVIDIA container runtime 按 UUID 隔离它；物理
GPU2/4/6/7 的四卡 NCCL 与 ZeRO-3 fused/reference smoke 均已通过，无需恢复 GPU5
或重启节点即可继续实验。

## 本轮实现

- `scripts/phase1/fused_ce.py`：分块计算 lm_head + token log-prob，避免完整
  `(sequence, vocabulary)` logits/dlogits；冻结 lm_head 时不分配约 2.03 GiB 的
  fp32 weight gradient，并复用分块 logits buffer。
- 修复原型中的两个正确性问题：log-prob 梯度符号反向、temperature backward 重复缩放。
- OpenRLHF Actor 分支使用 `OPENRLHF_FUSED_CE=1` 显式开启，输出
  `[OPENRLHF_FUSED_CE_ACTIVE]` 并累计调用次数；不支持的组合 fail closed。
- `patches/openrlhf-0.10.4-fused-ce.patch` 与
  `scripts/phase1/install_openrlhf_fused_ce.sh` 可在干净 OpenRLHF 0.10.4 环境重装，
  已验证 patch dry-run、实际应用和重复执行。
- LoRA targets 固定为 Qwen 的 q/k/v/o/gate/up/down projections，明确排除 frozen
  `lm_head`。
- 正式入口只接受 DeepSpeed `--include localhost:2,4,6,7`，并在模型加载前运行
  `nccl_preflight.py` 做物理映射检查和真实 NCCL all-reduce。

实现思路参考了 [Liger Kernel 官方 fused linear cross entropy 实现](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)，
但本项目保留自有实现及测试，以适配 OpenRLHF 的 per-token log-prob/multiturn mask。

## 验证证据

所有 GPU 任务均在 tmux 中运行，并在启动前重新查询 GPU 状态；单卡测试只用物理
GPU2，多卡测试只用物理 GPU2/4/6/7，物理 GPU0 从未进入容器可见列表。

| 层级 | 结果 | 证据 |
|---|---|---|
| CPU float32 数值测试 | PASS：输出、hidden/weight gradients、temperature=1/0.7、token mask | `scripts/phase1/test_fused_ce_cpu.py` |
| GPU bf16 数值测试 | PASS，同上 | `runs/phase1/fused_cuda_g2_bf16.log` |
| 完整 7B Actor 对照 | PASS：loss 差 0；token log-prob 最大绝对差 `1.907e-06`；LoRA 全局 gradient relative L2 `1.906e-02` | `runs/phase1/fused_actor_g2_seqlen32_r2.log` |
| 真实轨迹一步 SFT | PASS：1819 tokens、523 assistant loss tokens、8 个 assistant ranges；loss `0.92989695`；grad norm `0.53340524`；LoRA 参数最大更新 `4.9999e-05`；fused 调用 1 次 | `runs/phase1/sft_single_step_g2.log` |
| adapter 保存 | PASS：161,533,192 bytes；392 tensors / 40,370,176 elements 均非零 | `/media/imc/data/yzy/agent/project2/phase1/checkpoints/sft-single-step-smoke-20260810/` |
| 容器设备隔离 | PASS：容器逻辑 0–3 精确映射物理 GPU2/4/6/7；GPU0/GPU5 不可见 | `runs/phase1/container_gpu_isolation.log` |
| 四卡 NCCL | PASS：4 rank all-reduce，sum=10 | `runs/phase1/nccl_container_g2467.log` |
| 四卡 ZeRO-3 对照 | PASS：loss 差 0；每 rank 392 gradients；global relative L2 `1.416e-02`；cosine `0.99992771`；max abs `5.859e-03` | `runs/phase1/z3_container_g2467_seqlen128_r13.log` |

短序列 7B 对照中 fused 峰值显存为 17.69 GiB、普通路径 14.65 GiB；这是保存约
1.01 GiB gathered lm_head clone 和短序列分块开销的结果。因此不能用该测试宣称 fused
在短序列更省显存；优化目标是 24K 长序列完整 logits/dlogits，正式 ZeRO-3 显存收益
仍需多卡恢复后实测。

## GPU/NCCL 阻塞与安全记录

- 物理 GPU5（PCI `0000:9C:00.0`）在 2026-08-09 04:07:14 先报 PCIe hotplug
  `Link Down / Card not present`，随后 Xid 79 `GPU has fallen off the bus`；驱动记录
  Xid 154 `Node Reboot Required`，解绑 GPU5 并移除其 DRM device。PCI function 目前仍
  可见，但没有 kernel driver binding，link speed/width 无法读取。
- NCCL 初始化报 `nvmlDeviceGetHandleByIndex(5) failed: Unknown Error`。NVIDIA 官方文档
  也说明 NCCL 会依赖 NVML，并建议调试变量只用于诊断；本项目不会用屏蔽错误的方式
  把故障训练包装成有效结果。参考 [NCCL 环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。
- 曾有一次把 `CUDA_VISIBLE_DEVICES=2,4` 与 DeepSpeed `--num_gpus 2` 组合的启动；
  DeepSpeed 0.19 明确忽略该过滤并映射到物理 GPU0/1。日志出现映射后立即终止，未进入
  模型训练，GPU0 回到约 387 MiB 基线。现已永久改为显式 `--include`，测试/探针文档
  也加入禁止组合说明。
- 未执行 GPU reset、驱动重载、重启或停止他人服务；这些操作需要管理员协调和用户明确授权。
- GPU5 不是训练必需卡。NVIDIA container runtime 只注入物理 GPU2/4/6/7 的方案已
  实测成功，入口会核对四个 UUID 并 fail closed；GPU5 可保持故障状态。

## ZeRO-3 smoke 的边界

- `cuda:base` 镜像缺少 nvcc/C++，无法 JIT DeepSpeed FusedAdam；最终复用了本机已有、
  带 gcc/g++/nvcc 的 CUDA 12.6 镜像，并在 smoke 中用 client-created Torch AdamW。
- 开启非重入 gradient checkpoint 时，ZeRO-3 backward 重计算看到重新分片后的 shape
  `[0]` 参数并触发 PyTorch CheckpointError；128-token 正确性 smoke 因此显式关闭
  checkpoint。正式 24K 训练前必须解决该兼容性，不能直接照搬短测配置。
- ZeRO-3 后 `p.grad` 会被分片优化器消费，测试改用 autograd hooks 捕获 392 个 LoRA
  梯度，并以 global relative L2、cosine 与 max absolute difference 三重门槛判定；不会
  用“0 个梯度也 PASS”的假阳性。

## 门禁判断与下一步

- G1：满足。
- G2：**未满足**。单卡一步 smoke 只证明最小训练链路；仍需 ZeRO-3 正式 SFT 完成、
  checkpoint 加载验证及 eval pass@1 > 0。
- G3/G4：未开始。

后续顺序固定为：tmux 内逐卡检查 GPU2/4/6/7 → UUID 容器隔离 → NCCL preflight →
修复 gradient checkpoint + ZeRO-3 兼容性 → 24K 单步显存验证 → 238 条正式 SFT →
adapter 加载及 2–3 任务快速评测。任何一步失败都先保留日志，不越级进入 RL；无需以
恢复 GPU5 或节点重启为前置条件。
