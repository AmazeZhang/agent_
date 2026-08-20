# P3 clean-upstream GRPO vs GiGPO 状态（2026-08-20）

## 当前判定

本轮实验尚未启动。安全 preflight 在 GPU 检查阶段失败，因此没有创建新的
GRPO/GiGPO smoke、10-step training、checkpoint merge 或训练后 confirm-256 run。

## 已完成

- Search-aware Reward 0009 及 0007/0008 训练线冻结；本轮不使用 0001-0009。
- clean upstream worktree 固定为 `20bd331bdbc9026a5668e11362178e10ab7400c8`，状态干净，未发现
  `search_aware_step_reward` 标记。
- `Qwen2.5-3B-Instruct` 模型资产存在且包含完整配置、tokenizer 和两份 safetensors 权重。
- 共享入口 [`scripts/run_p3_grpo_gigpo_shared.sh`](../scripts/run_p3_grpo_gigpo_shared.sh)
  已由 commit `61fc0f2` 提交。GRPO/GiGPO 的 Hydra override 解析均已通过。
- Step0 Instruct clean-upstream confirm-256 greedy 已完成，作为训练前基线：EM 65/256
  （25.39%）、搜索率 177/256（69.14%）、有效查询率 333/333（100%）、
  search-to-answer 98.87%、search-to-correct 23.16%。

## 本次安全检查

- `tmux`：无活跃项目会话。
- 训练、vLLM、Ray、Retriever 进程：未发现残留进程。
- 数据盘：`/media/imc/data` 可用约 2.4 TB，满足磁盘空间要求。
- GPU preflight：失败，`nvidia-smi` 返回 exit 9：
  `couldn't communicate with the NVIDIA driver`。
- Retriever health：失败，`127.0.0.1:18080/health` 无法连接；当前没有监听进程。

## 未完成与恢复条件

以下步骤必须在 GPU driver 恢复、Retriever 按既定方式启动并重新通过只读 preflight 后，
按顺序执行：

1. GRPO 1-step smoke；
2. GiGPO 1-step smoke，并核对 step-group/advantage 证据；
3. 从同一 Instruct Step0 分别执行 GRPO10 和 GiGPO10；
4. 合并两个 checkpoint；
5. 在同一 confirm-256 上评测 Step0、GRPO10、GiGPO10；
6. 汇总 EM、搜索率、有效查询率、search-to-answer、search-to-correct、GRPO reward
   方差、GiGPO step-group 大小/有效比例、episode/step advantage 和资源状态。

不得因本次 preflight 失败而使用 GPU0/GPU5、跳过 `run_managed.sh`、绕过 Retriever health
检查、重启未知服务或复用已有 Run ID。

## 证据边界

当前全局实验审计仍为 `WARN`。现有 Step0 结果只支持作为同条件训练前基线，不支持质量提升、
收敛或完整 Search-R1 复现结论。
