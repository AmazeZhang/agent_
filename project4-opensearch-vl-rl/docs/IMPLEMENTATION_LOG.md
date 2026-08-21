# OpenSearch-VL 实施状态与问题日志

> 建立日期：2026-08-21  
> 更新规则：每完成一个可独立验收的步骤即更新本文、提交并推送。失败、阻塞和负面结果同样保留。  
> 声明边界：本文严格区分“规划”“工程验证”“训练更新”和“效果证据”。

## 当前摘要

| 项目 | 状态 |
|---|---|
| 上游源码 | 已固定到 submodule commit `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0` |
| 源码审计 | 已完成第一轮 |
| 安全规范 | 已建立，尚待 P0 脚本落地 |
| P0 受管运行 | 已完成 CPU 安全门禁；真实 GPU preflight 留待 P1 smoke 前执行 |
| 推理环境 | 未创建 |
| SFT 环境/训练 | 未开始 |
| RL 环境/训练 | 未开始 |
| 消融 | 未开始 |
| 本地效果结论 | 无；不得引用上游论文数字作为本地结果 |

## 固定运行边界

- 物理 GPU0 永久禁用。
- GPU5 仅在本项目中有条件恢复；必须通过专项门禁并显式设置 `ALLOW_UNSTABLE_GPU5=1`。
- 所有 GPU 作业使用新 Run ID、命名 tmux 和项目四受管脚本。
- 不使用全局 `pkill`、`killall`、`ray stop --force` 或 `tmux kill-server`。
- 大文件只写入 `/media/imc/data/yzy/agent/project4-opensearch-vl-rl/`。
- 批量下载不使用 Clash 7890；7891 属于同一 Clash 进程，未经再次确认也不用于大文件。
- 第三方 vendor 保持固定版本；本地改动使用项目四包装器、配置或可审查 patch。

## 阶段台账

| 阶段 | 状态 | 进入条件 | 完成证据 |
|---|---|---|---|
| P0 安全门禁 | 已完成 | 安全规范已审阅 | 受管脚本、CPU 假 GPU/进程组测试通过 |
| P1 环境冻结 | 未开始 | P0 通过 | 待补 |
| P2 资产准备 | 未开始 | 环境方案确认、下载清单和空间预算完成 | 待补 |
| P3 安全推理 | 未开始 | 固定模型/数据、本地工具安全补丁通过 | 待补 |
| P4 Agentic SFT | 未开始 | 推理闭环通过 | 待补 |
| P5 SFT→RL rollout-only | 未开始 | SFT checkpoint 通过固定对照 | 待补 |
| P6 小规模 RL | 未开始 | rollout/reward/mask 可审计 | 待补 |
| P7 消融 | 未开始 | RL 1→resume→5/20 step 通过 | 待补 |
| P8 结果审计 | 未开始 | 有 held-out、baseline 和原始结果 | 待补 |

## 问题与决策记录

### 2026-08-21：上游不是一键复现状态

- 现象：RL 默认从原始 Qwen3-VL 初始化，默认 RLOO；`visit` 未注册；工具命名不一致。
- 影响：不能直接把上游脚本执行成功视为论文双阶段复现。
- 决策：显式接入本地 SFT checkpoint，分别保留 RLOO 工程 baseline 和 GRPO 论文链路，增加工具注册测试。
- 状态：待实现。

### 2026-08-21：PythonInterpreter 不是真正沙箱

- 现象：上游在训练进程内使用 `exec` 和字符串黑名单，同时允许网络模块；线程超时不能终止执行线程。
- 影响：存在任意代码、网络访问和资源耗尽风险。
- 决策：默认从项目四工具注册中移除。未来只有独立非 root、只读、无网络、带 cgroup/超时的沙箱实现通过后才考虑恢复。
- 状态：待实现。

### 2026-08-21：Clash 流量限制与直连情况

- 现象：shell 默认继承 `127.0.0.1:7890`；Hugging Face 直连 443 超时。
- 小流量验证：PyPI、GitHub、ModelScope、`hf-mirror.com` 直连可用；7891 SOCKS 可访问 Hugging Face，但与 7890 属于同一 Clash 进程。
- 决策：环境依赖和权重优先使用可校验直连源；批量下载清空所有代理变量；7890/7891 不用于大文件。
- 状态：已写入安全规范，待 P0/P1 启动器强制执行。

### 2026-08-21：GPU5 历史故障与本项目例外

- 现象：GPU5 曾有 PCIe Link Down、Xid 79；当前 NVML 快照恢复且空闲。
- 用户授权：允许 OpenSearch-VL 项目有条件恢复 GPU5。
- 决策：默认仍拒绝；按 NVML/Xid→单卡压力→邻卡 NCCL→六卡 NCCL→受管 1-step 顺序晋级。任一异常立即撤销候选资格。
- 状态：门禁待实现，尚未运行 GPU5 测试。

### 2026-08-21：根分区空间不足

- 现象：根分区只剩约 91 GiB；数据盘约有 2.2 TiB 可用。
- 决策：环境、模型、数据、缓存、Run 和 checkpoint 全部放入项目四数据盘命名空间；启动门槛初始设为至少 300 GiB 可用。
- 状态：路径规范已确定，目录尚未创建。

## 变更记录

### Step D0：审计、安全规范与实施规划

- 状态：完成并推送，commit `db2097a`。
- 产物：
  - `docs/SOURCE_AUDIT_2026-08-17.md`
  - `docs/EXPERIMENT_SAFETY.md`
  - `docs/REPRODUCTION_IMPLEMENTATION_PLAN_2026-08-21.md`
  - `docs/IMPLEMENTATION_LOG.md`
- 验证：`git diff --check` 通过；未启动训练、未下载模型和数据。

### Step P0：项目四受管运行与精确停止

- 状态：完成，等待提交与推送。
- 新增：
  - `scripts/gpu_guard.sh`
  - `scripts/preflight.sh`
  - `scripts/run_managed.sh`
  - `scripts/start_tmux_run.sh`
  - `scripts/stop_managed.sh`
  - `tests/test_p0_safety.sh`
- 验证：
  - 物理 GPU0 无条件拒绝；
  - GPU5 默认拒绝，只有 `ALLOW_UNSTABLE_GPU5=1` 才通过列表门禁；
  - 目标卡出现未知 Compute Process 时拒绝；
  - 默认清空继承的 Clash 代理；
  - 新 Run 目录拒绝覆盖；
  - `stop_managed.sh` 校验 Run token，只终止对应进程组，无关进程保持存活；
  - 正常/停止路径均记录退出信息和 GPU cleanup；
  - `bash -n` 与 `git diff --check` 通过；当前主机未安装 shellcheck，已记录但不为进入 P1 的阻塞。
- 命令：`bash project4-opensearch-vl-rl/tests/test_p0_safety.sh`
- 结果：`P0 safety tests: PASS`。
- 资源：测试仅使用 `/tmp` 和伪造的 `nvidia-smi`，未调用真实 GPU、未下载依赖。
