# 项目二：Coding Agentic RL

## 上游源码

```text
vendor/SWE-agent/
vendor/SWE-smith/
vendor/rllm/
```

这些目录由工作区根仓库作为 submodule 管理。首次克隆后，需要将
`patches/sweagent-shallow-reset.patch` 应用到 SWE-agent；该补丁让真实仓库任务只按目标
commit 做浅拉取，避免意外暴露完整父历史：

```bash
git apply --directory=vendor/SWE-agent patches/sweagent-shallow-reset.patch
```

## 隔离环境

```bash
source .venvs/swe-tools/bin/activate
source .venvs/rllm-base/bin/activate
```

当前 `rllm-base` 用于 API、CLI 和 CUDA 基础检查，尚未安装 `rllm[verl]` 完整训练栈。完整训练栈、模型和 Docker 镜像均需在磁盘与 GPU 资源确认后单独安装。

所有 GPU 命令必须通过 `scripts/start_gpu_smoke_tmux.sh` 同类入口启动，并使用物理 GPU 1–7。

## DeepSeek + SWE-agent 可行性验证

- 配置：`configs/deepseek-feasibility.yaml` 和 `configs/deepseek-model-registry.json`。
- SWE-ReX 镜像：`agent/swe-rex-py311:20260806`，定义见 `environments/swe-rex-py311.Dockerfile`。
- 测试仓库：`fixtures/buggy-calculator`，基线为 2 failed / 1 passed。
- SWE-agent 使用 `deepseek-v4-flash` 完成 12 次函数调用，自主复现、定位并修复除法实现错误。
- 轨迹记录 14,245 输入 token、389 输出 token，instance cost `$0.0008292592`，退出状态 `submitted`。
- 补丁：`runs/sweagent-feasibility-deepseek-v4-flash-20260806-retry2/0ed001/0ed001.patch`。
- 独立复验：在断网 CPU-only 容器中重新应用补丁，3/3 单元测试通过，退出码 0。

当前边界：这里只验证了单个受控仓库上的端到端 coding-agent 链路；尚未安装 `rllm[verl]`、执行 SFT/GRPO，亦未在批量真实仓库任务上评估泛化能力。

## 五任务 Pilot

受控任务已扩展到 calculator、slug、inventory、pagination 和 dedupe。DeepSeek + SWE-agent 对 5/5 任务提交补丁，所有补丁经断网 CPU-only 容器独立复验，共 15/15 测试通过。

- 合计 68 次 API 调用；
- 100,006 输入 token、2,627 输出 token；
- 轨迹记录总成本约 `$0.0045382`；
- 8 次函数调用格式自动重试，最终均恢复。

重新汇总：

```bash
.venvs/swe-tools/bin/python scripts/summarize_pilot.py
```

详细结果与训练门槛见工作区根目录 `PILOT_REPORT.md`。

## 验证轨迹导出

```bash
sg docker -c '.venvs/swe-tools/bin/python scripts/export_verified_rollouts.py \
  --verify-containers --output-dir datasets/verified-pilot5'
```

当前导出 5 条、拒绝 0 条；每条都保留完整消息、动作轨迹、补丁、模型统计和离线容器证据。

## 真实 SWE-bench 探针

`pydicom__pydicom-1458` 已完成一条真实仓库 rollout。模型补丁通过 Float/Double Float 直接复现，但未满足官方测试中的 `BitsStored` 与 `PlanarConfiguration` 条件；离线 CPU 评测退出码 1，gold patch 在同一评测器上 4/4 通过。因此该条保留为 reward 0 失败轨迹，不进入正样本。

真实任务相关大文件位于 `/media/imc/data/yzy/agent/project2/real-probe/`，未使用 GPU。

## SWE-smith Pilot 20

`scripts/curate_swesmith_pilot.py` 从官方 train split 固定了 20 条候选：10 个轻量 Python 仓库、每库 2 条，保留 bug patch、官方镜像、FAIL_TO_PASS 和 PASS_TO_PASS。数据位于 `/media/imc/data/yzy/agent/project2/swesmith-pilot20/`。

前两条 OAuthlib 任务均已完成：各自 bug 基线都是 4/4 FAIL_TO_PASS 失败；DeepSeek 分别经 25 和 41 次调用提交补丁，均只修改 1 个源文件；两个补丁各自在独立、含隐藏测试的评测 checkout 上得到 673 passed、2 skipped、退出码 0，reward 均为 1。模型运行分支不含 FAIL_TO_PASS 测试，评测分支单独恢复测试，未发生测试泄漏。当前 2/2 只证明链路可行，不能代表跨仓库成功率。

第三条 Pygments VimLexer 任务得到首个严格失败：模型正确修复 `**options`，基础 API 测试转为通过，但两个隐藏 golden 测试仍失败；完整结果为 5114 passed、2 failed、16 skipped，reward 0。上游问题描述只提到 options，而 bug patch 还破坏了 builtin 映射和 `is_in` 逻辑，因此该条同时标记为 `problem_statement_underdescribes_bug_patch`，不会把任务质量问题伪装成纯模型能力问题。

完整性回查发现，第 4–6 次本地上传保留了父提交，其中第 5、6 条模型直接读取了 `Bug Patch` 与原始实现；第 4 条虽未读取，但答案仍可访问。因此三条全部从可信指标中排除，旧轨迹保留并标记 invalid。当前严格可信结果退回前 3 条：2 个 OAuthlib 成功、1 个 Pygments 失败。

本地实例生成器现强制拒绝存在 `HEAD^` 的仓库；受影响任务已生成单提交、父历史不可访问的净化快照，必须重跑后才能重新计入指标。

Funcy curry/compose 已完成首条净化重跑：模型无法访问父历史，修复 curry 但遗漏空 `compose()`/`rcompose()` 的 identity 行为，完整结果为 201 passed、2 failed，reward 0。当前可信结果为 4 条、3 个仓库：2 成功 / 2 失败；另有 3 条泄漏运行保留但排除。
