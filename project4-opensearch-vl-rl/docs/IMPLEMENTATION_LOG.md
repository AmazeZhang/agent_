# OpenSearch-VL 实施状态与问题日志

> 建立日期：2026-08-21  
> 更新规则：每完成一个可独立验收的步骤即更新本文、提交并推送。失败、阻塞和负面结果同样保留。  
> 声明边界：本文严格区分“规划”“工程验证”“训练更新”和“效果证据”。

## 当前摘要

| 项目 | 状态 |
|---|---|
| 上游源码 | 已固定到 submodule commit `c5c02a49780e26ae9cb6f1fb56731d1e594d59f0` |
| 源码审计 | 已完成第一轮 |
| 安全规范 | 已建立并由 P0 受管脚本落实 |
| P0 受管运行 | 已完成并推送；真实 GPU1 preflight 和受管 smoke 通过 |
| 推理环境 | 8B 基座已校验；离线单图生成通过 |
| SFT 环境/训练 | 独立环境已冻结；合成 agentic 数据解析/标注 smoke 通过，尚未执行参数更新 |
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
| P1 环境冻结 | 已完成 | P0 通过 | freeze、CPU import、受管 GPU1 FlashAttention 正反向 smoke |
| P2 资产准备 | 部分完成 | 环境方案确认、下载清单和空间预算完成 | 8B 基座已校验；SFT-36K 清单完成但 LFS 下载受阻 |
| P3 安全推理 | 进行中 | 固定模型/数据、本地工具安全补丁通过 | 基座离线单图 smoke 通过；agent 工具闭环未开始 |
| P4 Agentic SFT | 进行中 | 推理闭环通过 | 合成数据生成、工具调用解析和监督标签 smoke 通过 |
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
- 状态：数据盘目录已创建；环境、缓存与下载均写入项目四命名空间。

### 2026-08-21：上游 SFT extras 声明缺失

- 现象：README 使用 `pip install -e ".[torch,metrics,deepspeed,ray]"`，但当前
  `SFT/pyproject.toml` 没有 `project.optional-dependencies`。
- 影响：照抄 README 不会安装这些 extra，环境表面成功但缺少训练依赖。
- 决策：先固定 torch 栈，再 editable 安装 SFT，显式安装 DeepSpeed、Ray 和 VL 工具包；保存完整 freeze。
- 状态：已规避并记录。

### 2026-08-21：大文件源与 FlashAttention 构建

- 现象：PyTorch 官方 CUDA index 和 PyPI 直连可连接但吞吐过低；FlashAttention 官方
  GitHub release 资产 CDN 超时。华为云 PyPI wheel 分段测速约 3.7 MiB/s。
- 决策：所有代理变量清空；PyPI 包改走华为云镜像。FlashAttention 2.7.4.post1 从源码
  限并发构建，禁用 setup.py 自动尝试 GitHub wheel，仅生成 sm80 cubin。
- 结果：构建成功；物理 GPU1 上 sm89 正反向测试 output/gradient 均 finite。
- 状态：已解决。

### 2026-08-21：HF Xet 与模型下载吞吐

- 现象：hf-mirror 配合 huggingface-hub 1.28 默认进入官方 Xet CAS，返回 401；禁用 Xet 后
  普通镜像约 0.08 MiB/s。ModelScope 单连接约 0.3–0.4 MiB/s。
- 决策：停止的下载均只终止精确独立进程组并保留缓存；ModelScope 客户端设置每个 shard
  8 路 Range 下载。完成后逐文件 SHA256 对照固定 HF revision 的 LFS OID，不能仅凭文件名验收。
- 状态：8B 基座下载中；未使用 Clash 7890/7891。

### 2026-08-21：CPU 数据检查会误判多卡训练

- 现象：仅做数据解析/分词时，LLaMA Factory 看到主机 8 张物理 GPU，要求使用分布式启动器，
  首次检查在加载模型前即安全退出。
- 影响：CPU-only 检查若不隐藏 GPU，会被误判为训练任务；没有占用或修改任何 GPU。
- 决策：CPU 数据检查显式设置 `CUDA_VISIBLE_DEVICES=`，并开启 HF/Transformers/Datasets 离线模式；
  同时将 `HF_HOME`、`HF_DATASETS_CACHE`、`TRANSFORMERS_CACHE` 固定到项目四数据盘；真正训练仍必须
  经项目四 tmux 受管启动器选择物理卡。未设置缓存路径的复验被只读根盘正确拒绝，设置后通过。
- 状态：已解决；4 条合成记录解析成功，首样本 438 tokens、68 个监督 tokens、2 张图像。

### 2026-08-21：Qwen3-VL 模板 reasoning 警告

- 现象：LLaMA Factory 提示 `qwen3_vl` 是 reasoning 模板，并建议非推理任务考虑
  `qwen3_vl_nothink`。
- 决策：上游项目 SFT 配置明确使用 `qwen3_vl`，当前工程 smoke 保持上游对齐；此警告保留，
  不能据此宣称最终训练模板已完成效果验证。
- 状态：非阻塞，后续真实数据对照时复核。

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

- 状态：完成并推送，commit `7c4c2b8`。
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

### Step P1：独立 SFT 环境与真实 GPU 栈 smoke

- 状态：完成；本文档、freeze 与 GPU smoke 测试随同一 P1 提交推送。
- 环境：数据盘 `envs/sft-py311`，Python 3.11.15、torch 2.6.0+cu124、DeepSpeed 0.18.4、
  Ray 2.34.0、FlashAttention 2.7.4.post1；完整清单见 `environments/sft-py311.freeze.txt`。
- CPU 验证：核心训练包、LLaMA Factory CLI、FlashAttention 导入通过。
- GPU preflight：物理 GPU1 空闲，约 24.1 GiB 可用，数据盘约 2.2 TiB 可用；GPU0 未涉及。
- 受管 Run：`p1-gpu-stack-smoke-20260821`，物理 GPU1，tmux + 独立进程组，代理已净化。
- GPU 结果：进程内仅 1 张逻辑卡；RTX 4090 D sm89；BF16 FlashAttention forward/backward
  输出与梯度均 finite；`exit_code=0`，cleanup 后无遗留 compute process。
- 边界：这不是模型加载、训练更新或效果证据。

### Step P2a：固定资产清单、8B 基座和校验工具

- 状态：8B 基座部分完成并推送，commit `3bc8839`；SFT-36K 下载阻塞，不能把整个 P2 标记完成。
- P1 提交：`41ca701`，已推送。
- 清单：8B 基座固定 HF revision `0c351dd0...`；SFT-36K 固定 revision `2c1c460a...`。
- 模型：ModelScope 分段直连完成；15 个功能文件、17,545,914,364 B 与 HF 清单一致；
  `.gitattributes` 的镜像差异被显式忽略并记录。
- 数据：发布清单约 13.07 GB；ModelScope 无镜像，hf-mirror LFS 对象存储链路零字节卡住；
  未使用 7890/7891，未声称下载完成。
- 工具：有界 Range、禁用继承代理、拒绝覆盖、size/SHA256 校验和 snapshot manifest 复核；
  10.29 MB 真实下载与重复执行测试通过，CPU 单测通过。
- 边界：下一步合成数据只用于 SFT 工程 smoke，不替代官方 36K 数据。

### Step P3a：8B 基座离线多模态推理 smoke

- 状态：完成；推理测试与本记录随同一 P3a 提交推送。
- 受管 Run：`p3-base-infer-smoke-dtype-20260821`，物理 GPU1，命名 tmux + 独立进程组；
  进程内 `CUDA_VISIBLE_DEVICES=1`，GPU0 不可见。
- 加载：固定本地 `Qwen3-VL-8B-Instruct`，BF16、FlashAttention 2、
  `local_files_only=True`、`trust_remote_code=False`。
- 输入：程序生成的 224×224 白底红色方块，只用于确定性管线 smoke，没有引入外部图片。
- 结果：输入 83 tokens，输出 2 tokens，回答 `red`；峰值分配显存 16.362 GiB；
  `exit_code=0`，cleanup 后物理 GPU1 无 compute process。
- 警告：Transformers 报告模型 generation config 中 `temperature/top_p/top_k` 在贪心生成下被忽略；
  不影响本次 `do_sample=False` 结果，后续训练配置不继承这些生成参数。
- 边界：只证明基座离线图文加载/生成；尚未启用搜索、访问网页或其他 agent 工具，也不是效果评测。

### Step P4a：合成 agentic SFT 数据与监督链路 smoke

- 状态：完成；等待本步骤提交推送。
- 数据：在数据盘生成 4 条明确标记为 `synthetic=true`、`pipeline-smoke-only` 的样本，
  每条含原图、`crop` function call、工具 observation 图和最终答案；生成器拒绝覆盖已有目录。
- 校验：通过固定本地模型 tokenizer 和 LLaMA Factory `qwen3_vl` 模板实际解析；4 条记录均进入
  train split，首样本 438 tokens、68 个监督 tokens、2 张图像。
- 隔离：检查在 `CUDA_VISIBLE_DEVICES=` 和完整离线模式下完成，未使用 GPU、搜索 API 或网络。
- 边界：合成数据只用于验证工程通路，绝不替代官方 Search-VL-SFT-36K，也不能形成训练效果结论。

### Step P4b：有界单卡 LoRA SFT 启动器

- 状态：启动器实现和静态检查完成；GPU 参数更新 smoke 待执行。
- 约束：只接受 1–5 optimizer steps、单张非 GPU0 物理卡、项目四受管 Run 环境和全新输出目录；
  checkpoint 只能来自项目四 Run 目录，并校验 LoRA 权重与 trainer state。
- 配置：固定本地 8B 基座和合成数据，HF 全离线，缓存写项目四数据盘；BF16 + FlashAttention 2、
  rank-8 LoRA，视觉塔和多模态 projector 冻结，单卡 batch 1，gradient checkpointing。
- 验证：Ruff、Python 编译和 `git diff --check` 通过；脱离受管 Run 调用会在模型加载前拒绝。
- 边界：启动器最多执行工程 smoke，不授权真实 36K 数据或大规模训练。
