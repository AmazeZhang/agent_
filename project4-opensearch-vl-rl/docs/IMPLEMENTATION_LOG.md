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
| SFT 环境/训练 | 合成 agentic 数据 1→2→5 step LoRA、断点续训和 adapter 离线推理闭环通过 |
| RL 环境/训练 | 本地 reward/rollout-only 已通；两轮随机组审计均因组内零方差拒绝进入 optimizer，尚未启动 RL 更新 |
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
| P3 安全推理 | 本地工具闭环已通 | 固定模型/数据、本地工具安全补丁通过 | 基座单图 smoke；WIT image/text 双工具真实轨迹可执行 |
| P4 Agentic SFT | challenge-v5 1-step 完成 | 检索验证数据与 loss mask 通过 | 新协议 checkpoint 已生成，待固定 dev20 复评 |
| P5 SFT→RL rollout-only | 两轮随机组审计完成但未过门 | challenge SFT 通过固定对照 | 第二轮对齐上游采样，四类 train prompt 各 4 轨迹仍全部零方差 |
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

### 2026-08-21：上游 JSON 配置入口不可用

- 现象：首个 1-step 受管 Run `sft-lora-1step-20260821` 在参数解析阶段退出；固定版解析器对
  JSON 路径调用 `json.load(Path(...))`，触发 `AttributeError`。
- 影响：没有加载模型或执行参数更新；Run `exit_code=1`，GPU1 cleanup 为
  `compute_processes=none`，失败日志和配置完整保留。
- 决策：使用同一解析器原生支持且已由上游示例采用的 YAML 配置，不修改 vendor 源码；用新 Run ID
  重试，绝不覆盖失败 Run。
- 状态：已解决；YAML 新 Run 完成 1-step 参数更新并通过 cleanup。

### 2026-08-21：RL 已到联网 API 门禁

- 现象：RL 主执行器实际依赖 Serper/gateway 搜索；query-utility reward 没有 judge key 时固定为
  0；image search 还依赖发布仓库中缺失的 COS uploader 与 lens provider。
- 一致性问题：根 README 的 `SERPER_API_KEY` 与代码实际读取的 `SERP_API_KEY` 不一致；
  `VisitTool` 存在但未注册，`PythonInterpreter` 却默认注册。
- 资源问题：上游单机 preset 要 8 GPU/80GB 级卡，本机排除 GPU0 后只有 7×24GB，不能原样运行。
- 决策：遵照用户指示，在 API 配置前停止；不创建 RL 环境、不启动 rollout/训练。详细门禁见
  `docs/RL_API_AND_RESOURCE_GATE_2026-08-21.md`。
- 状态：等待用户提供或选择搜索、judge 和 image-search provider 方案。

### 2026-08-21：RL-8K 不是单一 Wikipedia 子集

- 固定 revision：`8ef567289043eef004b13da83b0e7bb7f5ae2daa`；已下载 8.9 MB JSONL，未下载图片包。
- 实测：7,992 行中 LiveVQA 3,746、WebQA 1,507、demo_1k 1,000；`wiki_en/wiki_zh/wikiart/palace`
  合计 1,555。每行一张且 7,992 个图像引用全部唯一。
- 影响：OVEN/WIT 与百科实体子集匹配，但不能在无分层覆盖率证据时替代完整开放网络工具。
- 决策：不修改官方 RL 样本；实现本地优先、低置信度可审计失败的 provider，覆盖率按八个子集分层。
  工程 pilot 可以使用派生 ID 清单，但必须与完整复现实验明确区分。
- 证据：`scripts/audit_rl_dataset.py` 和数据盘
  `datasets/manifests/search-vl-rl-8k-audit-8ef5672.json`；详细研究见
  `docs/OFFLINE_MULTIMODAL_RETRIEVAL_2026-08-21.md`。
- 状态：元数据审计完成；尚未下载 OVEN gated 资产或运行 RL。

### 2026-08-22：固定评测暴露协议歧义与 gold 提取错误

- v1 现象：Base、SFT step1、SFT step5 都把图像描述或文件名传给 `image_search`，而环境只接受未在
  prompt 中声明的 `img_1`；三组均 fatal。该结果证明协议缺字段，不能解释成模型能力或 SFT 无效。
- v2 修复：在 system、user 和工具 schema 中显式声明 runtime handle。Base dev5 的工具路径 5/5 正确、
  fatal 为 0，最终语义也正确，但旧 rubric 没有明确要求 `Title/Evidence` 两行格式。
- v3 修复：显式最终格式后，Base dev5 工具路径、标题和格式均为 5/5，严格证据为 4/5。
- v3 根因：`first_sentence()` 仅在标点位置不小于 39 时截断。`Cinder` 第一短句被错误并入第二句，模型
  实际输出与 prompt 所要求的第一句一致。旧 report 原样保留，不回写分数。
- v4 决策：第一处终止标点即结束，无标点才截断到 360 字符；manifest 显式固定提取算法，训练和评测
  fail-closed 校验该字段，并新增短句回归测试。
- 状态：v4 数据已在受管 GPU1 上 120/120 检索验证通过，待固定 Base 复评。

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

- 状态：完成；启动器已推送为 `bb9e6b6`，YAML 兼容修复已推送为 `4a50130`。
- 约束：只接受 1–5 optimizer steps、单张非 GPU0 物理卡、项目四受管 Run 环境和全新输出目录；
  checkpoint 只能来自项目四 Run 目录，并校验 LoRA 权重与 trainer state。
- 配置：固定本地 8B 基座和合成数据，HF 全离线，缓存写项目四数据盘；BF16 + FlashAttention 2、
  rank-8 LoRA，视觉塔和多模态 projector 冻结，单卡 batch 1，gradient checkpointing。
- 验证：Ruff、Python 编译和 `git diff --check` 通过；脱离受管 Run 调用会在模型加载前拒绝。
- 首次 Run：`sft-lora-1step-20260821` 在上游 JSON 参数解析阶段安全失败；无模型加载、无 GPU
  遗留，详见问题记录。失败 Run 不覆盖、不删除。
- 成功 Run：`sft-lora-1step-yaml-20260821`，物理 GPU1；`exit_code=0`，cleanup 后 GPU1
  `compute_processes=none`，GPU0 未参与。
- 参数更新证据：21,823,488 个 LoRA 可训练参数（总参数的 0.2483%）；trainer state
  `global_step=1`，checkpoint 含 87,368,144 B `adapter_model.safetensors` 和 optimizer state。
  单步记录 loss 2.64398、grad norm 2.96580，只作为计算链路有限值检查，不作为效果结论。
- 资源：监控采样看到 GPU1 最高 18,765 MiB 已用显存；这是离散采样值，不声明为精确峰值。
- 断点续训：Run `sft-lora-resume-step2-20260821` 从 checkpoint-1 明确恢复 model、optimizer、
  RNG 和 scheduler state，日志确认从 global step 1 继续并跳过首批；checkpoint-2 的
  `global_step=2`，历史同时保留 step 1/2。
- 续训有限值：step 2 loss 2.51669、grad norm 3.15616；只说明第二次参数更新为有限值。
  checkpoint-1/2 adapter SHA256 分别为 `10f43014...`、`2d881fb4...`，证明保存权重发生变化。
- 续训安全：Run `exit_code=0`，cleanup 后 GPU1 `compute_processes=none`，GPU0 未参与。
- 5-step 稳定性：Run `sft-lora-resume-step5-20260821` 从 global step 2 恢复；step 3/4/5
  loss 为 2.25844/1.83030/1.41446，grad norm 均有限；checkpoint-5 adapter SHA256 为
  `6be087a3...`。`exit_code=0`，cleanup 后 GPU1 无 compute process。
- 边界：5 条训练记录来自循环使用 4 条合成样本，loss 下降极易过拟合，不能外推为模型质量；
  启动器不授权真实 36K 数据或大规模训练。
- adapter 验收：Run `sft-adapter-infer-step5-20260821` 离线加载 checkpoint-5，PEFT
  `active_adapters=["default"]`；确定性单图回答 `red`，输入/输出 83/2 tokens，精确
  `torch.cuda.max_memory_allocated=16.493 GiB`。
- adapter 安全：物理 GPU1，Run `exit_code=0`，cleanup 后 GPU1 无 compute process；GPU0
  未参与，未使用网络或 API。
- P4 结论：SFT 工程闭环已完成，但真实 SFT-36K 数据尚未就绪，因此不能声称完成论文级 SFT
  复现或具备效果证据。下一步只进入 RL rollout/API 依赖审计，不启动大规模训练。

### Step P5a：RL 数据组成审计与离线视觉语料决策

- 状态：完成；RL 图片包下载属于后续 P5b，尚未宣称资产就绪。
- 官方 RL 数据：固定 `Search-VL-RL-8K` revision `8ef5672...`；JSONL 共 7,992 行且字段完整，
  八个来源中 `new_livevqa` 占 46.87%，证明 Wikipedia 图像库不能无条件替代开放网页检索。
- 数据原则：不改写官方 RL 样本，不把预测实体、检索 top-k 或答案相关覆盖标签写回原数据；
  pilot 只保存样本 ID，检索轨迹进入独立日志。
- OVEN：官方端点访问 gated 脚本返回 HTTP 401；未绕过访问门禁，也未通过 Clash 下载。
- WIT：选择公开 `wikimedia/wit_base` revision `ff6d4fb3...`；固定 330 个 Parquet、
  308,150,150,366 B。先做单片约 0.93 GB pilot，通过 schema/索引验收后再决定扩展。
- ZIP 安全：新增独立审计/解压器，拒绝路径穿越、反斜杠路径、符号链接、加密、重复成员、
  超限膨胀和覆盖；CPU 单测覆盖正常解压、覆盖拒绝、三类路径逃逸和符号链接拒绝。
- 验证：`python3 -m unittest tests/test_safe_extract_zip.py tests/test_audit_rl_dataset.py`、Ruff、
  `compileall` 和 `git diff --check` 均通过。
- 边界：没有启动 RL、没有调用搜索 API、没有使用 GPU；WIT 全量下载和 20 步以上 RL 仍受门禁。

### Step P5b：本地视觉检索契约与小语料精确索引

- 状态：完成；真实 WIT shard 尚未下载，当前只验收接口和数值实现。
- 实现：新增 `local_retrieval`，以归一化 float32 `.npy` + JSONL metadata + manifest 构建
  memory-map 只读精确余弦索引；输出 staging 原子发布且拒绝覆盖。
- observation：稳定返回 `title/source/summary/entity_id/similarity/corpus/corpus_revision`，并包装为
  上游兼容的 `Tool execution result:` JSON；低于阈值时返回空列表，不把 no-match 伪装成命中。
- 防错：拒绝非有限值、零向量、维度错误、元数据缺失、数量不匹配、非法 top-k/阈值和未知格式。
- 验证：合成三候选的排序、阈值、revision 传播、JSON observation、覆盖拒绝及三类非法输入均通过；
  unittest、Ruff 和 `git diff --check` 通过。
- 边界：精确 NumPy 后端只适合单 shard pilot，不声称能承载 647 万 WIT 样本；真实查询特征预处理
  要等 shard schema 核对后固定，避免与发布的 ResNet-50 embedding 空间错配。

### Step P5c：200 条分层清单与 RL 图片引用门禁

- 状态：完成；实现、CPU 单测、实际 200 条清单和全部图片审计均通过。
- 抽样：`select_stratified_rl_audit.py` 使用 dataset 比例、最大余数配额和 SHA256 稳定优先级；输出
  只含 sample ID、row index 和 dataset，显式记录 `uses_answer_for_selection=false`。修改 answer 内容
  不改变测试中的抽样结果。
- 实际 200 条配额：`new_livevqa=94`、`WebQA=38`、`demo_1k=25`、`wiki_zh=13`、
  `wiki_en=10`、`palace=9`、`wikiart=6`、`new_fvqa=5`。
- 图片门禁：`audit_rl_images.py` 对每个安全相对引用拒绝路径逃逸、反斜杠、符号链接、缺失文件和
  无法解码的内容；Pillow 进行完整 decode verify，并报告格式与尺寸范围。
- 验证：4 个 unittest、Ruff 和 `git diff --check` 通过；覆盖答案无关性、配额、非法样本量、
  JPEG/PNG 正常路径、路径逃逸、损坏图像和符号链接。
- 清单：数据盘 `datasets/manifests/search-vl-rl-8k-offline-audit-200-8ef5672.json`，21,977 B，
  SHA256 `cd16abbe9d91b4a09e0bbeddf012d23c87a0da58928375dab3271a9eee10fa7b`。首次审批失败的
  问题在权限恢复后通过原命令解决，没有改用旁路或改变抽样算法。
- 图片资产：2,693,241,993 B ZIP 整体 SHA256 与官方一致；安全审计确认 7,992 文件、
  2,704,382,981 B 解压大小、最大压缩比 29.44。逐图审计为 JPEG 4,026、PNG 3,149、
  WEBP 817，全部 7,992 个唯一引用存在并可解码，没有路径逃逸或符号链接。
- 报告：ZIP 审计和逐图审计分别为 298 B/792 B，SHA256 `74cb2e9a...`/`8755a6c2...`；原始
  8 路 part 保留作续传/取证证据，未删除。

### Step P5d：本地 Wikipedia 文本检索契约

- 状态：实现与合成 CPU 测试完成；真实 Wikipedia corpus 尚未接入。
- 实现：新增 SQLite FTS5 只读索引，固定 `corpus/corpus_revision`；支持实体 ID 精确 lookup 和
  `unicode61` 全文检索，输出与本地图像检索一致的证据字段及 `Tool execution result:` JSON。
- 安全：构建拒绝覆盖、空字段、重复实体和空语料；查询长度、token 数与 top-k 有界，用户输入先
  token 化再参数绑定，不允许直接注入 FTS 语法；运行时以 immutable read-only URI 打开。
- 验证：合成 bridge/painting 文档通过相关性排序、精确 lookup、no-match、revision 传播、JSON
  observation、覆盖拒绝和非法输入测试；unittest、Ruff 与 `git diff --check` 通过。
- 边界：这是替代在线 `text_search/visit` 的工具契约，不是 Wikipedia 语料就绪或检索效果证据。

### Step P6a：WIT Parquet 安全 schema 审计器

- 状态：工具与合成测试完成；真实 shard 正在非 Clash 直连下载，尚未声称数据就绪。
- 实现：`audit_wit_shard.py` 固定输出文件 SHA256、Parquet 行数/row group/字段类型、压缩 codec、
  压缩与未压缩列大小，并读取最多 1–10 个样本做有界类型摘要。
- 日志边界：图片 bytes 只记录长度与 SHA256，字符串最多预览 160 字符，向量只记录长度、有限值
  数量、范围和 L2 norm；不把完整图片、embedding 或大字段写入报告/终端。
- 安全：报告拒绝覆盖；`sample_rows` 有界；仅 CPU/pyarrow，不使用 GPU、搜索 API 或模型。
- 验证：合成 Parquet 覆盖 image struct、binary bytes、float embedding 与字符串字段；确认报告不含
  原始图片内容，unittest、Ruff 和 `git diff --check` 通过。

### Step P6b：WIT 单 shard 数据与发布向量双索引

- 状态：完成；发布向量索引只用于 schema/字段验证，encoder 校准失败后不用于真实查询。
- 数据：固定 revision `ff6d4fb3...` shard 00000，932,699,916 B，官方 LFS SHA256 校验通过；
  实测 19,629 行、2 个 row groups、2,048 维 float64 embedding 和多语言 Wikipedia 证据。
- 构建：英语优先、中文次优、其他语言回退；稳定 entity ID 由 revision+row index 生成。视觉和文本
  索引各 19,629 条，首样本的 self-search 与 FTS lookup 返回同一英文实体。
- provenance：输出 manifest 固定 source hash、语料 revision、字段选择、embedding 未知 checkpoint
  边界和 CC-BY-SA-4.0；输出目录 staging 原子发布并拒绝覆盖。
- 验证：端到端合成 WIT Parquet 测试及真实首样本 paired lookup 均通过；不构成外部查询效果证据。

### Step P6c：WIT 发布 encoder 校准与统一重编码门禁

- 状态：完成；发布空间校准明确失败，统一 encoder 的真实受管 GPU 重编码完成。
- 权重：torchvision ResNet-50 V1，102,530,333 B，完整 SHA256 `0676ba61b6795bbe...e722fb8a`，
  官方下载器按文件名 hash 前缀校验；缓存位于项目四数据盘。
- 校准：显式 `CUDA_VISIBLE_DEVICES=''`，CPU 上 16 样本；配对余弦均值 0.4416，identity top-1
  0.0625，relative L2 均值 0.9626，未达到 0.99/0.95 双门槛，禁止混用空间。
- 决策：固定 torchvision V1，同时重算候选和查询；不再猜测 WIT 未公开的精确 checkpoint。
- GPU 安全：重编码器必须存在 Project 4 Run ID/dir/token，且进程内恰好一张可见 CUDA 卡；输出拒绝
  覆盖、向量要求有限。启动仍必须经过 tmux/受管脚本和空闲卡检查，GPU0/5 不使用。
- 测试：alignment 接受同空间缩放、拒绝置换空间；WIT image bytes 解码和 managed identity guard
  通过；unittest、Ruff 和 `git diff --check` 通过。
- 真实 Run：`wit-reembed-resnet50-v1-20260822`，物理 GPU1；19,629 张编码完成，耗时 70.238 秒，
  峰值 allocated 800,757,248 B；`exit_code=0`，cleanup 后 GPU1 `compute_processes=none`，GPU0/5
  未参与。Pillow 对一个带 bytes transparency 的 palette image 给出转 RGBA 警告，脚本最终统一
  `convert("RGB")`，没有失败或跳过记录。

### Step P6d：可执行本地 image_search backend

- 状态：实现和真实端到端 smoke 完成；RL-8K 分层检索审计已在 P6e 完成。
- backend：查询图使用与候选完全相同的 torchvision ResNet-50 V1/preprocess；初始化时校验索引
  revision 包含完整 weights SHA256，阻止混合向量空间。
- 路径安全：只读取显式 allowed root 下的本地文件，支持相对引用，拒绝根目录逃逸、非文件和路径
  中任意符号链接；不访问图片 URL。
- smoke：WIT 首图原图 top-1 self similarity 1.0；中心 80% crop top-1 仍为 self，similarity
  0.980778；其余高位候选也是蜈蚣相关实体。该 smoke 不声明对 RL-8K 的覆盖率。
- 验证：encoder revision 一致/不一致与本地路径正常/相对/逃逸/符号链接测试通过；unittest、Ruff
  和 `git diff --check` 通过。

### Step P6e：RL-8K 200 条分层本地检索审计

- 状态：完成；单 WIT shard 不适合直接承载原 RL-8K，后续改用有 provenance 的
  WIT/Wikipedia 派生多跳任务做本地 SFT/RL pilot。
- 实现：`ExactVisualIndex.search_batch()` 一次计算有界 batch 的精确余弦搜索；
  `audit_local_retrieval_coverage.py` 只读固定 200 条清单，校验原 JSONL SHA256，不读取答案做选样或检索。
- 数值：top-1 cosine mean/p50/p90 为 `0.828596/0.821322/0.892470`；按预设阈值有
  189 high、11 medium、0 low。这只是视觉邻近置信代理，不是语义覆盖率或答案正确率。
- 语义核查：200 条中仅 4 条 top-1 similarity `>=0.9999`，候选为 196 个不同实体；抽查可见
  大量高余弦但与问题实体无关的近邻。因此不得用 `>=0.75` 当作可回答样本的自动筛选规则。
- 真实 Run：`wit-rl200-coverage-20260822`，固定 repo commit `de615de`，物理 GPU1，
  `exit_code=0`；启动前后 GPU1 均为 18 MiB，cleanup 确认 `compute_processes=none`，GPU0/5 未参与。
- 证据：数据盘报告 578,640 B，SHA256
  `1963d5d61a95a929d7801387bbf75e7e0c4cf7feb2e55fef83fb7ee2450d5082`；报告含每条 top-3、按八个子集的分层
  数值和 encoder/index revision，不包含 gold answer。
- 启动纠错：首次把 `PROJECT4_DATA_ROOT` 多写了项目子目录；安全守卫在接触 GPU 前拒绝，
  没有进程或结果文件。清理已退出的精确 tmux 会话后，用规定根路径重启并完成。

### Step P6f：WIT/Wikipedia 派生多跳数据与真实 1-step SFT

- 数据状态：完成 120 条 retrieval-verified pilot，按 `entity_id` 分为 train/dev/test
  80/20/20；120 个实体全部唯一，没有原图或同实体跨 split。
- 查询图：由 WIT 候选图做中心 90% crop 并重编码为 JPEG Q92，不用原图恒等 self-query。
  候选选择固定 seed，不使用模型或 gold answer。
- 检索门禁：120/120 变换图 top-1 仍为对应实体，cosine min/mean/median/max 为
  0.938418/0.968947/0.971144/0.991388；若任一样本实体不匹配、低于 0.90 或文本证据不一致，
  发布器会整批失败而不静默丢样。
- 多跳约束：轨迹固定为 `image_search -> text_lookup -> final`。image observation 只显示
  entity candidates，显式删除 page summary；最终证据只能从第二步 text lookup 获得。
- 数据证据：数据盘 `wit-agentic-pilot-v1`；manifest SHA256
  `3c7dd4806fc52a85116df6ee2016149b727f36a2d3edb05bf81c3e1e43ff15e1`，tasks SHA256
  `789adb00a30e17cf141bb72a1d02ce3207eceeef6688e41f28d3daca0048fc7a`。
- GPU 验证 Run：`wit-agentic-pilot-verify-20260822`，固定 commit `95c6d47`，物理 GPU1，
  `exit_code=0`；前后 18 MiB，cleanup 无进程，GPU0/5 未参与。
- SFT 解析：LLaMA Factory 实际解析 8 条抽样，每条 1 张图，1382–1525 tokens，91–128
  supervised tokens；observation 位置全为 `-100`，两个工具调用和 final 被监督。基座是
  Instruct 版，因此正式使用 `qwen3_vl_nothink`，不训练伪造的 hidden reasoning。
- 解析问题：首次 CPU 解析因 HF datasets 默认在只读 home cache 建 lock 失败；显式把
  `HF_HOME/HF_DATASETS_CACHE/TRANSFORMERS_CACHE` 改到 `/tmp` 后通过。该失败不是数据格式错误，也未使用 GPU。
- 真实 SFT Run：`wit-agentic-sft-1step-20260822`，固定 commit `6989ec6`，物理 GPU1。
  80 条 train 数据、rank-8 LoRA、冻结 vision/projector、BF16/FA2；单步 loss 0.4348、grad norm
  2.025，只证明数值和更新链路正常，不是效果结论。
- checkpoint：`global_step=1`，adapter 87,368,144 B，SHA256
  `7f1b7499a090ccebae1ec7201cf175c8fbdcf4f5758e7fedb9770dc24a83813b`。Run `exit_code=0`，
  GPU1 清理后 18 MiB/无 compute process，GPU0/5 未参与。
- 边界：尚未用固定 dev/test 跑 Base 对照，不会因 1-step loss 而声称 SFT 有效；下一步是实现同一
  本地工具环境的 Base/SFT rollout 评测，然后才决定是否续训 5/20 step。

### Step P6g：固定本地 Agent rollout 与评测协议校准

- evaluator：最多 3 turn，实际执行只读本地 SQLite `text_lookup`，图像检索使用已由同 encoder 验证的
  固定 top-k cache；每条保存工具调用、observation、token 数、fatal 和分项分数。Base 与 adapter 共用
  同一执行器，最多 20 条，禁止 GPU0/5。
- 首次失败：`wit-agent-base-dev5-20260822` 因 Transformers 要求所有多轮 message 使用结构化 content，
  在首条生成前退出；修复后保留失败 Run，cleanup 无进程。
- v1 协议负面结果：Base/SFT1/SFT5 dev5 都因隐含图像 handle fatal，不能作为三者效果对比。
- 真实 SFT step5：`wit-agentic-sft-resume-step5-20260822` 从 checkpoint-1 恢复；step 2–5 loss
  `0.451/0.366/0.2007/0.08135`，grad norm 均有限；checkpoint-5 adapter SHA256
  `ab85c84275d0946768ce6fcebbf42eab8dc90639b2fa6a68b4472dda98e8a7da`。这是旧 v1 协议的工程
  checkpoint，不用于宣称效果。
- v2 Base：dev5 工具路径 1.0、fatal 0，说明显式 handle 已修复工具协议；两行格式未固定，旧指标只作
  rubric 校准证据。
- v3 Base：Run `wit-agent-v3-base-dev5-20260822`，工具路径/标题/格式 `1.0/1.0/1.0`，fatal 0，
  evidence/full success `0.8/0.8`；report SHA256
  `069b086dba4f049a0d3324afde4663209328563dae4c5d5d17895489646f6f7e`。唯一失败来自上述 gold bug。
- v4 数据：Run `wit-agentic-pilot-v4-verify-20260822`，commit `7698bcb`，120/120 通过，物理 GPU1；
  `exit_code=0`，cleanup `compute_processes=none`，GPU0/5 未参与。manifest/tasks SHA256 分别为
  `4fbdc0a9db3268445b68e3a09853113a7744b4514956b3d4d0bc621db5d53d27` 和
  `17ea6412a8a83476959242d637eb6d8ca262e2b24032118bf0b7fce3f972d474`。
- 判断：当前实体识别→文本复制任务可能对 8B Instruct 基座过易。完成 v4 Base 复评后，若已接近满分，
  不会用它伪造 SFT/RL 增益；需加入候选冲突、no-match 与工具失败恢复等更难分层任务。
- v4 Base 复评：Run `wit-agent-v4-base-dev5-20260822`，固定 commit `8ce92b1`；5/5 均严格成功，
  `expected_tool_path/format/title/evidence/full_success=1.0`，`fatal_rate=0`。report SHA256
  `d2a5787dac703fa8498e3d3e730c0c99a5ddfc1513871ce4dd1c21ac01f69d45`；物理 GPU1，
  `exit_code=0`，cleanup `compute_processes=none`，GPU0/5 未参与。
- 结论：v4 clean 只作为执行链路正控，不再对它做能被解释成效果增益的 SFT 对比。下一数据版本固定增加
  `candidate-conflict`、`no-match`、`transient-tool-failure` 三类，并按任务类型分别报告，之后才启动新协议 SFT。

### Step P6h：有区分度的离线 Agent challenge

- 任务组成：固定 120 条、train/dev/test `80/20/20`；`candidate-conflict=48`、`clean=12`、
  `transient-tool-failure=36`、`no-match=24`。实体任务仍沿用 entity-disjoint split；no-match 使用明确标记的
  synthetic safety probe，不伪装成真实 WIT 命中。
- candidate-conflict：图像检索返回真实 top-3，但问题要求选择文本证据含唯一关键词的候选；oracle 为
  `image_search -> text_lookup(rank1) -> text_lookup(rank2) -> final`，防止只复制视觉 top-1。
- transient：环境第一次 `image_search` 确定性返回 `retryable=true/TRANSIENT_FAILURE`，第二次才返回固定真实
  cache；oracle 显式监督重试。no-match 返回空候选并要求固定拒答，不能猜测实体。
- v1 失败门禁：真实 `qwen3_vl_nothink` 解析发现部分四步轨迹达到 `cutoff_len=2048`，原因是两次文本
  observation 过长；训练未启动，v1 保留为失败证据。
- observation 修复：`LocalTextIndex.lookup` 固定最多返回 360 字符；challenge SFT 构建时从实际 SQLite
  lookup 取 observation，与评测运行时完全一致，不再误用带视觉 encoder revision 的内部候选记录。
- v2 虽通过解析，但 manifest 未包含 SFT 文件哈希和证据窗口，无法充分区分 observation 变更；保留但不作为
  训练入口。v3 manifest 增加 train/dev/test、dataset_info 哈希及 360 字符契约。
- v3 真实解析：CPU、`CUDA_VISIBLE_DEVICES=`、全离线，80/80 train 记录；每条恰好 1 图，tokens
  min/max `506/1989`，supervised tokens min/max `44/193`，均未触达 2048 截断。
- v3 证据：数据盘 `wit-agentic-challenge-v3`；manifest/tasks/train SFT SHA256 分别为
  `9097b28f0efc6ebb29ef034ec25d7007c61769b3fec056a7540e3a818b491fce`、
  `14435f66801d623ff55045693a31c55cf139f82a6985d2aa24a0c73a1ba6e70b`、
  `81d048997c13d1f637fbdeb72f70b7dd7ba91e0a7d0eb1a60c117724357e9f02`。
- 安全：构建与模板解析均不使用 GPU、网络或 API；下一步只在物理 GPU1 跑 Base dev20 固定评测，之后才
  判断是否进入 challenge SFT。
- 首次 Base dev20 Run `wit-agent-challenge-v3-base-dev20-20260822` 暴露 schema 契约不一致：工具 schema
  允许 `top_k=5`，固定 cache/执行器只允许 3；模型合法生成 5 后被执行器拒绝。该 Run 在第 9 条期间由
  `stop_managed.sh` 按 exact Run token/进程组 TERM，`exit_code=143`，cleanup
  `physical_gpu=1 compute_processes=none`；日志保留，不能算模型结果。
- v4 修复：训练 tools schema、评测 schema、cache 执行器和 manifest 全部固定 `top_k<=3`，新增回归
  断言；v4 再次真实解析 80/80，通过相同 `506–1989` tokens 与 `44–193` supervised token 门禁。
- v4 证据：manifest/tasks/train SFT SHA256 分别为
  `ec1dcc3f424b375fc5f8a78c42f4aa5637acb3db8d406432cdb96bb8f5084479`、
  `14435f66801d623ff55045693a31c55cf139f82a6985d2aa24a0c73a1ba6e70b`、
  `d79ec7dfa0363244e010a0e85ccc823ab294eff63433b6f4710970c70b1d30d7`；v1–v3 均保留但不再作为训练入口。
- v4 Base 重跑首条走出 `image_search + 3×text_lookup`，但 4-turn 环境在 final 前截断；对 top-3
  候选逐个查证是合法策略，因此该限制会错误惩罚谨慎模型。Run
  `wit-agent-challenge-v4-base-dev20-20260822` 按 exact token/进程组停止，`exit_code=143`，cleanup
  `compute_processes=none`，不计入模型结果。
- v5 将环境最大 turn 固定为 5，覆盖最坏合法路径 `image_search + 3×text_lookup + final`；SFT oracle
  仍只查到命中候选后立即 final（4 turn），不监督冗余调用。v5 再次通过完整 80 条真实模板解析。
- v5 manifest/tasks/train SFT SHA256 分别为
  `341194a665682699853fe6704d45bfe49f5e520179011b29b092cb666a7cbbf1`、
  `14435f66801d623ff55045693a31c55cf139f82a6985d2aa24a0c73a1ba6e70b`、
  `d79ec7dfa0363244e010a0e85ccc823ab294eff63433b6f4710970c70b1d30d7`；v1–v4 保留但不作为训练入口。
- v5 首次 Base Run 的首条最终标题/证据正确且无 fatal，但旧 `full_success` 又要求工具序列与最短 oracle
  完全相等；模型多查一个合法候选因此被错误判失败。Run
  `wit-agent-challenge-v5-base-dev20-20260822` 精确停止并保留，不能计入模型结果。
- 评分修复：`full_success` 只由无 fatal + 格式/标题/证据正确决定；最短工具序列另报
  `oracle_path_exact`，作为效率指标而非正确性门。这样仍能比较冗余调用，又不否定有充分证据的正确答案。
- 最终 Base Run：`wit-agent-challenge-v5-base-dev20-v2-20260822`，固定 commit `99e8aa5`，20/20
  完整执行；总体 `full_success=0.55`、`title_exact=0.95`、`evidence_exact=0.55`、
  `format_valid=1.0`、`fatal_rate=0`、`oracle_path_exact=0.65`。
- 分层 full success：candidate-conflict `0.50`（8 条）、clean `0.00`（2 条）、
  transient-tool-failure `0.50`（6 条）、no-match `1.00`（4 条）。候选冲突最短路径只有 `0.125`，
  说明基座常查完更多候选；这作为独立效率信号，不覆盖正确性。
- 报告：SHA256 `d01d6719bff45f40ffb4ccfa93a4d59c92e2a28606bad0d555193f26d6228a60`；
  dataset manifest/tasks hash 与 v5 固定值一致。物理 GPU1，`exit_code=0`，cleanup
  `compute_processes=none`，GPU0/5 未参与。
- 进入判断：challenge 已产生非零、非满分且按任务类型可解释的基线，允许进入最多 1-step challenge SFT
  和同 dev20 复评；仍不授权大规模 SFT、多卡或 RL。
- challenge SFT Run：`wit-agent-challenge-v5-sft-1step-20260822`，固定 commit `00cc479`；80 条 train、
  rank-8 LoRA、冻结 vision/projector、BF16/FA2、batch 1。单步 loss `0.0790444`、grad norm
  `1.454829` 均有限；低 loss 只反映首个抽样轨迹与 Instruct 基座已较匹配，不作为效果结论。
- checkpoint：`global_step=1`，21,823,488 trainable params；adapter 87,368,144 B，SHA256
  `637169695b4b96022e003b2ad59bea780288da0c31aab94a1c64f962856399f5`。物理 GPU1，
  `exit_code=0`，cleanup `compute_processes=none`，GPU0/5 未参与。
- provenance：明确 `fully_synthetic=false`、`contains_synthetic_safety_probes=true`，不隐藏 24 条 no-match
  合成安全探针。下一步只加载该 adapter 跑完全相同的 dev20；未看到 held-out 改善前不续训。
- SFT1 dev20 Run：`wit-agent-challenge-v5-sft1-dev20-20260822`；总体 full success/title/evidence/format/fatal
  与 Base 完全相同，分别为 `0.55/0.95/0.55/1.0/0`，没有任务正确性增益。
- 唯一变化：candidate-conflict 的 `oracle_path_exact` 从 `0.125` 到 `0.25`，总体从 `0.65` 到
  `0.70`；相当于 1 条 held-out 样本少走一次 lookup，是弱行为更新证据，不足以声称 SFT 有效。
- SFT1 评测报告 SHA256 `1bfaa1da7f1a702e84791c6c094e8ca6afbdfbc2f7bdc20a926ece703d2f39d3`；
  adapter/dataset/tasks hash 均与固定记录一致。物理 GPU1，`exit_code=0`，cleanup 无进程。
- 决策：1 step 只消费 1/80 train 样本。允许从 checkpoint-1 有界恢复到 step5，再跑同 dev20；若仍无
  correctness 增益，则停止 SFT 扩大并转向数据/损失分析，不把路径小变化包装成效果。
- challenge SFT step5：Run `wit-agent-challenge-v5-sft-resume-step5-20260822` 从 checkpoint-1 明确恢复；
  step2–5 loss `0.14633/0.07097/0.04002/0.03842`，grad norm `1.6097/0.9843/1.0087/1.4100`，
  均有限。loss 非单调且仅 5/80 样本，不作收敛结论。
- checkpoint-5 adapter SHA256
  `3e308b8f991e0ab6d2113c9a21879e05874b1243c4da22fa604613c85141bd89`；trainer state 保留 step1–5
  历史。物理 GPU1，`exit_code=0`，cleanup 无进程，GPU0/5 未参与。
- 下一步只做同一 dev20 的 SFT5 固定评测；在该结果前不允许 step20 或扩大数据。
- SFT5 dev20 Run：`wit-agent-challenge-v5-sft5-dev20-20260822`；总体 full success 从 Base/SFT1
  `0.55` 降到 `0.50`，evidence exact `0.50`，title exact `0.95`，format `1.0`，fatal `0`，
  oracle path exact 从 Base `0.65` 降到 `0.60`。
- 分层 trade-off：candidate-conflict full success 从 Base `0.50` 升到 `0.625`，但
  transient-tool-failure 从 `0.50` 降到 `0.1667`；clean/no-match 仍为 `0/1.0`。例如 transient
  `wit-00012098` 在 Base 正确，SFT5 却在拿到正确工具结果后输出错误实体 `Claude Choules`，是真实回归而非
  评分格式问题。
- 报告 SHA256 `2ccb7502c2a810f5f958f72ca12af2ebc5e315e609c955004eaca53601b8e20c`；adapter/dataset/tasks
  hash 均与固定记录一致。物理 GPU1，`exit_code=0`，cleanup 无进程。
- 决策：停止 step20 和 SFT 扩大。当前只支持“5-step 改善冲突子类但造成恢复类灾难性遗忘”的结论，
  不支持“SFT 总体有效”。下一步回到 CPU 侧审计采样/损失平衡，并先实现可审计 reward/rollout-only，
  不直接启动 RL 参数更新。
- 离线 reward 合约：保留上游 `0.8*r_accuracy + 0.2*r_query` 和 `r_format` 乘法门；外部 query judge
  替换为规则化的“gold evidence 是否实际取得 × oracle 工具数/实际工具数”，每条均可追溯到保存的 tool call。
- fatal 合约：记录 learnable prefix；fatal 在首 turn 时 hard mask，否则保留前缀。GRPO 组内先均值中心化，
  再仅对 fatal 轨迹做 `advantage=max(0, A)` 单侧 clamp；单测验证 `[1,0]` 组的 fatal 负 advantage
  从 `-0.5` 变为 `0`，正常正 advantage 保留。
- CPU 重放：Base/SFT1/SFT5 的 `r_accuracy/r_query/r_format/total` 分别为
  `0.55/0.90/1.0/0.62`、`0.55/0.90/1.0/0.62`、`0.50/0.85/1.0/0.57`，与分层评测的 SFT5
  回归一致，没有 reward normalization 抬分或隐藏失败。
- 报告：数据盘 `offline-reward-replay-base-sft1-sft5-20260822.json`，SHA256
  `a16a6940f33d2a819e11d52edbe61a4941631d0c49349f0a3e633ce113f584d6`；模式明确标记
  `deterministic-rules-only-no-api`。构建、测试和重放均为 CPU，无网络/API/GPU。
- RL 门禁：贪心单轨迹没有组内 reward 方差，不能直接形成 GRPO 更新。下一步必须先做 1–2 个 train task、
  每题少量随机 rollout 的 group variance/reward hacking 审计；通过前不启动 optimizer step。
- 随机 rollout 审计实现：`audit_stochastic_rollout_groups.py` 复用同一真实本地工具执行器，固定 seed、
  temperature/top-p 和 adapter/data hash；每条保留完整 turn、tool observation、fatal、分项 reward 和
  fatal-aware advantage。输出明确标记 `no-optimizer-no-api`，门禁失败仍原子保留报告并返回非零退出码。
- 首个随机组 Run：`wit-agent-challenge-v5-sft1-stochastic-groups4-20260822`，固定 commit `79c1131`，
  SFT1 adapter；两个 train task 各 4 条，`temperature=0.7/top_p=0.9`，物理 GPU1。candidate-conflict
  `wit-00000885` 四次均输出同一错误实体，reward 全为 `0.15`；transient `wit-00008586` 四次均严格
  正确，reward 全为 `1.0`。两组 population variance 均为 `0`，raw/clamped advantage 全为 `0`。
- 完整性解释：candidate 组 4/4 都是 query-only reward，报告没有把它们算作 accuracy；transient 组
  4/4 accuracy/query/format 都通过。8/8 格式合法、0 fatal，因此失败不是工具协议或格式崩溃，而是当前
  prompt/采样设置下策略与离散 reward 都饱和。不得用跨 prompt 均值差伪造组内 GRPO 方差。
- 报告 SHA256 `082174558e55181349c8854be6498a321f29bdcbe34035462a311f94afdf8889`；受管 Run
  按预期 `exit_code=4` 表示门禁未过，cleanup `compute_processes=none`。GPU1 峰值观测约 18.4 GiB、
  64°C；结束后 18 MiB，GPU0/5 未参与。
- 决策：不启动 1-step GRPO。下一轮应先在不改 reward 的前提下做有界采样校准（更接近上游 rollout
  temperature，并选择更多非饱和 train prompt）；门槛和失败报告必须预先固定。若仍大面积零方差，
  再讨论可追溯的 evidence fidelity shaping，而不是事后归一化或把跨题差异当作组内 advantage。
- 第二轮协议在运行前提交为 `configs/stochastic_rollout_calibration_v2.json`：四种 task type 各取按
  task ID 排序后首个未被 v1 使用的 train prompt，每题 4 条；固定 seed `20260830`。上游项目四
  8B single-node RL 脚本实际使用 `temperature=0.7/top_p=1.0/top_k=-1`，因此本轮严格使用
  `0.7/1.0`，没有为了制造差异而事后升温。
- 批级门槛同样在运行前固定：variable group fraction 至少 `0.25`、format-valid fraction 至少
  `0.75`、fatal fraction 至多 `0.25`。允许零方差 prompt（它们真实贡献零 GRPO gradient），但至少
  1/4 prompt 必须有题内非零 reward variance；禁止把跨 prompt reward 差异用于组内 advantage。
- 第二轮 Run：`wit-agent-challenge-v5-sft1-calibration-v2-g4n4-20260822`，commit `9ff478c`，物理
  GPU1，16/16 格式合法、16/16 full success、0 fatal。candidate-conflict 四条均因额外一次合法 lookup
  得 `0.95`；clean、no-match、transient 各四条均得 `1.0`。4/4 group 的 population variance 和
  raw/clamped advantage 全为 `0`，variable group fraction `0`，批级门禁失败。
- 报告 SHA256 `eb15a211cfef344e492deb734f786e4f0ad5803a815b424bfd9c8a031999eecd`；模式为
  `stochastic-rollout-only-no-optimizer-no-api`，adapter/data/task hashes 均固定。受管 Run 按预期
  `exit_code=4`；GPU1 峰值观测约 18.3 GiB、63°C，cleanup 后 18 MiB/无 compute process，GPU0/5
  未参与。
- 结论：首轮零方差不是 `top_p=0.9` 截断造成；当前抽到的四类 train prompt 对 SFT1 均已饱和，直接
  1-step GRPO 没有更新信号。停止重复扩大同分布采样。下一步应在 CPU 侧构建/筛选更接近决策边界的
  RL prompt，并优先把 evidence fidelity 变成可追溯的分级信号；新 reward 必须先对 Base/SFT1/SFT5
  历史轨迹重放、检查排序和 reward hacking，再允许新的 rollout-only 门禁。
- 决策边界数据构建器：从 retrieval-verified WIT pilot 选择真实 top-3，生成 rank2/rank3 双唯一线索、
  transient+rank2 双线索和 retry 后空结果四类任务。正向线索只来自目标 SQLite summary 且不出现在
  其他 top-3 summary/目标标题；排除线索来自 rank1 且不在目标 summary。最坏 oracle 为 5 turns。
- 首次发布 `wit-rl-boundary-v6`：120 条，train/dev/test `80/20/20`；rank2/rank3 各 36、transient
  24、no-match safety probe 24。120 张图全部 decode；所有 top-3 candidate entity ID 跨 split 零重叠；
  线索、gold entity 和本地 SQLite evidence 逐条一致。manifest/tasks/train SFT SHA256 分别为
  `7a001dd3bc9b974e3e7992d2923083eb54571c678cc591b288d867b21363dc16`、
  `c86929792a5660da2452aeeeda0250fcb295c21b75e19cf2fbc41be1e5981f11`、
  `8ca3b4c3bbe9ef7cec118e0e8a9705010697fc8bf8bdfc9371356288cbd99489`。
- v6 真实模板门禁失败：用离线 CPU、`CUDA_VISIBLE_DEVICES=` 和真实 `qwen3_vl_nothink` 在
  `cutoff_len=4096` 做诊断，80 条 token 范围 `573–2237`；21 条达到/超过拟用的 2048，其中 rank3
  20/24、transient 1/16。最长 2237。v6 不作为训练入口且目录保留，不覆盖。
- 根因与 v7 修复：三次 lookup observation 重复携带 corpus revision、URL 等非决策元数据。v7 固定
  `boundary-compact-v1`：image_search 只返回 `entity_id/title/similarity`，text_lookup 只返回
  `entity_id/title/summary`；完整来源仍由 manifest、source hash 和 text index 固定。不得截短 gold
  evidence 或删除任务判定所需线索。构建器与 compact-field 回归测试已通过。
- v7 发布与模板门禁：数据盘 `wit-rl-boundary-v7`，任务/图片/划分与 v6 相同，仅 SFT/tool observation
  序列改变；manifest/tasks/train SFT/dataset-info SHA256 分别为
  `dd2714d4c8727405d619d760251f7d4edb29c9f4d8dddc0aaa5135a9a78d09e1`、
  `c86929792a5660da2452aeeeda0250fcb295c21b75e19cf2fbc41be1e5981f11`、
  `ba185294248f0d164440b0e6a243246ef8adb20acd3391899b767377393b8541`、
  `c6af8b94cad773bea557393f27c6fbbf9569a1ee0df838fe6853357faefe6ce6`。
- 真实 `qwen3_vl_nothink` CPU 离线解析 80/80：每条 1 图，tokens `573–1485`、supervised tokens
  `74–283`，全部低于 2048 且有监督；显式 `CUDA_VISIBLE_DEVICES=`，无网络/API/GPU。v7 通过数据入口
  门禁，可进入分级 reward 的 CPU 历史重放设计；尚未据此启动 SFT 或 RL。
- evidence-fidelity reward v2：保留总式 `r_format*(0.8*r_answer+0.2*r_query)`；answer 预先固定为
  `0.5*strict_success+0.2*title_exact+0.3*evidence_token_F1`。格式仍是乘法硬门；完整正确仍为 1，partial
  evidence 只提供连续但受限的信号。query 只有在正确 entity 的真实 lookup observation 中确实包含 gold
  evidence 时才成立，再乘 oracle/实际工具数效率；仅调用 entity ID 或 observation 缺证据均为 0。
- v2 单测覆盖 partial evidence、缺失 observation evidence、retry no-match 未走完整 oracle、旧 v1 格式门和
  fatal clamp。旧 rules-v1 默认入口和历史报告保持不变；重放脚本通过显式 `--reward-version` 选择 v2。
- 历史 dev20 CPU 重放：Base/SFT1 的 `r_exact/r_title/r_evidence_f1/r_answer/r_query/total` 均为
  `0.55/0.95/0.74055/0.68717/0.90/0.72973`；SFT5 为
  `0.50/0.95/0.71560/0.65468/0.85/0.69374`，仍正确识别整体回归。
- 分层一致性：SFT5 candidate-conflict total 从 `0.63648` 升到 `0.72689`，但 transient 从
  `0.75640` 降到 `0.51589`，与严格指标的 trade-off 一致。已知 `wit-00012098` 从 Base/SFT1 `1.0`
  降至 SFT5 `0.048`（title 错、evidence F1 0.2、未取得正确 evidence），没有被连续分掩盖。
- reward-hacking 检查：Base/SFT1 的 9 条、SFT5 的 10 条严格错误轨迹中，最高 v2 reward 均仅 `0.552`，
  错误轨迹 `>=0.8` 为 0。报告 SHA256
  `0221e41eb45638149eeaadc34d91d824dd3785cff38a60b81980af3e88ca9671`，路径
  `offline-evidence-fidelity-v2-replay-base-sft1-sft5-20260822.json`；CPU、无网络/API/GPU。
- 判断：v2 通过历史排序和初步防投机门，可接入 v7 rollout-only evaluator；仍须用新随机轨迹检查是否真正
  产生组内方差，不能因为历史重放更细就直接启动 optimizer。
- v7 真实评测接入：`evaluate_local_agent.py` 新增受限 `--dataset-root`，仅允许项目 data 盘下现存的
  processed dataset；同时校验 no-leak 公共合约、可选 tasks hash 和 `tool_observation_schema`。v5 默认
  行为保持不变，v7 自动选择 `boundary-compact-v1`，报告固定实际 dataset path/hash/schema。
- 训练/推理一致性：运行时 compact image observation 只保留 `entity_id/title/similarity`，text
  observation 只保留 `entity_id/title/summary`。用真实 v7 首条 train 任务 `wit-00000885` 和 SQLite
  index 逐字节比对，image/text 两种运行时 observation 均与 `sft_train.json` 完全相同。
- stochastic rollout 审计器同步支持受限 dataset root 和显式 `evidence-fidelity-v2`，并兼容 v1/v2
  的严格成功字段；其余预声明 group variance/format/fatal 门槛不变。相关 17 个单测、Ruff 和 diff check
  通过；本步纯 CPU、无网络/API/GPU，尚未启动模型 rollout 或 optimizer。
- v7 小规模难度基线在运行前固定：dev 四类各 1 条，按类内首个 task ID 选择
  `wit-00001777`（rank3）、`wit-00004467`（rank2）、`wit-00014422`（transient）和
  `boundary-no-match-dev-000`；先 Base、再既有 v5 SFT1 adapter，均 greedy、`max_new_tokens=192`。
  评测器新增显式 task-ID 白名单并保持请求顺序，防止用“前 N 条”误采成单一类别。每个模型单独受管
  Run，只用空闲物理 GPU1；不使用 GPU0/GPU5，不启用 optimizer，结果无论好坏均保留。
- 首次 Base 启动 `wit-boundary-v7-base-dev4-20260822` 在模型加载前失败：误用了宿主 Conda
  `llama_factory` Python，其 `peft/transformers` 来自 user-site 且缺少 `tqdm`，退出码 1。没有安装或升级
  依赖，没有生成轨迹；GPU1 始终 18 MiB/0% 且 cleanup `compute_processes=none`。失败 Run 和 traceback
  原样保留。后续改用此前 SFT/评测已验证并冻结的 data 盘 `envs/sft-py311/bin/python`，不修改环境。
- Base 重试 Run `wit-boundary-v7-base-dev4-v2-20260822` 使用固定 sft-py311 环境成功完成，commit
  `6f86f18`、物理 GPU1、greedy、4 条、无 optimizer/API/网络。严格指标：format/title/evidence/full
  success/oracle-path/fatal 分别为 `0.50/0.50/0.25/0.25/0.50/0.25`。
- 分题：rank3 `wit-00001777` 严格成功；rank2 `wit-00004467` 标题正确但输出了两句而非 gold 首句，且
  多查一次 rank3，严格失败；transient `wit-00014422` 在 retry 后查了 3 个 entity，耗尽 5 turns 而
  没有 final；no-match 在正确 retry/空结果路径后只输出 `NO_MATCH`，未遵守两行 final 格式。说明 v7
  确实暴露了证据边界、工具预算和格式三类非饱和错误，不是旧集合的全成功复刻。
- Base 报告 SHA256 `77bc29168d9caf318f507ea9869eb694189ead0603caf002d142aad40d03f380`；
  `exit_code=0`，stderr 无 traceback/OOM/NaN/Xid。GPU1 启动/结束均 18 MiB，最高轮询温度 56°C，cleanup
  `compute_processes=none`；GPU0/GPU5 未参与。下一步只运行同四题 SFT1 对照，不据 4 条样本声明提升。
- SFT1 对照 Run `wit-boundary-v7-sft1-dev4-20260822`，commit `36831be`，加载固定 v5 1-step adapter
  SHA256 `637169695b4b96022e003b2ad59bea780288da0c31aab94a1c64f962856399f5`。同四题的完整
  `results` 对象与 Base 逐字段相同，严格指标也完全相同；因此该 1-step SFT 对 v7 小样本没有可观察改善。
- 对两份新轨迹做 CPU evidence-fidelity-v2 诊断，Base/SFT1 mean 均为 `0.37003`；分题均为
  rank3 `1.0`、rank2 `0.48013`、transient `0`、no-match `0`。后两题虽取得 query-path 分，但格式硬门
  将总 reward 置 0，未把无 final/错误 final 当成成功。
- SFT1 报告 SHA256 `2689e63d4f176b36bc3972ee18494c6c5301af21bcffaa88b0bd2f3a453bd70e`；
  dataset/tasks hash 与 Base 一致，`exit_code=0`、无 traceback/OOM/NaN/Xid。GPU1 启动/结束 18 MiB，
  轮询最高 57°C，cleanup 无 compute process；GPU0/GPU5 未参与。下一步只允许 SFT1 的 v7 train
  stochastic rollout-only 方差门禁，仍不启动 optimizer。
- v7 rollout-only 协议在运行前冻结于 `configs/stochastic_rollout_boundary_v7.json`：train 四类按 task ID
  排序首条，依次为 `wit-00000885`、`wit-00001521`、`wit-00011482`、
  `boundary-no-match-train-000`；每题 4 条、seed `20260901`、`temperature=0.7/top_p=1.0`、
  evidence-fidelity-v2。dataset/tasks/adapter hash 已固定，批级门槛沿用预声明的
  variable-group `>=0.25`、format-valid `>=0.75`、fatal `<=0.25`。只用物理 GPU1、无网络/API/optimizer；
  不根据运行结果更换题目、reward 或门槛。
- v7 rollout-only Run `wit-boundary-v7-sft1-rollout-g4n4-20260822`，commit `00df34c`，16 条全部按
  冻结协议完成。四组 reward/方差依次为 rank3 `0.48545/0`、rank2 `0.95/0`、transient `0/0`、
  no-match `0/0`；所有 raw/fatal-clamped advantages 均为 0，不能形成任何 GRPO 梯度。
- 严格分布：rank3 0/4 full success、4/4 query-only partial；rank2 4/4 full success，但均多一次 lookup，
  故效率折扣到 `0.95`；transient 4/4 在 retry+3 lookup 后耗尽 turns 并 fatal；no-match 4/4 走对工具路径
  但输出格式错误。批级 variable-group fraction `0 < 0.25`、format-valid `0.50 < 0.75`、fatal
  `0.25 <= 0.25`，正式 gate 失败，受管 Run 以预期 `exit_code=4` 保留报告；未启动 optimizer。
- 报告 SHA256 `b0210b7af31457932e39c201c9be594d8de167e7c65f67cbe38ebdaedb14d259`；
  adapter/manifest/tasks hash 与冻结配置完全一致，stderr 无 traceback/OOM/NaN/Xid。GPU1 启动/结束
  18 MiB，轮询最高 60°C，cleanup 无 compute process；GPU0/GPU5 未参与，无网络/API。
- 决策：RL 继续阻断。现有 adapter 是在 v5 上只训 1 step，并未学习 v7 compact protocol；下一步回到
  晋级链的 v7 SFT 1-step smoke，再做同条件 held-out 和 rollout 门禁。不得通过提高 temperature、事后换题
  或取消 format 门来人为制造 advantage。
- SFT 启动器现支持受限 `--dataset-root`，仅接受 project4 processed 目录下的两个显式 profile：原 v5
  `wit_agentic_train_v1` 与 v7 `wit_agentic_train_v6`。v5 默认不变；v7 会校验固定 status/purpose、
  split/task counts、no-leak/turn/tool 合约，以及 tasks/dataset-info/train-SFT hashes。
- resume 新增同数据集 provenance hash 门：checkpoint 必须属于受管 Run、包含 trainer state/adapter，且
  原 Run 的 manifest hash 与本次完全一致，禁止把 v5 checkpoint 当成 v7 resume。真实 v7 manifest 验证、
  3 个启动器单测、Ruff 和 diff check 通过，CPU、无网络/GPU。
- 下一 Run 预定为 v7 SFT 1-step smoke：固定 Qwen3-VL-8B、LoRA rank8、vision/projector frozen、
  batch1、cutoff2048、seed42、80 条 train 数据入口，只执行 1 个 optimizer step 并保存完整 checkpoint。
  单独受管 tmux Run、仅物理 GPU1；不使用 GPU0/GPU5，不 resume v5，不联网，不覆盖既有结果。完成后先
  验证 checkpoint/梯度/loss/清理，再决定 held-out 对照；1 step 只证明链路，不声称质量提升。
- v7 SFT Run `wit-boundary-v7-sft-1step-20260822`，commit `8b4e404`，按预定只完成 1 step：loss
  `0.001019`、grad norm `0.1637`、lr `1e-4`、epoch `0.0125`，均为有限值；运行时约 3.2 秒。
- checkpoint-1 完整包含 adapter、trainer state、optimizer、scheduler、RNG、training args 和 tokenizer，
  global step=1，无 `.partial`。adapter/config/provenance SHA256 分别为
  `45cb6a5867e462f5499c8a74e8e34e70781f3b1b363463331b8bc72e240f339a`、
  `212e0be52f0185b4baadf7a40cc70376c5da54790815656ed92aa736fc0b18db`、
  `16733f18e5ee3eb9a19039c692dc7f3fb60df1a366ff053b8874f8f96500ca4b`。
- `exit_code=0`，无 traceback/OOM/NaN/Inf/Xid/segfault；GPU1 启动/结束 18 MiB，cleanup 无 compute
  process，GPU0/GPU5 未参与，无网络/API。1-step 只通过训练/保存/resume-ready 链路门；下一步先跑已固定
  的同四条 v7 dev greedy 对照，不直接 resume 扩大。
- v7 SFT1 held-out Run `wit-boundary-v7-sftv7-1step-dev4-20260822`，commit `00eac34`，同四题的完整
  results 对象与 Base 逐字段相同，指标仍为 format/title/evidence/full/oracle/fatal
  `0.50/0.50/0.25/0.25/0.50/0.25`；1 step 没有可观察行为变化。报告 SHA256
  `03de6a11e90ac44d78c20e4d5d7648b23bd48a5a916fbbb6a84c084c4684826e`，adapter/data/tasks hash
  均匹配；`exit_code=0`，GPU1 启动/结束 18 MiB、最高轮询 57°C、cleanup 无进程，GPU0/GPU5 未参与。
- 下一晋级固定为从 v7 checkpoint-1 resume 到总 step5（只新增 4 step），保持 dataset/hash、LoRA、batch、
  lr、seed、cutoff 和 GPU1 不变；resume 同数据集门必须通过。step5 后仍先跑同四题 held-out，若仍无变化
  或出现回归就停止扩大；不启动 RL，不把训练 loss 下降当作质量提升。
- v7 SFT resume Run `wit-boundary-v7-sft-resume-step5-20260822`，commit `7f38793`，同数据集 provenance
  门通过并从 step1 连续到 global step5；trainer state 明确含 step1–5。新增 step2–5 loss 为
  `0.01987/0.06864/0.08103/0.00005953`，grad norm `1.123/0.6083/0.581/0.008457`，均有限；
  总 train loss `0.03392`。这些只用于数值健康验收，不作质量结论。
- checkpoint-5 完整且无 `.partial`，adapter/config/provenance SHA256 分别为
  `fc8c922fd1e233fd30833dcb03454fe2c2afe3d31c32dd1f0ba4be655fb6ce97`、
  `63a38f2868b02489d9ffcb6188a0c947b20d2baf25512915d6ecea781abd9213`、
  `6f3472e0b1f29439d13db4b867cadf4d345d3c0c27a2e456b579c676b46fa9b4`。`exit_code=0`，无
  traceback/OOM/NaN/Inf/Xid；GPU1 启动/结束 18 MiB、cleanup 无 compute process，GPU0/GPU5 未参与。
  下一步只跑同四题 held-out，不继续训练。
- v7 SFT5 held-out Run `wit-boundary-v7-sftv7-step5-dev4-20260822`，commit `188e8d6`：rank3 保持
  严格成功，rank2 从 Base/SFT1 的“标题对但证据多一句”变为严格成功；transient/no-match 轨迹与失败
  保持不变。full/evidence exact 从 `0.25` 升到 `0.50`，但 format 仍 `0.50`、fatal `0.25`、oracle
  path `0.50`。evidence-v2 mean 从 Base `0.37003` 升到 `0.48750`，仅由 rank2 `0.48013→0.95` 驱动。
- 报告 SHA256 `f590b9941b0da9e796a7f26c268a37ffa76074376b8df1cefea5e50795e5836b`；adapter/data/tasks
  hash 匹配，`exit_code=0`、无异常。GPU1 启动/结束 18 MiB、最高轮询 56°C、cleanup 无进程；GPU0/GPU5
  未参与。该 4 条结果只支持“出现局部改善信号”，不足以声明 SFT 有效或重开 RL。
- 下一步预声明扩大评测而非训练：在完整固定 v7 dev20 上分别跑 Base 与 SFT5 greedy，同一 evaluator、
  max-new-tokens192、物理 GPU1；比较四类 strict/format/fatal/oracle 和 evidence-v2，不增训练 step、不运行
  rollout/optimizer。若总体无改善或恢复类回归则停止；若 held-out 改善稳定，再回到 rollout-only 门禁。
- 完整 Base Run `wit-boundary-v7-base-dev20-20260822`，commit `2b986d6`：format/title/evidence/full/
  oracle/fatal 为 `0.60/0.60/0.30/0.30/0.50/0.20`，evidence-v2 mean `0.43318`。rank2/rank3
  各 6 条且 full success 均 `0.50`，分层 v2 mean `0.68309/0.76084`；4 条 transient 全部
  maximum-turn fatal、4 条 no-match 全部格式失败，两类 full/v2 均为 0。
- Base dev20 报告 SHA256 `310a6cda8a850e531ff64fa6ff0018f99cc071c189f0c0478d0e2b696f89b7a0`；
  `exit_code=0`、无异常、cleanup 无 compute process；GPU1 轮询最高 63°C，GPU0/GPU5 未参与。现在只跑
  已预声明的 SFT5 dev20 对照，不改数据/参数/评分。
- 完整 SFT5 Run `wit-boundary-v7-sftv7-step5-dev20-20260822`，commit `357efb3`：format/title 保持
  `0.60/0.60`，evidence/full 从 Base `0.30/0.30` 升到 `0.55/0.55`，fatal/oracle 保持
  `0.20/0.50`；evidence-v2 mean 从 `0.43318` 升到 `0.55717`。
- 分层证据：rank3 full `0.50→1.00`、v2 `0.76084→1.00`；rank2 full `0.50→0.8333`、v2
  `0.68309→0.85725`。但 transient 仍 0/4 full、4/4 fatal、v2=0；no-match 仍 0/4 full、0/4
  format-valid、v2=0。SFT5 的收益是真实 held-out evidence 输出改善，但没有改善 retry 后的 turn-budget
  决策或 NO_MATCH 两行格式，不能宣称 agent 全面提升。
- SFT5 dev20 报告 SHA256 `be46dc0949ae6abd3e301fec96a094110927a7a6c35fcf4dcadc6c78acfa669a`；
  adapter/data/tasks hash 固定，`exit_code=0`、无 traceback/OOM/NaN/Xid。GPU1 启动/结束 18 MiB、轮询
  最高 65°C、cleanup 无 compute process；GPU0/GPU5 未参与，无网络/API。
- 当前停点：SFT 双阶段中的小规模 v7 step5 已出现可复核更新，但 RL 仍未满足 format/variance 门禁。
  下一合理动作是先做 recovery/no-match 定向且不泄漏 dev answer 的 train curriculum/采样审计，再考虑总
  step20；安全规范要求 20 step 及以上在运行前重新说明并获得用户确认。现在不启动 step20、rollout 重跑
  或 RL optimizer，等待用户审阅本次更新与下一阶段资源/停止条件。
- 用户要求额度有限下做最小尝试。新协议固定于 `configs/stochastic_rollout_boundary_v7_sft5.json`：只把
  adapter 换为已验证 v7 SFT5，其余沿用原四题、每题 4 条、相同 seed/temperature/top-p/reward/gate，
  总计 16 条 rollout-only。仅物理 GPU1、无网络/API/optimizer；不增到 8、不换题、不改 reward。若仍
  无题内方差或 gate 失败，本轮后立即停止，不自动扩大。
- 最小 SFT5 Run `wit-boundary-v7-sft5-rollout-g4n4-20260822`，commit `bc6c529`，16 条按固定协议完成。
  rank3 4/4 strict、reward 全 1；rank2 4/4 strict，其中 1 条用 oracle 两次 lookup 得 1，3 条多查一次
  得 0.95，population variance `0.00046875`，raw advantages 为
  `[+0.0375,-0.0125,-0.0125,-0.0125]`。新 SFT5 已从旧 SFT1 的全零 advantage 转为真实题内信号。
- transient 仍 4/4 maximum-turn fatal、no-match 仍 4/4 format invalid，两组 reward/variance 均 0。
  批级 variable group fraction `0.25` 达标且 `has_nonzero_advantage=true`，fatal `0.25` 达标；唯一未过项是
  format-valid `0.50 < 0.75`，所以正式 gate 仍失败并以预期 `exit_code=4` 保存，不启动 optimizer。
- 报告 SHA256 `b2249ae78bbb7b872f666531725b3edd99526c77440d06435fcc50a2a6983714`；adapter/data/tasks
  hash 匹配，无异常。GPU1 启动/结束 18 MiB、轮询最高 61°C、cleanup 无 compute process；GPU0/GPU5
  未参与，无网络/API。结论：4 rollouts 已足以发现方差，当前没有必要增到 8；瓶颈明确是 transient 与
  no-match 的格式/turn-budget，而不是纯粹 rollout 数量。本轮按用户要求停止，不做更多大规模改动。

### 2026-08-22：v8 官方式本地协议对齐（CPU 与数据发布）

- 源码对照结论：官方 agent 使用自然语言 `text_search(q, top_k)`、`<response>...</response>` 最终包裹、
  单动作循环和最多 50 次模型调用；无效动作与临时工具错误会以 observation 反馈重试。v7 并非只替换了
  搜索内容，还额外压缩为 `text_lookup(entity_id)`、两行裸 final 和 5 turn，导致 transient/no-match 的
  失败不能归因到 SFT 或 RL 方法本身。
- 变更：新增 `official-local-v1` 兼容协议。冻结 WIT/Wikipedia 后端保持不联网、无 API key，但模型改用
  `image_search(image=img_1)` 后的 `text_search(q, top_k)` 取回 `entity_id/title/source/summary`；最终回答
  必须置于 `<response>`，且为可审计的本地 benchmark 保留内部 `Title/Evidence` 字段。v7 的 legacy
  evaluator 与数据入口继续兼容，未被覆盖。
- 数据：从不可变 `wit-rl-boundary-v7` 非覆盖派生
  `/media/imc/data/yzy/agent/project4-opensearch-vl-rl/datasets/processed/wit-rl-protocol-v8`。保持 120 条、
  80/20/20、图像与 split 不变；将专家轨迹/Oracle 的 `text_lookup` 全部改为 `text_search`，最大动作数仅从
  5 放宽到 8（刻意不直接照搬官方 50）。v8 manifest 和 SFT launcher 的 SHA/profile 校验均通过。
- 验证：新增 CPU 合约测试，覆盖 `<response>` 强制门与 `text_search` 不接受 entity_id；既有 evaluator、
  SFT launcher、local-text-retrieval、fatal-aware reward 共 20 个相关测试通过，`py_compile` 与
  `git diff --check` 通过。对首条真实任务的冻结索引查询返回同一候选的 title/source/summary。
- 边界与下一步：本次未下载、未联网、未调用 API、未启动 GPU/SFT/RL，也未变更优化器或 reward 公式。
  后续只允许 v8 的 1-step LoRA SFT smoke（GPU1、受管 tmux、无 GPU0/GPU5、无网络/API），再做固定 dev20
  对照；通过后才重新冻结 4-rollout gate。任何达到 20 step、扩卡、GPU5 使用或 RL optimizer 更新仍需在
  执行前重新说明并获得用户确认。

### Step P4：v8 官方式本地协议 LoRA SFT 1-step smoke

- Run：`wit-protocol-v8-sft-1step-20260822`，代码 commit `f6d06f3`；启动前 GPU1 预检通过（24,066 MiB
  可用、47°C），数据盘可用 2,166 GiB。受管 tmux、独立进程组、代理净化、离线模型/数据均生效；GPU0/GPU5
  未加入 `CUDA_VISIBLE_DEVICES`，未联网、未使用 API。
- 配置：Qwen3-VL-8B 本地基座，v8 80 条 train、LoRA rank 8、视觉塔/projector 冻结、batch 1、cutoff 2048、
  BF16/FlashAttention-2、seed 42、max_steps=1。训练日志实际展示 `image_search` 后 `text_search(q, top_k)`
  与 `<response>` 监督 token，说明新协议进入训练，而非仅改 manifest。
- 结果：global step=1，loss `0.3231127`、grad norm `0.7615781`、learning rate `1e-4`、epoch `0.0125`，均为
  有限数；运行时 3.206 秒。checkpoint-1 包含 adapter、trainer state、optimizer、scheduler、RNG、tokenizer
  与 training args，且无 `.partial`；adapter/config/provenance SHA256 分别为
  `27249547f4c31af50fafc532c84bc080e5f806f82c2022203b6fee9e4571874a`、
  `04eff499f824714069369f8f79e165f2281ecd85b0735ea251a3e73d95f9261d`、
  `76387ffb995433c5ae0eb7204f92605f2c3398c2b29c12c5e846d1d472526267`。
- 安全验收：`exit_code=0`，stdout/stderr 无 traceback/OOM/NaN/Inf/Xid/segfault；GPU1 峰值轮询约
  21,239 MiB、50°C，cleanup 后仅 18 MiB 且 `compute_processes=none`。已仅关闭该 Run 的精确 tmux
  会话，未触及其他会话或进程。
- 声明边界：该 step 只证明新数据协议可被实际 SFT、保存和清理，不能证明质量提升。下一步是从同一固定
  v8 dev20 运行 Base 与该 SFT1 的 greedy 对照；在生成 checkpoint 之前不进入 RL，也不把 1-step loss
  当作效果证据。

### 2026-08-22：v8 Base 评测工具白名单泄漏（已精确停止并修复）

- 现象：首个 v8 Base dev20 Run 在第 1/20 条生成 `image_search → text_search → text_lookup`。虽然 prompt
  说使用 `text_search`，评测器当时仍把遗留 `text_lookup` 同时传给 chat template，违反“统一工具环境”要求。
  因此该 Run 的部分轨迹不构成 v8 对照，不能用于任何指标或训练判断。
- 处置：使用 `stop_managed.sh wit-protocol-v8-base-dev20-20260822` 校验 identity token 后只向该 Run
  的 process group 发送 TERM；Run 为预期 `exit_code=143`，cleanup 显示 GPU1 `compute_processes=none`。
  保留 Run、日志和已生成的第一条轨迹作为失败证据，未删除或覆盖；精确 tmux 会话随后关闭。
- 修复：评测器改为按 manifest `tool_protocol` 构建工具白名单。`official-local-v1` 只注册
  `image_search/text_search`，legacy 仅注册 `image_search/text_lookup`；执行器也 fail-closed 拒绝协议外调用。
  新增白名单回归测试，protocol/evaluator/reward 16 个相关 CPU 测试、`py_compile` 和 diff check 通过。
- 下一步：以新 Run ID 重跑完全相同的 v8 Base dev20 greedy 对照；停止的 Run 不重用、不改数据、reward、
  seed 或生成参数。未重新启动 SFT、RL 或 optimizer。

### 2026-08-24：官方工具协议透明本地 Provider

- 纠偏：用户确认允许搜索语料变化，但要求模型侧工具名称、参数和调用方式保持官方不变；本地数据集只替换
  执行后端。v7/v8 私有协议保留为历史证据，不再作为官方 SFT→RL 主链路。
- 官方契约：根据固定 vendor 源码冻结 `image_search(url)` 和
  `text_search(q/query, hl/lang, top_k)`；新增 `OfficialLocalSearchProvider`，不修改 vendor。图像结果仅向
  模型投影 title/source，文本结果保持官方 `[Passage]/Title/URL/Summary` 布局。
- 隔离：离线 image provider 只接受注册的 `img_N`，拒绝外部 URL、路径和非官方参数；模型 observation
  不含 entity ID、similarity、SQLite、文件路径、corpus/revision 或 provider 名称。backend 异常统一脱敏，
  不把异常详情回传模型。
- 验证：6 个新增 CPU 合约测试、既有 local text/image 与 v8 protocol 回归测试、Ruff、`py_compile` 和
  diff check 通过。真实 WIT replay 以 `image_search({"url":"img_1"})` 返回 3 个 title/source，再以
  `text_search({"q":"Genny Lim","hl":"en","top_k":3})` 返回本地 Wikipedia Passage；内部字段扫描为 0。
- 资源：本步无下载、无网络/API/GPU，未创建 Run 或 checkpoint。详细契约见
  `docs/OFFICIAL_LOCAL_PROVIDER_CONTRACT_2026-08-24.md`。下一步为官方 SFT-36K 的非代理下载与数据审计，
  在完整性门禁通过前不启动 SFT。

### 2026-08-24：官方 SFT wiki_en 下载并发事故与恢复门禁

- 事故：编排层已返回后台 session，但没有及时展示 session ID；操作者误判任务结束，对同一输出又启动了
  下载器。最多三个 JSON 下载器并发追加同一组 Range part，导致每个 JSON part 超过固定范围长度；图片
  下载只有一个进程，但处于未完成中断态。最终文件均未发布，size/SHA256 原子发布门禁避免了错误资产进入
  数据集。
- 精确处置：先按完整命令行和 PGID 确认四个下载进程，只对 PGID `912251`、`921075`、`933026`、
  `944419` 发送 TERM；未使用 `pkill`、`killall`、全局 Ray 或 tmux 清理。复核无残留进程后，不删除任何
  part，将 JSON 和图片分片分别改名为 `.parts.concurrent-corrupt-20260824` 与
  `.parts.interrupted-20260824` 保留证据。
- 修复：下载器新增按最终输出路径命名的 `flock(LOCK_EX|LOCK_NB)` 独占锁。第二个相同目标下载器现在会
  fail-closed，不会打开或追加 part；锁释放后的新任务可正常取得锁。新增回归测试覆盖“并发拒绝”和
  “释放后可重新取得”，项目固定 Python 环境测试、`py_compile` 和 Ruff 均通过。
- 恢复原则：只从新的空 `.parts` 目录下载固定 revision 的 `wiki_en` JSON 与 images.zip；继续禁用环境
  代理，不使用 Clash 7890/7891；只有固定字节数和官方 SHA256 同时通过才发布。该步骤没有使用 GPU、
  API 或 tmux；GPU0/GPU5 均未参与。

### 2026-08-24：官方 wiki_en 审计与固定 SFT-1000 发布

- 下载恢复：固定 revision `2c1c460...` 的 JSON（131,910,169 B）和 images.zip（104,683,505 B）
  均通过官方 SHA256；下载器显式忽略代理环境，未使用 Clash 7890/7891。ZIP 安全审计通过并非覆盖解压，
  4,084 个文件、105,121,529 B 解压大小、最大压缩比约 10.286。
- 全量审计：3,503 条、4,084 个唯一图片引用，工具声明完全一致。发现唯一不完整源样本索引 1900 以
  observation 结束；官方源哈希正确。未伪造或补写答案，明确记录后从候选池排除，剩余 3,502 条。
- 发布：固定 seed `opensearch-vl-official-sft-v1` 对源索引和完整内容做哈希排序，非覆盖发布 1,000 条、
  1,155 张复制图片、约 67 MiB；精确索引与源/输出哈希写入数据盘 manifest。数据 JSON SHA256 为
  `af5eb4adc2a9e4fcc0529ed2c6cfc523fca3753740aec1178c57acd54b4a3dd7`。
- 工具覆盖：子集含 image_search 982、text_search 1,318、crop 115、layout_parsing 97、
  super_resolution 33、sharpen 7 次调用；不改官方 system、工具声明、observation 或答案。详细证据见
  `docs/OFFICIAL_SFT_DATA_STATUS_2026-08-24.md`。
- 资源：本步为 CPU/网络/磁盘操作，无 API/GPU/tmux；GPU0/GPU5 均未参与。下一步先做 LLaMA Factory
  数据加载门禁，再用新 Run ID、受管 tmux 和物理 GPU1 执行 SFT 1-step。

### 2026-08-24：官方 SFT-1000 单卡 LoRA launcher 门禁

- 新增 `scripts/run_official_sft.py`，只接受项目四受管 Run、单张稳定物理卡，并硬拒绝 GPU0/GPU5 和
  多卡选择；训练步数限定 1..50，输出目录拒绝覆盖，resume 只接受项目四 Run 内含 trainer state、adapter
  和匹配 provenance/profile 的 checkpoint。
- 数据门禁固定官方 revision、源 JSON、子集 JSON、dataset_info 与 selection indices SHA256，并确认样本
  数 1,000、排除源索引 1900、图片 payload 完整；本机 LLaMA Factory `0.9.5.dev0` 可见。
- 资源适配：保持官方 Qwen3-VL template 和完整监督轨迹，但因单卡 24GB 将官方 full fine-tune 改为 LoRA
  rank 8，冻结视觉塔/projector，cutoff 从 32K 降为 2K，图像像素上限 65,536；所有偏差写入每个 Run 的
  provenance。该配置验证的是可训练闭环，不宣称等价于官方 128-GPU full SFT。
- CPU policy tests 覆盖受管 Run、GPU0/GPU5/多卡拒绝、LoRA/冻结/非覆盖/save-step 配置；真实 manifest
  校验与 `llamafactory-cli version` 通过。本步未使用 GPU；下一步提交后才启动 1-step。

### 2026-08-24：官方 SFT 首次 1-step 在 optimizer 前 fail-closed

- Run `official-sft-wiki-en-1step-20260824` 按规定由命名 tmux/受管脚本仅使用物理 GPU1；数据转换和
  1,000 条 tokenizer 处理成功，模型加载成功，但首个 forward 报 image features=77、image tokens=0，
  未执行 optimizer step、未生成 checkpoint。
- 根因：2K cutoff 被官方长 system prompt 与完整工具声明占满，在用户图像占位符之前截断。CPU 对照
  2,048/3,072/4,096 均为零 image token；5,120 起恢复。全 1,000 条以 5,120 审计为零缺图、零无监督
  label，image token 64～245；997 条仍截断，明确属于工程闭环而非完整轨迹复现。
- 安全：Run `exit_code=1`；GPU1 before/after 均 18 MiB、46°C、cleanup 无 compute process；GPU0/GPU5
  未参与。失败 Run 全量保留。launcher 将 cutoff 固定为 5,120，下一次仍仅做新 Run ID 的 1-step；OOM
  或占位符问题立即停止，不自动扩卡或改数据。
