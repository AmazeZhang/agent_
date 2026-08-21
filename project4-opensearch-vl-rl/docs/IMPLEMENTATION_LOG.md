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
| RL 环境/训练 | 已完成 API/资源门禁和 RL-8K 元数据审计；正在准备本地优先多模态检索，尚未启动 RL |
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
| P4 Agentic SFT | 工程 smoke 完成 | 推理闭环通过 | 合成数据 1→2→5 step、断点续训、adapter 离线推理通过 |
| P5 SFT→RL rollout-only | 本地检索准备中 | SFT checkpoint 通过固定对照 | RL-8K 元数据已审计；OVEN/WIT 适用范围已收窄，未启动 RL |
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
