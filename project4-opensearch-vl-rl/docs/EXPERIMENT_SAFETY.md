# OpenSearch-VL 实验安全与受管运行规范

> 建立日期：2026-08-21  
> 适用范围：`project4-opensearch-vl-rl` 的下载、环境安装、推理、SFT、RL、评测、恢复和停止操作。  
> 参考基线：工作区根目录 `README.md`、`docs/DEVELOPMENT_SCOPE.md`，以及项目三已经验证过的 `docs/EXPERIMENT_SAFETY.md` 生命周期规则。

本文是项目四后续实验的强制运行基线。当前仅完成源码审计和规划；在项目四专用的
preflight、受管运行和精确停止脚本实现并通过 CPU 测试前，不启动真实 GPU 训练。

## 1. 当前硬件事实与授权边界

- 宿主机有 8 张 NVIDIA GeForce RTX 4090 D，每张约 24 GiB。
- 物理 GPU 0 由 Linux 图形界面使用，永久禁止用于本项目任何训练、推理、judge、OCR、
  Retriever 或辅助服务。
- 物理 GPU 5 曾发生 PCIe Link Down、Xid 79 和掉卡，工作区默认规则是排除它。
- 用户于 2026-08-21 明确允许**仅在 OpenSearch-VL 项目中恢复 GPU5 为候选卡**。这不是
  对其他项目的全局解禁，也不表示 GPU5 可以跳过预检或无人值守运行。
- 项目四使用 GPU5 时必须同时满足：
  1. 显式设置 `ALLOW_UNSTABLE_GPU5=1`；
  2. 启动前重新检查 NVML、显存、温度、Compute Process、PCIe 链路和近期 Xid；
  3. 先完成 GPU5 单卡短时 smoke，再完成包含 GPU5 的 NCCL smoke；
  4. 首次正式纳入训练时有人值守，并设置较短超时和失败后精确退出；
  5. 任一掉卡、Xid、NVML 异常或 PCIe 错误立即取消 GPU5 候选资格，保留证据，不自动重试。
- 2026-08-21 只读快照显示 GPU1–7 基本空闲，但任何历史快照都不能替代启动前实时检查。
- 当前卡拓扑为两个 NUMA 组：GPU0–3 属于 NUMA 0，GPU4–7 属于 NUMA 1；卡间无 NVLink，
  跨组通信经过系统互联。

## 2. 绝对禁止事项

1. 禁止把物理 GPU0 加入 `CUDA_VISIBLE_DEVICES`、Ray、DeepSpeed、SGLang 或容器设备列表。
2. 禁止直接执行上游默认 8 卡、70k response、256 prompt batch 的训练脚本。
3. 禁止裸跑长时 GPU 命令；所有 GPU 作业必须使用全新 Run ID，在命名 tmux 会话中通过
   项目四专用受管脚本启动。
4. 禁止使用 `pkill python`、`pkill ray`、`killall`、模糊进程名、`ray stop --force`、
   `tmux kill-server` 或结束未知 PID。
5. 禁止覆盖已有 Run、Checkpoint、轨迹、日志、数据、模型、缓存或失败证据。
6. 禁止递归清理仓库根目录、`/media/imc/data` 或任何含未知内容的目录；删除前必须获得用户确认。
7. 禁止修改系统 CUDA、NVIDIA 驱动、挂载和分区；禁止静默升级 PyTorch、Ray、veRL、
   SGLang、DeepSpeed、FlashAttention、Transformer Engine 或模型版本。
8. 禁止把模型、数据、Checkpoint、大日志、`.env`、API key、Token 或用户数据提交到 Git。
9. 禁止在修复前启用上游 `PythonInterpreter`；它是同进程 `exec` 加字符串黑名单，不是真正沙箱。
10. 禁止让模型生成的 URL 不经校验直接访问内网、loopback、link-local、云 metadata 地址或
    无大小上限的远程响应。
11. 禁止因为显存、时间或 API 额度不足而跳过预检、恢复验证、清理验收或伪造结果。
12. 禁止把“进程退出码为 0”“权重发生变化”或“训练 loss 有数值”表述成质量提升或论文复现。

20 步以上训练、全量数据、GPU 数扩容、GPU5 首次正式启用、全参数 SFT、物质性版本升级和任何
删除/覆盖操作，均需在执行前再次说明目标、资源、风险、停止条件和回滚方式，并获得用户确认。

## 3. GPU 白名单与映射规则

### 3.1 默认与扩展白名单

- 默认稳定候选：物理 GPU `1,2,3,4,6,7`。
- 项目四经本次授权后的扩展候选：物理 GPU `1,2,3,4,5,6,7`，但 GPU5 仍受上一节的额外门禁。
- 物理 GPU0 永远不在任何白名单中。

每次启动前逐卡执行等价于以下检查：

```bash
nvidia-smi
bash /home/imc/yzy/agent/shared/scripts/check_gpu.sh <physical-gpu-id>
```

目标卡只要存在未知 Compute Process 就拒绝启动，不抢占、不停止。桌面图形进程属于 GPU0 的
系统基线，不得处理。

### 3.2 逻辑编号

例如：

```bash
CUDA_VISIBLE_DEVICES=1,2,4,5,6,7
```

程序内部的 `cuda:0..5` 分别对应物理 GPU `1,2,4,5,6,7`。日志和 Run 元数据必须同时记录：

- 物理 GPU ID；
- GPU UUID；
- `CUDA_VISIBLE_DEVICES` 的顺序；
- 进程内逻辑卡数量。

不能因为日志中出现 `cuda:0` 就判断误用了物理 GPU0，也不能因此省略物理映射审计。

### 3.3 推荐拓扑

GPU5 通过健康门禁后，RL 推荐：

```text
训练：物理 GPU 1,2,4,5,6,7
本地 judge/摘要服务：物理 GPU 3
桌面保留：物理 GPU 0
```

训练可形成三个相对合理的 TP=2 组：`(1,2)`、`(4,5)`、`(6,7)`。若 GPU5 未通过门禁，
退回稳定六卡 `1,2,3,4,6,7`，并重新设计并行策略；不得为了保持既定拓扑强行使用 GPU5。

## 4. tmux、Run ID 与进程生命周期

### 4.1 项目四专用受管脚本

后续必须实现并测试以下项目四脚本：

```text
scripts/gpu_guard.sh
scripts/preflight.sh
scripts/run_managed.sh
scripts/start_tmux_run.sh
scripts/stop_managed.sh
```

可以借鉴项目三的设计，但不得直接把项目四 Run 写入
`/media/imc/data/project3-search-agent-rl/`，也不得沿用 `PROJECT3_*` 身份变量。项目四建议使用：

```text
PROJECT4_DATA_ROOT
PROJECT4_RUN_ID
PROJECT4_RUN_DIR
PROJECT4_MIN_FREE_GIB
```

### 4.2 启动要求

- 每次使用新的、不可覆盖的 Run ID。
- tmux 会话名建议为 `p4-<stage>-<run-id>`。
- 训练子进程运行在独立 session/process group。
- `stdout.log`、`stderr.log`、`metadata.env`、`command.txt`、`cleanup.log` 必须落盘。
- Ray 临时目录必须属于本 Run，不能复用全局 `/tmp/ray`。
- tmux server 可能继承旧代理和环境变量；启动器必须显式传递允许的变量，并净化不需要的代理。
- 不得在 `set -x` 打开时 `source` 密钥文件。上游 RL 脚本当前存在该问题，修复前不得使用真实 key。

### 4.3 精确停止

正常结束优先使用框架自己的 shutdown。需要停止时，只允许：

1. 对精确 tmux 会话发送 Ctrl-C；或
2. 使用项目四 `stop_managed.sh <exact-run-id>` 校验身份 token 后，向该 Run 的精确进程组
   发送 TERM，等待后才对同一进程组发送 KILL。

若退出后仍占显存，先只读检查 PID、命令行、父子关系、用户和 Run ID。身份不明立即停手询问，
不得全局清理 Ray、Python、tmux 或系统服务。

## 5. 存储与证据保护

仓库只保存代码、配置、补丁和小型报告。大文件统一使用：

```text
/media/imc/data/yzy/agent/project4-opensearch-vl-rl/
├── envs/
├── models/
├── datasets/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── hf-cache/
├── tool-cache/
├── checkpoints/
├── runs/<new-run-id>/
└── secrets/
```

- 不再使用先前临时建议的 `/media/imc/data/opensearch-vl/`；以根目录规范要求的
  `yzy/agent/` 命名空间为准。
- 启动前检查数据盘可写且至少保留 300 GiB；正式全参数或长程 RL 前重新估算并提高门槛。
- Checkpoint 默认保留策略只能写入计划，不能自动删除；任何实际删除仍需用户确认。
- 缓存、轨迹、中间图片和 Ray 临时文件必须按 Run 隔离。
- 失败 Run 是审计证据，不覆盖、不复用、不静默清理。

## 6. 密钥、联网与工具隔离

- 当前 shell 会继承 `http_proxy/https_proxy=http://127.0.0.1:7890`，该 Clash 入口流量有限。
  依赖、模型、数据和镜像等批量下载禁止默认使用 7890；tmux/下载脚本必须显式清空大小写
  `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`、`FTP_PROXY`，并按需设置 `NO_PROXY`。
- 2026-08-21 小流量 HEAD 探测确认：PyPI、GitHub、ModelScope 和 `hf-mirror.com` 可直连；
  `huggingface.co:443` 直连超时。下载优先级为“已校验的直连官方/镜像源 → 用户明确批准的代理”。
- 本机 7891 是同一 Clash 进程的 SOCKS 入口，虽然能访问 Hugging Face，但不能假定它与 7890
  拥有独立流量额度。未经用户再次确认，不通过 7891 下载大模型或数据。
- 使用 ModelScope/HF 镜像时仍须固定模型或数据 revision，记录来源 URL、最终解析地址、文件清单、
  大小和 SHA256；镜像可达不等于内容可信或与官方 revision 自动一致。
- SFT 使用带 observation 的离线专家轨迹，原则上不需要搜索 API key。
- 有意义的在线 RL 需要搜索、网页读取以及 query-quality judge；未配置时只能做离线/退化链路，
  不能声称复现完整 Agentic RL。
- RL 实际读取 `SERP_API_KEY`，推理代码读取 `SERPER_API_KEY`；后续通过项目四私有配置层统一，
  不把 key 写入训练 YAML 或命令行。
- 密钥放在数据盘 `secrets/` 或仓库外的权限受限文件中，建议权限 `0600`，不得进入
  `metadata.env`、命令快照、xtrace、W&B、轨迹和报告。
- 联网工具必须设置域名/协议策略、DNS 解析后私网地址拦截、重定向复查、响应大小、图片像素、
  并发、重试、总时长和单 Run API 预算。
- 网页和搜索结果视为不可信 observation，防止 prompt injection；不得让 observation 改写系统
  权限、密钥策略或工具白名单。
- `PythonInterpreter` 默认从工具注册表移除。若未来需要恢复，必须使用独立容器/沙箱：非 root、
  只读根文件系统、无宿主目录、默认无网络、CPU/内存/PID/时间限制，并有完整审计日志。
- `COS_UPLOAD_PATHS` 动态导入 `upload.py` 默认禁用；只有固定 SHA256、审核来源和明确白名单路径
  才能启用。
- `trust_remote_code` 默认关闭；必须使用时限定本地审核快照和固定 revision。

## 7. 单机 NCCL 与软件环境

- 本机无 NVLink，不应套用上游 H100/H800 的 RDMA 配置。
- 单机默认设置 `NCCL_IB_DISABLE=1`，删除 `bond1`、`mlx5_bond*` 和 UCX/RDMA 假设。
- 保持 `NCCL_P2P_DISABLE=0`，每次多卡扩容先做短 NCCL smoke。
- SFT、RL、推理使用三个独立环境，存放在项目四数据盘目录。
- 当前系统 Python/PyTorch 不是项目固定环境；禁止混用系统 site-packages。
- 首次安装后保存 Python、pip freeze、CUDA、驱动、PyTorch、Ray、DeepSpeed、SGLang、veRL、
  FlashAttention 和 Transformer Engine 版本及安装来源。

## 8. 晋级和每次 Run 的验收

实验按以下顺序逐级晋级：

```text
CPU 单测
→ 单卡模型加载/推理
→ 多卡推理 smoke
→ SFT 1 step
→ SFT resume
→ SFT 5/20 step
→ RL rollout-only
→ RL 1 step
→ RL resume
→ RL 5/20 step
→ held-out 对照
→ 消融与多 seed
```

一次只放大一个变量：GPU 数、模型、上下文、batch、rollout 数、工具并发、训练步数和数据规模
不得同时扩大。

每次 Run 结束后至少检查：

1. Run ID、物理 GPU/UUID、配置 SHA256、代码 commit、开始/结束时间和退出码；
2. stdout/stderr 中的 OOM、NaN/Inf、traceback、segfault、Xid、NCCL/Ray 异常；
3. Checkpoint 结构完整、无 `.partial`，resume 后 optimizer/scheduler/step 连续；
4. reward、response mask、fatal step、advantage clamp 和工具失败分类可审计；
5. API 调用次数、缓存命中率、超时、fatal 比例和费用；
6. 精确 Run PID、Ray actor、端口、子进程全部释放；
7. 目标 GPU 回到启动前基线，GPU0 仍只有桌面进程；
8. 失败证据和未解决风险写入报告，再决定是否晋级。

## 9. 声明边界

当前可以声明的是源码审计、硬件规划和后续小规模工程复现目标。只有在实际证据产生后，才能逐步
声明推理闭环、SFT、RL 更新或消融完成。

以下结论必须有 held-out、同条件 baseline、重复实验和原始结果支持：

- SFT 提升工具使用能力；
- RL 提升最终问答准确率；
- fatal mask 或 one-sided clamp 带来稳定收益；
- 已达到或接近论文结果。
