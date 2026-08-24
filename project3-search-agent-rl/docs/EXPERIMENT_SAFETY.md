# Project 3实验安全与AI交接强约束

本文是项目三所有实验的强制安全规范，适用于人类操作者和后续AI。任何运行、恢复、停止、
下载、清理或扩容操作前必须完整阅读。若任务与本文冲突，必须停止并向用户说明，不能自行放宽。

## 1. 当前事实与最高优先级

- 项目已在veRL框架下完成Qwen2.5-3B全参数FSDP六卡的GRPO/GiGPO工程复现，以及
  Search-aware v2两组seed、10-step训练和固定confirm256 held-out配对评测；
- 最终证据支持工程复现和搜索/作答行为改善，不支持Search-aware v2的EM稳定提升：
  seed1234净−1题、seed2026净+1题，两次精确McNemar均`p=1.0`；
- “训练能跑、权重变化、退出码0”仍不等于质量提升或论文官方完整规模复现；
- 物理GPU 0永久保留给Linux图形界面，任何实验都不得使用；
- 物理GPU 5有历史不稳定记录，默认排除；
- 最近正式训练使用物理GPU`1,2,3,4,6,7`，评测使用GPU1；这不代表这些卡永远空闲，
  任何新任务启动前必须重新检查；
- 数据盘挂载点为`/media/imc/data`，项目专用目录为
  `/media/imc/data/project3-search-agent-rl/`。不得格式化、重新挂载或清理整块盘。

## 2. 绝对禁止事项

1. 禁止把物理GPU0加入`CUDA_VISIBLE_DEVICES`、Ray资源或任何训练/推理/Retriever任务；
2. 禁止直接执行上游8卡脚本，禁止默认占满所有GPU；
3. 禁止使用`pkill python`、`pkill ray`、`killall`、模糊进程名匹配、无范围的
   `ray stop --force`，也禁止结束未知PID；
4. 禁止删除或覆盖已有Run、Checkpoint、Rollout、日志、数据集、模型和Retriever索引；
5. 禁止对`/media/imc/data`、仓库根目录或含未知内容的目录执行递归删除/清理；
6. 禁止自动格式化、分区、重新挂载磁盘，或修改系统CUDA/显卡驱动；
7. 禁止静默升级CUDA、PyTorch、Ray、vLLM、FlashAttention、veRL或模型版本；
8. 禁止把模型、Corpus、Index、Checkpoint、运行日志大文件、Token或密钥提交到Git；
9. 禁止因为额度、时间或显存压力跳过预检、缩短安全退出流程或编造实验结论；
10. 禁止把Ray daemon正常关闭的`EXPECTED_TERMINATION`与训练Actor异常退出混为一谈。

任何物质性删除、覆盖、版本升级、GPU5启用、20步以上训练、全量数据训练或多卡扩容，均需
先向用户说明目标、资源、风险和回滚方式并获得明确同意。

## 3. GPU与多卡规则

### 3.1 物理卡白名单

默认候选稳定卡为物理GPU `1,2,3,4,6,7`，但“候选”不等于“当前可用”。每次启动前都要：

```bash
nvidia-smi
bash scripts/preflight.sh 1
```

目标卡存在任何Compute Process时必须拒绝启动；不得抢占或结束它。还需检查显存、温度、
掉卡/Xid记录和其他用户任务。GPU5只有在用户明确批准、有人值守并设置
`ALLOW_UNSTABLE_GPU5=1`时才可临时使用。

### 3.2 逻辑卡映射

例如`CUDA_VISIBLE_DEVICES=1,2`时：

```text
程序 cuda:0 -> 物理 GPU1
程序 cuda:1 -> 物理 GPU2
```

所以日志里的`cuda:0`不能单独证明误用了物理GPU0；必须结合Run的`metadata.env`中
`physical_gpu_ids`和启动环境判断。反过来，也不能看到逻辑`cuda:0`就放松物理卡检查。

### 3.3 多卡晋级

veRL支持多卡，但单卡能跑时不应仅因为“框架支持”就扩容。多卡前必须完成：

- 明确列出物理卡白名单，仍排除GPU0和默认排除GPU5；
- 逐卡空闲/所有者检查与显存预算；
- 核对`trainer.n_gpus_per_node`、Ray资源、FSDP、rollout tensor parallel和LoRA配置；
- 确认NCCL/CUDA/PyTorch/vLLM版本兼容；
- 使用新Run ID先跑最小Smoke，再验证多卡Checkpoint恢复；
- 退出后逐卡确认显存、PID和全部Ray Actor释放。

一次只扩大一个变量：模型大小、上下文、batch、rollout并发、训练步数、数据规模和GPU数不能
同时变化，否则出现OOM、挂死或效果退化时无法归因。

## 4. 存储与证据保护

启动前必须设置并确认：

```bash
export PROJECT3_DATA_ROOT=/media/imc/data
df -h /media/imc/data
```

受管脚本默认要求至少150GiB可用空间。所有数据、模型、Index、Run和Checkpoint只写入
`${PROJECT3_DATA_ROOT}/project3-search-agent-rl/`下对应子目录，不触碰数据盘上的其他项目。

每个实验使用全新、可追踪的Run ID。`run_managed.sh`会拒绝已有Run目录；不得绕过这一保护。
普通和audit Rollout采用`.partial`、flush/fsync、原子rename并拒绝覆盖。失败Run同样是证据，
必须保留。磁盘不足时先暂停并列出占用，未经用户确认不得删除旧Checkpoint或数据。

## 5. 固定软件基线

- veRL vendor提交固定为`20bd331bdbc9026a5668e11362178e10ab7400c8`；
- 项目环境位于数据盘的`envs/searchr1-repro-cu124`；
- 上游修改通过`patches/`中的独立补丁重放，不直接提交vendor脏状态；
- 启动脚本会校验veRL提交和必需补丁；校验失败应修复环境或解释差异，不能注释掉门禁；
- 修改环境前保存`pip freeze`、CUDA/驱动/PyTorch/Ray/vLLM版本和变更理由；
- 禁止在实验中混用系统Python、用户site-packages或另一个项目的环境。

## 6. 标准启动流程

### 6.1 只读预检

先检查Git状态、磁盘、GPU、Retriever端口、残留进程、Run ID是否已存在、配置和恢复源。
不得把未知的dirty worktree、未知PID或已有Run目录当成“可以直接覆盖”。

Retriever端口已ready时，不能仅凭`/health`推断“本轮Retriever启动成功”。启动新服务前必须
先用`ss -ltnp`解析listener PID，并记录完整`/proc/<pid>/cmdline`、owner、start time、PPID和
资源/并发配置。若是既有且配置匹配的共享服务，可明确记录后复用；若身份或配置不明则停止并
询问用户。只有确认端口未占用时才允许创建新的Retriever tmux；本轮未创建的既有服务不得在
收尾时被停止。

```bash
export PROJECT3_DATA_ROOT=/media/imc/data
bash scripts/preflight.sh 1
```

### 6.2 tmux与受管运行

训练必须在命名tmux会话中调用`run_managed.sh`，不能直接裸跑Python：

```bash
tmux new-session -d -s project3-<stage>-<run-id> \
  "cd /home/imc/yzy/agent/project3-search-agent-rl && \
   export PROJECT3_DATA_ROOT=/media/imc/data && \
   bash scripts/run_managed.sh <new-run-id> 1 -- <exact-command>"
```

交给用户的查看方式：

```bash
tmux list-sessions
tmux attach -t project3-<stage>-<run-id>
# 从attach中安全离开：Ctrl-b，然后按d
tail -f /media/imc/data/project3-search-agent-rl/runs/<new-run-id>/stdout.log
```

tmux显示`Pane is dead (status 0)`表示命令正常结束，并不表示卡死；最终仍须核对`metadata.env`
中的`exit_code`、日志、Checkpoint和资源清理。tmux没有输出时优先`tail -f stdout.log`，不要因
“看起来没动”就重复启动第二个实验。

## 7. 精确停止与清理

正常结束优先让训练自己的shutdown流程运行。需要从另一个终端停止受管Run时：

```bash
export PROJECT3_DATA_ROOT=/media/imc/data
bash scripts/stop_managed.sh <exact-run-id>
```

该脚本核验Run身份Token，只向对应进程组发送TERM，30秒后才对同一组发送KILL。Retriever若在
独立tmux中，使用精确会话名发送Ctrl-C：

```bash
tmux send-keys -t <exact-retriever-session> C-c
```

不得用全局命令“顺手清理”。如果退出后仍占显存，先只读收集以下证据：

```bash
nvidia-smi
ps -fp <pid>
tr '\0' ' ' </proc/<pid>/cmdline
pstree -aps <pid>
```

只有确认PID属于当前Run、用户一致、命令行与父子关系一致后，才能通过受管脚本或精确进程组
处理。若身份不明，立即停手并询问用户。不得杀Xorg、GNOME、桌面服务或其他用户进程。

## 8. 每次Run后的强制验收

必须同时检查，不能只看tmux退出码：

1. `metadata.env`的start/end、物理GPU和`exit_code`；
2. stdout/stderr中OOM、NaN/Inf、traceback、segfault和unexpected failure；
3. 预期Checkpoint结构、最新step、模型/Optimizer/Extra/Data State和无`.partial`；
4. 恢复实验中Optimizer、Scheduler和参数确实连续推进，不是重新初始化或重复保存；
5. Rollout/audit数量、raw score、Prompt loss mask和typed retrieval failure；
6. Actor日志无`SYSTEM_ERROR`、`RAY_WORKER_FAILURE`、异常SIGTERM；Ray daemon的
   `EXPECTED_TERMINATION`单独记录；
7. 精确Run PID、Ray进程、Retriever PID/端口均释放；
8. 所有目标物理GPU回到基线显存，GPU0仍只有桌面进程；
9. 记录问题原因、修复、未解决风险和SHA256，更新执行/进度文档后再晋级。

## 9. 实验晋级与声明边界

按`Inference → 1 Step → Resume → 5 Step → held-out evaluation → 20 Step`逐级进行。每级都要
确认无OOM/NaN、Retriever错误可区分、Checkpoint可恢复、进程完全退出、显存恢复。20步以上、
全量数据、3B/7B模型或多卡需要重新估算时间、显存、磁盘和故障影响，并获得用户批准。

当前只允许如下声明：

- veRL下Search-R1工程链路、真实Wiki Retriever、Qwen2.5-3B全参数FSDP、GRPO与GiGPO
  对照、Checkpoint合并和固定held-out评测已跑通；
- Search-aware v2在两组seed上稳定提高搜索率和search-to-answer，并在seed2026降低
  max-steps耗尽与真实冗余搜索；
- 反事实检索评测支持Aware-v2的多数搜索正确答案依赖真实Retriever证据；
- Checkpoint、Loss Mask、检索元数据、逐题统计、训练曲线和Actor生命周期有证据支持。

当前禁止如下声明：

- Search-aware v2稳定提高EM、已经收敛或达到SOTA；
- 已完成论文级/官方Search-R1完整规模效果复现；
- 参数变化、非零梯度或训练reward等同held-out性能；
- 四轮交互上限已经修复。

当前实验已收尾，无需自动扩大训练。任何后续实验必须先预注册单变量假设；四轮上限优先做
evaluation-only协议检查，再决定是否进行新的小步训练。

## 10. 交接时的最短检查清单

```text
[ ] 已完整阅读本文和根目录AGENTS.md
[ ] 没有使用物理GPU0；GPU5未被默认启用
[ ] 目标物理GPU已重新检查且无未知Compute Process
[ ] 数据根为/media/imc/data，空间足够，没有覆盖已有Run
[ ] veRL提交、环境和补丁门禁一致，没有静默升级
[ ] 使用新Run ID、tmux和scripts/run_managed.sh
[ ] 停止只针对精确Run/会话，不使用全局kill/ray stop
[ ] 退出后核对PID、端口、Ray Actor、逐卡显存和Checkpoint完整性
[ ] 失败证据保留，任何删除/覆盖先征得用户同意
[ ] 实验结论没有超过held-out、baseline和多seed证据范围
```
