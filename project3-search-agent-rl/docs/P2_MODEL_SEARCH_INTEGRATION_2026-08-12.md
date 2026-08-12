# P2：Qwen2.5-1.5B模型与Search-R1环境集成报告

- 执行日期：2026-08-12
- 阶段结论：通过R1功能复现门槛；未开始训练
- 上游框架：`verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`
- 模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 模型revision：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- 执行GPU：仅物理GPU1；GPU0未暴露，GPU5未使用

## 1. 本阶段回答的问题

P2不是评测Search-R1论文分数，而是确认以下真实链路在本机可执行：

```text
Qwen生成原始文本
  -> verl-agent search_projection
  -> SearchEnvironmentManager
  -> SearchEnv
  -> localhost /retrieve
  -> <information>历史回填
  -> 下一轮模型动作
  -> Exact Match Reward
```

驱动脚本直接复用上游`SearchEnvironmentManager`、`SearchMultiProcessEnv`、
`search_projection`和`SearchEnv`，提示词也由上游管理器生成，并使用与上游Rollout相同的
`tokenizer.apply_chat_template(..., add_generation_prompt=True)`方式。P2使用Transformers做冻结推理，
没有把它伪装成veRL/vLLM训练；veRL真实参数更新属于P3。

## 2. 模型快照与供应链记录

模型许可证为Apache-2.0，下载至：

```text
/media/imc/data/project3-search-agent-rl/models/Qwen2.5-1.5B-Instruct
```

Git中的完整文件大小和SHA256清单位于：
`configs/model_manifests/qwen2.5-1.5b-instruct.json`。

关键权重：

```text
model.safetensors
bytes: 3087467144
sha256: dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
```

该SHA与模型仓库LFS对象SHA一致。执行后data盘约3.1 TiB可用。

## 3. 实验边界和诚实口径

本阶段的Retriever是P1构建的Ground-Truth衍生Fixture。它只用于检验HTTP协议、搜索工具、
多轮历史、动作投影、Reward和故障观测，严禁用于：

- 报告模型准确率或Search-R1基准分数；
- 与论文、其他模型或训练方法比较；
- 训练模型；
- 声称模型具备真实Wikipedia检索能力。

因此下面的`reward_one`仅是集成证据，不是质量指标。进入P3前必须换成与答案无关的真实
Corpus Retriever。

## 4. 非破坏性执行方式

每次运行均通过`scripts/run_managed.sh`：

- `CUDA_VISIBLE_DEVICES=1`，模型进程内只看到一个逻辑GPU；
- 启动前拒绝已有计算进程的GPU；
- 每个Run使用独立且不可覆盖的目录；
- 子进程位于独立Session/Process Group；
- 异常时只向该Run的进程组发送TERM，30秒后才定向KILL；
- 不使用`pkill`或全局`ray stop`；
- localhost Retriever运行在线程中，随同一Python进程关闭；
- 所有Run退出码均为0，cleanup均报告GPU1无计算进程。

训练阶段仍必须使用tmux；P2是秒级受管冻结推理，未启动训练或Ray集群。

## 5. 逐级门禁结果

### 5.1 自然策略1条

Run：`p2-qwen15b-fixture-n1-20260812`

- 1/1正常终止，1/1动作格式有效；
- 0次搜索，模型直接给出错误答案；
- 峰值已分配显存3,114,304,000 bytes，保留3,313,500,160 bytes；
- 结果SHA256：`516f29e88d82852fbe5c8b257646634729c7be41c66a27d09f4f3e16c22d6a48`。

### 5.2 自然策略4条

Run：`p2-qwen15b-fixture-n4-20260812`

- 4/4正常终止，4/4动作格式有效；
- 0次搜索、0条Reward=1；
- 峰值已分配显存3,183,588,352 bytes，保留3,445,620,736 bytes；
- 结果SHA256：`74b71ed13566991a8de3c17d4d80be7cc2d0cc720c5059cbd96ad9631766374f`。

这说明“允许搜索”的提示不能保证1.5B模型主动搜索。

### 5.3 强制首轮搜索诊断1条

Run：`p2-qwen15b-fixture-forced-search-n1-20260812`

该Run设置`--require-search-first`，只在第一轮追加显式搜索要求，并在JSON中标记
`diagnostic_prompt_modified=true`。它不与自然策略混算。

- 2个投影search动作，2次实际检索均为`success`；
- 3个时段动作中仅1个格式有效；
- 模型前两轮同时输出search和answer，投影器只执行search并将动作判为无效；
- 第二轮原文含正确答案，但没有形成可执行answer动作，最终Reward仍为0；
- 峰值已分配显存3,196,903,424 bytes，保留3,435,134,976 bytes；
- 结果SHA256：`5093ffaae7998d0f62ce02b1b7bbebcc25cec8a0124bd0fb45afc6c11e8263e8`。

该诊断证明检索回环技术可用，也定位了“多动作输出被投影丢弃”的策略失败。

### 5.4 自然策略16条（正式P2证据）

正式Run：`p2-qwen15b-fixture-n16-v2-20260812`

| 指标 | 结果 |
|---|---:|
| Episodes / 正常终止 | 16 / 16 |
| 活跃时段动作 | 24 |
| 格式有效动作 | 18 |
| 投影search动作 | 9 |
| 实际执行检索 | 8 |
| 成功检索 | 8 |
| Retriever失败 | 0 |
| answer动作 | 15 |
| Fixture Reward=1 | 2 |
| 总耗时 | 28.45秒 |
| 峰值已分配显存 | 5,213,613,056 bytes |
| 峰值保留显存 | 6,402,605,056 bytes |

结果SHA256：`2a905571b1388ebf44a65710da09741e2ef570a6b4c4243e6bba36c838a7900c`。

先前Run `p2-qwen15b-fixture-n16-20260812`保留为审计历史。其逐轮Trace正确，但旧汇总把
达到最大步数后未执行的search动作列成null检索状态。脚本已改为分别统计投影search和实际
执行检索，并用新Run ID重跑；没有覆盖或删除旧结果。

## 6. 本阶段发现的问题

1. **主动搜索不足**：前4条样本全部直接回答；16条中才自然出现搜索。
2. **格式遵循不稳定**：模型可能同时输出`<search>`和`<answer>`，按上游规则动作无效；
   `search_projection`仍保留第一个search片段，因此环境会执行搜索，但训练时应对无效动作处理做审计。
3. **最大步数语义**：达到`max_turns`时，search文本可被投影，但环境先终止，不会实际调用工具。
   因此“search动作数”和“Retriever请求数”必须分开报告。
4. **上游解析不一致**：投影器大小写不敏感，`SearchEnv._parse_action`大小写敏感，仍是后续修复候选。
5. **Gym维护风险**：运行出现Gym停更警告；当前NumPy固定为1.26.4，P3前需确认不影响训练。
6. **SDPA警告**：Qwen加载报告Sliding Window Attention未在SDPA实现。P2未观察到崩溃，
   但P3将使用veRL/vLLM路径，不能用本次Transformers结果替代训练引擎验证。
7. **观测补丁缺陷已修复**：旧保存补丁使用无上下文hunk，曾错误插入文件尾导致
   `IndentationError`。门禁在GPU实验前发现；现补丁有完整上下文，12项P1测试通过，且
   `git apply --reverse --check`和`git apply --check`均通过。

## 7. 可复现命令

```bash
cd /home/imc/yzy/agent/project3-search-agent-rl
bash scripts/apply_project_patches.sh

export PROJECT3_DATA_ROOT=/media/imc/data
PY=/media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124/bin/python
MODEL=/media/imc/data/project3-search-agent-rl/models/Qwen2.5-1.5B-Instruct
DATA=/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke

bash scripts/run_managed.sh <new-unique-run-id> 1 -- "$PY" \
  scripts/run_p2_model_search.py \
  --model "$MODEL" \
  --records "$DATA/records.jsonl" \
  --corpus "$DATA/fixture_corpus.jsonl" \
  --count 16
```

不要复用现有Run ID。`run_managed.sh`会拒绝覆盖已有目录。

## 8. P2结论与下一门禁

R1功能复现已经满足：冻结模型自然产生search、收到Retriever信息、继续生成最终answer，
Exact Match Reward可得到1，Retriever状态与模型动作分别记录。该结论仅限Fixture协议环境。

下一阶段不是直接在Fixture上训练，而是P2.5：固定真实Wikipedia Corpus和Retriever版本，完成
无答案泄漏的8/16检索质量与延迟检查。通过后才进入P3，并按用户要求使用tmux启动1步veRL
GRPO训练。
