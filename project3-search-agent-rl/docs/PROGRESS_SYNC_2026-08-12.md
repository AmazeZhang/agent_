# Search-R1复现进度同步

- 同步日期：2026-08-12
- 上一同步提交：`4543cf3`（P0/P1）
- verl-agent固定提交：`20bd331bdbc9026a5668e11362178e10ab7400c8`
- 当前阶段：P0、P1、P2完成；P2.5真实Retriever待执行
- 训练状态：未开始

## 已完成内容

### P0：环境、存储和非破坏性实验门禁

1. 确认`data`盘为`/dev/nvme0n1`，挂载在`/media/imc/data`，约3.2 TiB可用；
2. 建立项目专用的cache、datasets、models、indexes、runs、checkpoints目录；
3. 记录8×RTX 4090 D、驱动595.45.04、CUDA Toolkit 12.4.131及NUMA/PCIe拓扑；
4. GPU0硬禁用，避免影响Xorg/GNOME Remote Desktop；GPU5默认禁用；
5. 固定Python 3.10、Torch 2.6.0+cu124、vLLM 0.8.5.post1、Ray 2.43.0、
   TensorDict 0.8.3和Transformers 4.51.1；
6. 在data盘建立隔离环境并生成完整`pip freeze`锁；
7. 建立GPU空闲检查、项目锁、独立Session/Process Group、定向TERM/KILL、退出后显存核查；
8. 建立tmux训练包装。GPU1基础MatMul和FlashAttention SM89测试通过，退出后无残留进程。

对应文档：

- `docs/ENVIRONMENT_BASELINE_2026-08-12.md`
- `docs/EXPERIMENT_SAFETY.md`
- `configs/requirements-searchr1-repro.lock.txt`

### P1：数据、Retriever协议与Reward闭环

1. 从`PeterJinGo/nq_hotpotqa_train` revision
   `b7d80abfee334a7a91cb377544f09180d58b34f6`下载原始train/test Parquet；
2. 使用上游脚本处理为veRL需要的`prompt/reward_model/extra_info/env_kwargs`结构；
3. 得到训练169,615条、测试51,713条；发现原始数据10个规范化问题跨Split重叠；
4. 使用稳定SHA256排序生成8条训练、16条验证Smoke，重建前后哈希一致；
5. 实现只用于协议测试的确定性Fixture Retriever，其文档明确标记为Ground Truth衍生；
6. 跑通`<search>`→HTTP `/retrieve`→`<information>`→`<answer>`→严格EM；
7. 增加Retriever状态观测补丁，使API错误可与模型错误分开统计；
8. 12项CPU测试通过，耗时2.54秒；生成一条Reward=1的模型无关Trace；
9. 未下载Wikipedia Corpus或E5 Flat Index，未启动训练。

对应文档：`docs/P1_SEARCH_PIPELINE_AUDIT_2026-08-12.md`。

### P2：冻结模型与Search环境集成

1. 固定并下载`Qwen/Qwen2.5-1.5B-Instruct` revision
   `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，许可证Apache-2.0；
2. 权重落在data盘，3.087GB safetensors的SHA256与上游LFS对象一致；
3. 直接复用上游环境管理器、动作投影、SearchEnv和Chat Template实现模型驱动多轮链路；
4. 使用物理GPU1按1、4、16条逐级执行，所有Run退出码为0且无显存进程残留；
5. 正式16条Run有9个投影search、8次实际检索，8次均成功，16条全部终止；
6. 发现主动搜索不足、多动作格式无效以及最大步数search不实际执行三类基线问题；
7. 通过强制首轮搜索的独立诊断验证完整回环，但不把它混入自然策略指标；
8. Fixture由Ground Truth衍生，Reward结果只作为集成证据，不作为模型质量或论文分数；
9. P2达到Fixture环境下R1功能复现，训练仍未开始。

对应文档：`docs/P2_MODEL_SEARCH_INTEGRATION_2026-08-12.md`。

## 数据和产物位置

大文件不进入Git：

```text
/media/imc/data/project3-search-agent-rl/datasets/searchr1-upstream
/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke
/media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124
/media/imc/data/project3-search-agent-rl/models/Qwen2.5-1.5B-Instruct
/media/imc/data/project3-search-agent-rl/runs/p2-qwen15b-fixture-*
```

关键哈希：

```text
processed train.parquet  aa98bf95dec9466899395e5d44e56e1b765cef7bc6b9ea226f5e6129bd0d360a
processed test.parquet   b00cd074f2e5de5eb464d17a0289217159d332a13879dd1ac2bae5312cc167de
Smoke manifest           c2f92e66702e1e5597a9e47759172bef6db5b5d17e988a08e3d448a89ef6c3b3
deterministic trace       5c2ac4dc40462720b44578ee64be507321c43440501419279cbdef1f7da28a62
direct requirements       dd03ff2a705b4b0446dce48358e83fc8846a26a2979855b2456dd7f76d36c530
third-party package lock  758ef9afec5a0796ecec4a9c072bcd72f2c973719ab756206959fe35bd177a65
Qwen model weights         dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
P2 official result JSON   2a905571b1388ebf44a65710da09741e2ef570a6b4c4243e6bba36c838a7900c
```

上游数据集Metadata没有许可证字段。当前状态为“许可证未声明、待人工核验”，不能推断为
Apache或据此重新分发数据。Git仅保存生成脚本、Manifest规则和哈希，不保存数据内容。

## 从干净仓库同步P1

```bash
cd /home/imc/yzy/agent/project3-search-agent-rl
git submodule update --init --recursive
bash scripts/apply_project_patches.sh

export PROJECT3_DATA_ROOT=/media/imc/data
source /media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124/bin/activate

python vendor/verl-agent/examples/data_preprocess/preprocess_search_r1_dataset.py \
  --local_dir /media/imc/data/project3-search-agent-rl/datasets/searchr1-upstream

CUDA_VISIBLE_DEVICES='' python scripts/build_p1_smoke.py \
  --source-dir /media/imc/data/project3-search-agent-rl/datasets/searchr1-upstream \
  --output-dir /media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke

CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/vendor/verl-agent:$PWD" \
  python -m pytest -q tests/test_search_p1.py
```

预期：`12 passed`。localhost HTTP测试需要允许绑定`127.0.0.1`随机端口。

## 下一步：P2.5与P3

1. 固定真实Wikipedia Corpus、E5 Retriever模型、索引来源、revision、许可证和哈希；
2. 优先用CPU或裁剪真实Corpus检查无答案泄漏的8/16检索结果、延迟和失败分类；
3. 禁止使用Ground-Truth Fixture训练；
4. 真实Retriever门禁通过后，翻译并审计可执行的veRL Hydra单步配置；
5. P3训练仅使用物理GPU1，通过tmux和`run_managed.sh`启动；
6. 首次只做1个非零参数更新、Checkpoint保存/重载和Reward到Loss审计；
7. 训练启动时向用户提供tmux attach、日志查看和安全停止指令。
