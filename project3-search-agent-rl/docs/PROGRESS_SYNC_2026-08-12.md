# Search-R1复现进度同步

- 同步日期：2026-08-12
- 根仓库基线：`c3b946272c70dea17744988fa3607d834e2bbf1e`
- verl-agent固定提交：`20bd331bdbc9026a5668e11362178e10ab7400c8`
- 当前阶段：P0、P1完成；P2待执行
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
9. 未下载Wikipedia Corpus、E5 Flat Index或任何模型，未启动训练。

对应文档：`docs/P1_SEARCH_PIPELINE_AUDIT_2026-08-12.md`。

## 数据和产物位置

大文件不进入Git：

```text
/media/imc/data/project3-search-agent-rl/datasets/searchr1-upstream
/media/imc/data/project3-search-agent-rl/datasets/searchr1-smoke
/media/imc/data/project3-search-agent-rl/envs/searchr1-repro-cu124
```

关键哈希：

```text
processed train.parquet  aa98bf95dec9466899395e5d44e56e1b765cef7bc6b9ea226f5e6129bd0d360a
processed test.parquet   b00cd074f2e5de5eb464d17a0289217159d332a13879dd1ac2bae5312cc167de
Smoke manifest           c2f92e66702e1e5597a9e47759172bef6db5b5d17e988a08e3d448a89ef6c3b3
deterministic trace       5c2ac4dc40462720b44578ee64be507321c43440501419279cbdef1f7da28a62
direct requirements       dd03ff2a705b4b0446dce48358e83fc8846a26a2979855b2456dd7f76d36c530
third-party package lock  758ef9afec5a0796ecec4a9c072bcd72f2c973719ab756206959fe35bd177a65
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

## 下一步：P2

1. 下载并固定`Qwen/Qwen2.5-1.5B-Instruct`到data盘；
2. 在下载前记录模型revision、文件列表、许可证与来源；
3. 使用物理GPU1依次执行1条、4条、16条模型驱动Search；
4. 首轮使用明确标记的Fixture Retriever，只证明动作/环境/模型集成，不报告质量；
5. 保存完整Trace、格式错误、搜索次数、延迟、峰值显存和Retriever错误分类；
6. 每次任务退出后确认GPU1无Ray/vLLM/Python残留；
7. 达到R1后再准备真实Corpus Retriever，训练仍需等P3并通过tmux启动。
