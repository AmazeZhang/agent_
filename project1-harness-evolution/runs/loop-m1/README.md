# M1 基线采集与诊断记录（2026-08-07）

## 基线采集（retail40-v1）

- 40/40 任务完成（tmux p1-m1-baseline40，18:49 启动，约 10 分钟/5 任务）
- **成功率 0.9（36/40）**，失败任务: **16, 27, 34, 38**（reward=0.0）
- 成本: agent $0.0458 + user $0.0124 ≈ $0.058 总计（40 任务，~$0.0015/任务）
- 数据: /media/imc/data/yzy/agent/project1/baseline/retail40-v1/results.json（数据盘）
- 汇总: data/baseline_summary.json（Git）

## 数据集与划分（M2 已就位）

- data/datasets/tau2_retail.jsonl + task_manifest.json（40 任务）
- data/partition.py: dev 24 / val 8 / holdout 8（partition_hash 3694ebbc383f0e1e）
- 失败任务落位: 16→dev, 38→dev, 27→val, 34→holdout（三集都有失败样本）

## 诊断（AgentRx × DeepSeek）

- 4 条失败轨迹全部跑 AgentRx 六阶段（oneshot，DeepSeek）
- **问题 1（已修）**: AgentRx 目录输入时多轨迹共享 run_dir，judge 输出互相覆盖
  （judge_output/runs/run1.json 只留最后一条）→ 改为逐轨迹独立 run_dir（traj_<id>/）
- **问题 2（记录中）**: 部分轨迹 static 检查阶段崩溃——AgentRx LLM 生成的检查代码
  引用未定义辅助函数 `user_authentication_required_before_order_operations`
  （exec 命名空间里没有）→ 脚本容忍单轨迹失败，失败条目如实记录
- 诊断结果: data/diagnostics/summary.json（含 failed_trajectories 字段）

## 关键发现（诚实记录）

1. 首轮诊断判 task 16 的轨迹为 category 10（"no failure"）——AgentRx 的失败语义
   （agent 任务完成度）与 tau2 reward（DB 校验）**不对齐**。诊断是"预测"而非"标注"，
   闭环中仅作为提示反馈注入，不当作 ground truth。
2. 诊断准确率评测需要人工标注（调研报告口径），本轮不声称诊断准确率。

## 环境（M3 集成用）

- agent-lightning venv（py3.11）通过 PYTHONPATH 引入 tau2 src：
  `PYTHONPATH=vendor/tau2-bench/src:vendor/agent-lightning`
- 复制纯 Python 依赖（loguru/tenacity/typer/deepdiff/addict/docstring_parser/orderly_set/
  pytz/shellingham/tabulate/toml/tzdata）到 agent-lightning venv
- pandas 2.3.2 (cp311) 手动下载 wheel 安装（uv pip 卡死不可用，PyPI 直连可用）
- tau2_rollout smoke 通过: 任务 30 → reward 1.0（v0 资源 prompt 注入生效）
