# M4 GEPA 接入记录（2026-08-07）

## 包安装

- gepa 0.1.4（PyPI）核心零依赖（litellm 只在 full extra/examples）
- 纯 Python 复制到 agent-lightning venv site-packages（与 M1 复制依赖同方案）
- `gepa.optimize()` 主入口确认：seed_candidate/trainset/valset/adapter/reflection_lm/
  max_metric_calls/run_dir（断点续跑）

## 新增文件

- optimizers/gepa_adapter.py:
  - `Tau2GEPAAdapter`: evaluate（candidate prompt → tau2 仿真 reward）+ 
    make_reflective_dataset（失败轨迹反馈，diagnosis 模式注入 AgentRx 诊断 ASI）
  - `DeepSeekLM`: GEPA LanguageModel 协议（openai 同步客户端，与 APO 臂同模型）
  - **线程安全指令注入**: patch LLMAgent.system_prompt 读 thread-local
    （tau2 的 AGENT_INSTRUCTION 是模块全局，多线程并行仿真有竞态）
  - evaluate 内 ThreadPoolExecutor 并行（max_workers=2）
- optimizers/run_gepa.py: GEPA 闭环 runner（进化 → 候选过滤 → val 独立重跑 → gate →
  版本更新），复用 run_apo_loop 的 gate/版本管理

## 自检（全部通过）

- 诊断格式化（类别映射 10 类）、reflective dataset 结构
  （Task Input/Generated Output/Feedback 三键，匹配 InstructionProposalSignature）
- plain 模式不注入诊断、DeepSeekLM 构造、run_gepa import 链

## 运行状态（20:16 启动）

- p1-m4-gepa-plain / p1-m4-gepa-diag: max_metric_calls=40，max_workers=2
- 预算估算: seed val 8 + ~3 轮迭代 × (3 minibatch + 8 val) ≈ 40 calls ≈ 90-120 分钟
- rollout_log.jsonl 每仿真诚实记录

## 消融矩阵（SPEC 04）

| 臂 | 优化算法 | 诊断反馈 |
|---|---|---|
| baseline | —（M1 已完成） | — |
| apo-plain | APO | 无 |
| apo-diagnosis | APO | AgentRx 诊断注入（诊断注入适配器） |
| gepa-plain | GEPA | 无 |
| gepa-diagnosis | GEPA | AgentRx 诊断 ASI 注入（reflective feedback） |
