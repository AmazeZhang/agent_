# P3 StepSearch-3B 外部开源策略兼容性 Smoke 预注册（2026-08-28）

## 目的与声明边界

本实验只验证公开 `StepSearch-3B-Base` 能否在现有 Project 3 Wiki-18 Retriever 和
SearchEnv 中执行其官方 `plan -> search -> information -> observation -> replan` 协议。
它是 **external open-source baseline compatibility smoke**，不是训练、不是 Aware-v2
同起点消融，16 题结果不得用于质量提升声明。

## 冻结输入

- 代码：commit `faea311`；StepSearch adapter 已通过 14 项 CPU 测试；
- 开源源码参考：`Zillwang/StepSearch` commit
  `43215bab9118a4c8e01b15082f74b2aea30c1fc8`；
- 模型：`Zill1/StepSearch-3B-Base` revision
  `a89ec38cd2a21461320f9a81eb29be019c142fe5`；
- 数据：既有 `searchr1-smoke/test.parquet` 16 题及其 manifest SHA 门禁；
- Retriever：既有共享 PID 1355816，CPU Wiki-18 E5 IndexFlatIP，21,015,324 vectors，
  `127.0.0.1:18080`；本轮复用，不创建、不停止；
- 推理：greedy temperature=0，1 rollout/question，top-k=3，max_steps=4，
  history_length=4，max_new_tokens=256；
- 资源：只允许物理 GPU1；物理 GPU0/5 禁用；命名 tmux + `start_tmux_run.sh` +
  `run_managed.sh`；新 Run ID，不覆盖任何已有目录。

## 单变量与适配语义

相对既有 v2 evaluation harness，只改变模型、其配套 tokenizer 和 prompt/history 协议：

- 首轮使用 StepSearch 公开训练 prompt；
- 后续上下文保留模型原始 `<plan>/<observation>/<search>` 文本与真实 Retriever 返回的
  `<information>`；
- 工具侧仍使用上游 `search_projection`，只执行首个合法 `<search>` 或 `<answer>`；
- 空 query 仍在 HTTP 前 fail closed；
- Retriever、top-k、解码、终止奖励、evidence matcher 与数据均不改。

`max_steps=4` 是为了与当前项目协议对齐；StepSearch 官方训练使用 max_turns=5，因此后续若
需要评估第5轮，必须作为新的单变量 evaluation-only 实验另行预注册，不能混入本 smoke。

## 通过门禁

只有以下条件全部满足才允许进入 confirm256 外部基线：

1. 模型 revision/文件大小/safetensors SHA 校验通过，完整 load gate 通过；
2. 受管 Run `exit_code=0`，无 OOM、NaN/Inf、traceback、segfault；
3. 正好 16 个 episode，输出采用原子写入，无 `.partial`；
4. 至少一个非空合法 search，使 round-2 prompt gate 实际检查到 question/query/information，
   且全部通过；
5. Retriever 无 api error/timeout，既有 PID/端口身份不变；
6. GPU1 在结束后无本 Run compute process，GPU0 仍只有桌面进程；
7. 原始 action 中的 plan/observation 确实进入下一轮 prompt（不能只验证投影后的 query）。

EM、搜索率和 evidence-hit 在 smoke 中只记录，不作为晋级阈值。若协议不兼容，保留失败证据，
只修复兼容性问题后使用新 Run ID 重跑，不直接进入 confirm256 或训练。

