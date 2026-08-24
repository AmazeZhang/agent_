# P3 fresh v2 gs10 反事实检索评测预注册（2026-08-24）

## 1. 目的

验证 fresh Search-aware v2 gs10 的 held-out 正确答案是否依赖真实检索证据。
本实验不训练、不调参，只对已完成的同一 merged checkpoint 改变评测时返回的证据内容。

## 2. 冻结对象与协议

- 模型：`models/p3-v2-tenstep-gs10-merged-20260823a`
- 数据：`searchr1-official-confirm256-v1/heldout.parquet`，SHA256
  `ffebf468e756a673da267f5830cfc67f2e9c4dc44ec41c979a389c1efebfff60`
- 已完成 real 对照：`p3-eval-v2-tenstep-gs10-confirm256-20260823a`，EM 73/256
- 解码：vLLM native greedy，temperature=0.0，num_rollouts=1，seed=0
- 环境：max_steps=4，history_length=4，topk=3
- 代码：pristine `20bd331b` + `patches/v2/v2-0001..0007`，wrapper 重建门禁必须通过
- GPU：仅物理 GPU1；物理 GPU0 永久禁用，GPU5不启用
- Retriever：固定 Wiki-18 E5 index，21,015,324 vectors，loopback `127.0.0.1:18080`

## 3. 反事实条件与 Run ID

1. shuffled：模型查询仍先真实检索；成功结果的证据替换为固定映射
   `(i + 17) mod 256` 对应问题的真实检索文档。非成功状态原样保留。
   Run ID：`p3-eval-v2-tenstep-gs10-confirm256-shuffled-20260824a`。
2. no-evidence：每次成功搜索返回 3 条固定中性文档，不发起该次 HTTP 检索；
   非搜索协议、解码和步数预算不变。
   Run ID：`p3-eval-v2-tenstep-gs10-confirm256-noevidence-20260824a`。

每个入口必须在 episode loop 前原子写入
`retrieval_condition_preregistration.json`；shuffled 映射 SHA 必须与固定 offset
独立复算一致。

## 4. 判定门禁

机制证据通过要求同时满足：

1. real EM > shuffled EM；
2. real EM > no-evidence EM；
3. real-only correct > counterfactual-only correct（两个条件分别判定）；
4. real 中“搜索且正确”的题目在反事实条件下出现可审计翻转；
5. 差异不能由 data SHA、prompt、解码、GPU、API error 或题目缺失解释。

逐题报告精确双侧 McNemar p 值，但不以 `p<0.05` 作为唯一通过条件；本实验验证
证据依赖机制，不验证 v2 相对 clean GRPO 的模型间提升。

## 5. 中止条件与安全

- 启动前重新执行 `nvidia-smi` 与 `bash scripts/preflight.sh 1`；GPU1存在未知
  compute process 时不启动，不抢占、不结束未知进程。
- 每个条件使用新 Run ID、命名 tmux、`scripts/start_tmux_run.sh` 和
  `scripts/run_managed.sh`；禁止裸跑 GPU 评测。
- 任一 OOM、NaN/Inf、Xid、模型/数据/patch 门禁失败、Retriever health/vector
  数异常或 GPU 映射不是物理 GPU1，立即停止晋级并保留失败证据。
- 两个条件顺序执行，前一个完成并通过退出/日志/GPU/进程验收后才启动后一个。
- 不删除或覆盖任何旧 run、模型、checkpoint、日志或失败证据。

## 6. 后续决策

- 机制门禁通过：再设计多 seed 的 clean GRPO vs Search-aware v2 小规模训练对照。
- 机制门禁不通过：不扩大训练，先检查 reward proxy、证据匹配和搜索/作答收敛。
- 本预注册不授权 20 步以上、GPU5、全量数据或新的多卡扩容。
