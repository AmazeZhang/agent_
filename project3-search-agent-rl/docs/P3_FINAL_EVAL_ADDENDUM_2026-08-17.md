# P3 final-confirm512 评测 Addendum（dated 2026-08-17）

**性质**：本文件是对预注册 `P3_PHASE2_PREREG_2026-08-16.md` 的**探索性补充说明**，
**不修改**原预注册的任何主要假设、判据与门禁结构。原确认性检验**仍然唯一**：
Step 300 vs Base 在 final-confirm512 上的配对 EM / McNemar 检验
（PASS / INCONCLUSIVE / FAIL-TO-OBSERVE 三档，判据不变）。

---

## 1. 为什么新增本 addendum

Step 100 开发集（official-confirm256-v1，256 题）评测发现：EM 增益
（Base 20 → gs100 38，McNemar p=0.0005）**不是来自学会搜索**——gs100 全部
38 个正确答案零搜索依赖，搜索调用 124→12（真正 success 检索 10→1），
增益主要来自格式合规（compliance 51.6%→95.3%）与无效搜索抑制。

该观察先于 final-confirm512 盲测存在，故在**不触碰主判据**的前提下，
为最终结果提供一套**探索性机制指标**，用于解释（而非判定）Step 300 的
表现来源。这些指标与开发集分析使用完全相同的 episodes.jsonl 字段口径。

## 2. 探索性机制指标（Step 100 后提出，仅解释用）

全部基于评测 episodes.jsonl 的既有字段：`executed_search`、
`info.retrieval.status`（success / invalid_query / api_error / no_results）、
`error_observation`、每 episode 的 `steps` 长度与最终 reward。

| 指标 | 定义 |
|---|---|
| answer compliance | reward ≥ 0.1（有格式正确的 `<answer>`）的题目占比 |
| search attempts | 总 `executed_search=true` 步数 |
| valid/success/invalid/error | `info.retrieval.status` 分布 + `error_observation` 步数 |
| search→correct / search→wrong | 执行过搜索且最终 EM=1 / EM=0 的题目数 |
| no-search→correct | 未执行搜索且最终 EM=1 的题目数 |
| 每题搜索次数 | search steps / 题目数（含多搜与零搜） |
| 一步完成率 | 1 步完成的题目占比（`len(steps)==1`） |

**声明**：以上指标**来自 Step 100 后提出的探索性解释框架**，结果只作
描述性解释，**不冒充、不替换、不补充预注册确认性结论**；主判定仍只依据
Base vs Step 300 配对 EM/McNemar。

## 3. 官方 Search-R1 checkpoint：仅描述性参考

- `models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`（官方 3B GRPO
  checkpoint）在 final-confirm512 上用**同一冻结评测入口**评测；
- 只作描述性参考（量级/方向上下文），**不参与** PASS / INCONCLUSIVE /
  FAIL-TO-OBSERVE 判定，不改变任何判据。

## 4. 盲测协议（不变量）

- 运行前只校验数据 SHA：`heldout.parquet` =
  `94b39266c2d9c54a55b4471e90daa493ab083a889d8f23510dadd8194b304ecc`
  （manifest `outputs.heldout.sha256` 同值，512 行）；
- 不人工打开题目，不中途查看任何模型输出；
- 依次完整评测：Qwen2.5-3B Base → Step 300 → 官方 Search-R1；
- **三个模型全部 exit_code=0 完成后**才统一揭示与分析；
- 主判定仅计算 Base vs Step 300。

## 5. 工程变更记录（不改变评测语义）

- `scripts/run_p3_eval_vllm_official.sh`：新增 `final-confirm512` 数据选项
  （数据路径 + manifest key `heldout`），其余门禁、prompt、tokenizer、
  retriever、解码参数与 official-confirm256-v1 完全一致；SHA 门禁由
  `run_p3_eval_vllm_official.py` 的 `outputs[manifest_key].sha256` 校验。
- 三段冻结配置与冻结工具未修改。
