# P3 干净 upstream 20bd331b 评测：管线交付与官方模型评测（2026-08-19）

状态：**评测管线完成并验证**。首轮 smoke-16 使用了**错误的模型**（自训 Step300
`p3-formal-segment-100-300-gs300-merged-20260817b`，其 greedy 零搜索是**已知坍缩**，
不能回答官方模型行为问题）——该结果已更正，见 §4/§5。以下评测以**真官方
Search-R1 checkpoint**（`SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`）进行。

## 0. 更正：模型身份（用户指正，2026-08-19）

首版 smoke-16（run p3-eval-upstream-clean-smoke16-20260819a）误把自训模型
`p3-formal-segment-100-300-gs300-merged-20260817b`（Step300）当作官方
Search-R1 3B checkpoint。该模型 greedy 零搜索属**训练坍缩**，不能代表官方
模型行为；原 §4 中"官方 checkpoint 在 greedy 下从不搜索"的表述作废。管线本身
（门禁、生成、env 交互、记录、fail-closed）不受影响，协议对齐验证（§1）仍成立。

## 1. 评测管线（commit f65c929，已推送）

| 文件 | 内容 |
|---|---|
| `scripts/run_p3_eval_upstream_clean.py` | 干净线评测主体：门禁 + vLLM greedy + SearchEnvironmentManager + 上游 search_projection + 单层官方 Search prompt |
| `scripts/run_p3_eval_upstream_clean.sh` | 受管 wrapper（门禁：clean tree pin+pristine+无补丁标记、GPU1、merged-model verify、retriever health、managed env） |
| `tests/test_eval_upstream_clean.py` | 22 项 CPU 测试全绿 |
| `.gitignore` | `vendor/upstream-20bd331b/`（派生 worktree，不入库） |

### 与上游训练逐行对齐的证据（干净线协议 = 训练协议）

- **第一轮 prompt**：`preprocess_single_sample`（`agent_system/multi_turn_rollout/
  rollout_loop.py`）每轮从 `obs['text']` 构造 **user-only chat，完全忽略数据集
  raw_prompt**；第一轮 = `SEARCH_TEMPLATE_NO_HIS`（含 task question）。
  评测同构。✓
- **后续轮**：`SearchEnvironmentManager.build_text_obs` = `SEARCH_TEMPLATE`
  （task_description + step_count + memory_context，历史 = `Step n:<search>q</search>
  <information>…</information>`）。评测同构。✓
- **tokenizer**：官方训练 `actor_rollout_ref.model.path=Qwen2.5-3B`（Base）→ 训练时
  apply_chat_template 即 Base Qwen 模板；评测 pin Base tokenizer，字节一致。✓
- **env 协议**：`SearchMultiProcessEnv` + `SearchEnv`（`_is_done` = turns≥max_steps 或
  raw action 含 `<answer>`+`</answer>`；`_parse_action` 只解析 `<search>…</search>`）。
  ✓
- **奖励累计**：`vanilla_multi_turn_loop` 语义——每轮 step 全部 env，但 reward 仅按
  `active_masks` 累计（done 轮不计）；中间轮恒 0 → episode_reward == 终局 reward。
  评测同构。✓
- **终局 reward**：上游 skyrl `compute_score(chat_history, ground_truth)`
  format_score=0.0 → EM = reward ≥ 1.0。✓
- **max_steps=4 / history_length=4 / topk=3 / timeout=180 / seed=0 / greedy**。

### 门禁（全部 PASS）

clean tree pin 20bd331b + `status --porcelain` 空 + 全树无 `search_aware_step_reward`
标记；GPU1-only；`verify_p3_merged_model.py` → `VERIFY_MERGED: PASS`；Retriever
/health 21,015,324 向量 ready；数据 SHA256 与 manifest 一致；smoke 16 条 ∩ 训练 0
重叠；VLLM_USE_V1=0（训练 rollout 引擎路径）。

## 2. 首轮 smoke-16（自训 Step300，已更正，保留作历史）

run p3-eval-upstream-clean-smoke16-20260819a（模型 = 自训
`p3-formal-segment-100-300-gs300-merged-20260817b`）：16/16 条第 1 轮直接
`<answer>`、零搜索、EM 0/16、`checked_episodes=0` → fail-closed exit 2。该行为
是**自训模型的已知坍缩**，不反映官方模型。管线在此 run 上验证通过（门禁、
生成、env 交互、记录、fail-closed 全按设计）。

## 3. 官方模型资产指纹（启动前记录，2026-08-19）

- 绝对路径：`/media/imc/data/project3-search-agent-rl/models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo`
- `config.json` SHA256：`b27a7aadfdb9c5967ccb48edb034c6dc7edddc8c0600e9e9d47db3f445a39fcd`
- 权重 SHA256：
  - `model-00001-of-00003.safetensors` `7ac54e1b9762c3c6d639da28a2cca177fe7db092ff5cf6e5a9a7849a36a9dabf`
  - `model-00002-of-00003.safetensors` `98b373c4a6805af7723f2b31a5e72a919f4d7c021b6f4e67d91f579a08db8c67`
  - `model-00003-of-00003.safetensors` `f1607045409131e298ad87b485a7fb74d02891178a0106a2df79cf8daf7b2c54`
  - `model.safetensors.index.json` `3a899e7f6ef595e15c34b5ffc7e4a1df6131d7c371418b1a6e8f823ab8a7302d`
- 结构：Qwen2.5-3B（36 层 / 16 heads / 2 kv heads / hidden 2048 / vocab 151936 /
  tie_word_embeddings），`_name_or_path=Qwen/Qwen2.5-3B`
- wrapper 默认 `PROJECT3_EVAL_MODEL` 已改为该官方模型；指纹写入 wrapper 注释与
  results.json（model_path）。

## 4. 官方模型评测计划（2026-08-19 起，用户指令）

1. **greedy 主评测 smoke-16**（temperature=0，run 前缀
   `p3-eval-upstream-clean-official-smoke16-*`）：round-2 prompt 检查（原问题 +
   search query + information）通过 → 进 confirm-256；
2. **行为诊断**（同 16 题，temperature=1、每题 5 rollout、固定 seed：
   rollout i 用 seed 0+i）：验证"搜索→answer→correct"链路是否存在，策略支持
   性质，**不设 fail-closed 门**；
3. 两组均报告 **search→answer / search→correct**（episode 粒度 + per-question
   聚合）；greedy 仍是主评测口径，采样只作策略支持诊断；
4. 不启动任何训练、不修改自研 Reward；patch 0009 不进行。

## 5. 官方模型评测结果（2026-08-19/20）

### 5.1 greedy smoke-16（run p3-eval-upstream-clean-official-smoke16-20260819a）

temperature=0、num_rollouts=1、seed=0、16 题（smoke test.parquet）。

| 指标 | 值 |
|---|---|
| episodes | 16 |
| 搜索率（searched_episodes） | 16/16 = 100% |
| 搜索成功（search_successful_steps） | 41/41 = 100% |
| round-2 prompt 检查（fail-closed 门禁） | checked 16 / passed 16 / **PASS** |
| EM | 1/16 = 6.25% |
| **search→answer** | 1.0 |
| **search→correct** | 0.0625 |
| no_search_episodes | 0 |

### 5.2 行为诊断（run p3-eval-upstream-clean-official-diag16-sampling-20260819b）

同 16 题，temperature=1、每题 5 rollout（rollout i 用 seed 0+i）、main_mode=False
（不设 fail-closed 门，仅策略支持诊断）。

| 指标 | episode 级（80） | question 级（16） |
|---|---|---|
| 搜索率 | 80/80 = 100% | 16/16 = 100% |
| 搜索成功 | 207/207 = 100% | — |
| answer_compliance | 78/80 = 97.5% | 16/16 = 100% |
| **search→answer** | 0.975 | 1.0 |
| **search→correct** | 0.0875 | 0.3125 |
| EM | 7/80 = 8.75% | 5/16 题 ≥1 rollout 正确 |

round-2 prompt 检查 80/80 passed（诊断模式不 fail-closed）。

### 5.3 结论（官方 checkpoint 行为）

- 官方模型在 **greedy 主评测下搜索链路完好**：100% 搜索、100% 搜索成功、
  search→answer=1.0 —— 与自训 Step300 的"greedy 零搜索坍缩"形成鲜明对比，
  Step300 的结果（§2）确实不能代表官方模型。
- 但 search→correct 仅 6.25%（greedy）/ 31.25%（sampling, question 级）：
  搜索→正确回答的链路存在但不稳定；采样诊断提升 question 级正确率 5 倍，
  说明该模型具有搜索后答对的能力，greedy 主评测是其下限表现。
- greedy 仍是主评测口径；采样只作策略支持诊断。

### 5.4 confirm-256（run p3-eval-upstream-clean-official-confirm256-20260820a/b）

- 2026-08-20 启动（GPU1，greedy 主评测，official-confirm256-v1 heldout.parquet，
  256 题，batch 24）。
- **run a/b 事故（20260820a/b，均已中止）**：搜索请求收到 retriever 422
  （`string_too_short`，query 为空），重试仍失败 → 该批搜索失效、EM 必为 0。
  **根因（2026-08-20，已确诊并修复）**：
  - 官方模型在**后期轮次**（R2-R4）对部分 question 生成 `<search></search>`
    空查询（查询退化，如 `your query`、`and`、重复查询）；
  - 上游 `SearchToolGroup.search` 只拦 `None` 不拦空串 → 空查询到达
    retriever → 其 schema `query: str = Field(min_length=1)` 返回 422 →
    错误文本污染上下文、搜索链断裂；
  - 完整复现：24 题 × 4 轮（多批、真实 env+retriever）稳定复现空查询
    （~6-8 个/72 题）；最小 `SearchEnv.step("<search></search>")` 直接复现
    422；smoke-16/diag16 轮次短（多数 R2 answer）无空查询所以未触发；
  - **修复（eval 端，不改 clean tree）**：`sanitize_empty_search_actions`
    把空查询替换为 `""`（env 返回空 observation、**不发 HTTP**，与
    projection 的 no-tags 语义一致）并标 `valids=0`；新增 6 项单元测试，
    共 29 项全绿（commit 待推送）。
- **run c（20260820c，最终结果）**：修复后重跑，256 题 greedy 全部完成，
  **全程 0 个 422**（对比 run b 同期 8 个）：

  | 指标 | 值 |
  |---|---|
  | EM（env 口径，训练同构） | 7/256 = **2.73%** |
  | 搜索率 | 256/256 = 100% |
  | 搜索成功 | 685/685 = 100% |
  | **search→answer** | 0.996（255/256） |
  | **search→correct** | 0.0273 |
  | round-2 prompt gate | checked 254 / passed 254 / **PASS** |
  | offline audit（正确） | 34/256 = 13.3%（含 27 个格式违规） |

  - **offline audit 差异（27 个 env=0 / offline=1，全为 one-sided）**：模型在
    `search+answer` 同轮输出（both tags）时，投影只取 `<search>` 块并丢弃
    `<answer>`（valids=0，env 只执行 search）→ 该答案按上游协议作废
    （env 的 chat_history 存投影后 action，训练同构）。offline 用完整
    raw_action 重新评分发现这 27 题**模型其实写对了答案**——属格式违规，
    不是不知道答案。**EM 口径以 env（训练同构）为准**；offline 仅审计。
  - **分源 EM**：triviaqa 12.5%、2wiki 6.25%、nq 1.56%、hotpotqa/popqa/
    musique/bamboogle 0%。
  - **结论**：官方 Search-R1 3B 在 heldout-256 greedy 下**搜索链路完好**
    （100% 搜索、100% 搜索成功、search→answer 0.996）但 **EM 仅 2.7%**；
    与采样诊断对比（question 级 search→correct 31.25% vs greedy 2.7%），
    **采样是提取正确答案的关键路径，greedy 是下限表现**；另有 ~10.5%
    的正确答案因 both-tags 格式违规被上游协议丢弃。

## 6. 资源状态

- 首轮 smoke-16（Step300）正常退出（exit 2，fail-closed 路径），GPU1 回 18MiB
  基线，无残留进程；
- run 目录完整：`runs/p3-eval-upstream-clean-smoke16-20260819a/`（results.json +
  episodes.jsonl + metadata.env + stdout/stderr.log）；
- Retriever 未受影响（21,015,324 向量就绪）。
