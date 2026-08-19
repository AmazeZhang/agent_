# P3 干净 upstream 20bd331b 评测：管线交付与 smoke-16 结果（2026-08-19）

状态：**评测管线完成并验证**；GPU1 smoke-16 跑通但 **fail-closed 生效**——官方 3B
GRPO checkpoint 在干净上游语义 + greedy 下 16/16 条第 1 轮直接 `<answer>`、零搜索、
EM 0/16。该结果是**模型行为**（非管线缺陷），管线各环节已与上游训练逐行对齐验证。

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

## 2. smoke-16 结果（run p3-eval-upstream-clean-smoke16-20260819a，GPU1）

- 16 条全部第 1 轮输出 `<think>…</think>\n<answer>…</answer>`（raw action 完整，
  无截断：len 85–241，16/16 含闭合标签）；
- **16/16 条 raw action 中从未出现 `<search>` 标签**（零搜索倾向）；
- 全部第 1 轮即 done（`<answer>` 双标签）→ 无第二轮；
- EM = 0/16（模型直接回答且全部幻觉答错）；
- `check_round2_prompts` → `checked_episodes=0`（无"第 1 步搜索且继续"的 episode）→
  **fail-closed 生效，脚本 exit 2，confirm-256 未启动**（按用户流程设计）。

## 3. 交叉验证：不是管线缺陷

| 项 | official 线（此前 confirm256-gs50） | 本干净线 smoke-16 |
|---|---|---|
| prompt | 裸问题（`_sync_reset` 原样返回 question） | SEARCH_TEMPLATE_NO_HIS |
| 第 1 轮直接 `<answer>` | 218/256（85%） | 16/16（100%） |
| 出现 `<search>` 标签 | **0 次**（38 条"tool_calling"为无格式开放文本，query=None → 工具 error） | 0 次 |
| EM | 30/256 = 11.7% | 0/16 |

- 两条线独立实现（raw action 直进 env vs 完整 manager+投影），**官方 checkpoint
  greedy 解码下从不按格式搜索**为一致结论；
- 训练第一轮 prompt 即 SEARCH_TEMPLATE_NO_HIS（含"可以搜索"的措辞），但 greedy
  收敛到最高概率路径 = 直接回答（GRPO 模型在确定性解码下的已知倾向）；
- 此前 official 线 19.1% EM（49/256）同为第 1 轮直接答对的子集，与搜索无关。

## 4. 结论与待决策

- **评测管线本身已交付并验证**（门禁、生成、env 交互、记录、fail-closed 全部按
  设计工作）；
- 用户要求的"第二轮 prompt 含原问题 + search query + information"检查**无检查对象**
  （模型不搜索）——这是官方 checkpoint 在 greedy 下的真实基线行为；
- 待用户决策（不启动任何训练、不修改自研 Reward）：
  - A. 保持 greedy 直接跑 confirm-256：获得 256 条规模 EM 基线（预期 ≈ 直接答对率，
    搜索链路指标如实报告为 0/不存在）；
  - B. 采样（temperature>0）重试 smoke：验证"搜索→answer→correct"链路是否存在
    （改评测解码口径，需批准）；
  - C. 暂停，先确认官方 checkpoint 的预期行为/训练设置。

## 5. 资源状态

- 评测进程正常退出（exit 2，fail-closed 路径），GPU1 回 18MiB 基线，无残留进程；
- run 目录完整：`runs/p3-eval-upstream-clean-smoke16-20260819a/`（results.json +
  episodes.jsonl + metadata.env + stdout/stderr.log）；
- Retriever 未受影响（21,015,324 向量就绪）。
