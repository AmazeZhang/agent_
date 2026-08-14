# P3 Held-out 评测计划：Base / Step 2 / Step 5 纯评测

## 1. 目标与边界

五步工程闭环（Attempt H）通过后，下一步是建立"只评测、绝不训练"的 held-out 评测，
在相同条件下比较基础模型（Step 0）、Step 2（Attempt G）与 Step 5（Attempt H），
判断训练是否带来可测量的改善，并为零 reward 原因分析提供证据。**本阶段不扩大训练步数。**

声明边界（与 EXPERIMENT_AUDIT 一致）：

- smoke-16 只验证评测管线本身，**绝不用于质量声明**（manifest forbidden_use）；
- heldout-32 是第一轮正式对比的**小样本初步证据**（32 条、分源 2–8 条），
  不能单独构成"完整 Search-R1 复现"或"泛化"声明；
- 评测解码走 HF transformers 贪心生成，与训练的 vLLM rollout 机制不同；
  三模型之间完全同条件，但若 heldout-32 出现明确提升，必须先用 verl/vLLM
  原生评测复核关键结论，再考虑多 seed 与更大评测集。

## 2. 固定评测条件（与训练 hydra overrides 一致）

| 参数 | 值 | 来源 |
|---|---|---|
| 模型 | `Qwen2.5-1.5B-Instruct`（base）/ +LoRA rank32 alpha32 all-linear | run h overrides |
| 解码 | 贪心（temperature 0），HF transformers | 训练 `val_kwargs.temperature=0` |
| max_steps | 2 | `env.max_steps=2` |
| history_length | 2 | `env.history_length=2` |
| topk / timeout | 3 / 180s | `env.search.*` |
| max_input / max_new | 2048 / 256 | `data.max_prompt_length` / `max_response_length` |
| seed | 0 | `env.seed=0` |
| 动作语义 | `search_projection` + skyRL 严格 EM（`compute_score`） | 与训练相同 |
| Retriever | 真实 CPU Wiki-18 `IndexFlatIP` 21,015,324 向量，`127.0.0.1:18080` | /health 门禁 |
| 环境 | `envs/searchr1-repro-cu124`，veRL `20bd331b…`，补丁 0001–0003 | 与训练相同 |

## 3. 评测集

| 集 | 构成 | 用途 |
|---|---|---|
| smoke test（16 条） | smoke 集内与 train 完全不相交（manifest overlap=0） | 管线门禁：模型加载、LoRA 挂载、门禁、指标记录、原子产物 |
| heldout-32 | 上游 test 51,713 条确定性抽样（SHA256 升序 + 分源配额），排除上游 train 169,615 条、smoke train 8 条、smoke test 16 条中出现的规范化问题 | Step 0/2/5 第一轮正式对比 |

heldout-32 构建器 `scripts/build_p3_heldout_eval.py`（CPU-only）：

- 选择域 `searchr1-p3-eval-v1\0source\0normalized_question`，升序取配额
  （nq 8 / hotpotqa 8 / popqa 4 / 2wikimultihopqa 4 / triviaqa 4 / musique 2 / bamboogle 2 = 32）；
- 拒绝覆盖已有输出；产物 `heldout.parquet` + `records.jsonl` + `manifest.json`
  （含每文件 SHA256、泄漏计数、源行索引与答案哈希）；
- 重建确定性已验证：两次构建 `heldout.parquet` 与 `records.jsonl` 字节一致。

## 4. 评测脚本与零训练保证

`scripts/run_p3_eval_heldout.py`（HF 侧，克隆 P2 已验证骨架）：

- **构造性零训练**：不创建优化器/scheduler，不调用 backward，不 import Ray，
  全部生成在 `torch.inference_mode()` 内；base 模型 + `PeftModel.from_pretrained(
  is_trainable=False)` 只读挂载 LoRA；
- **启动门禁（abort）**：必须在 `run_managed.sh` 内（`PROJECT3_RUN_ID/RUN_DIR`）；
  单逻辑 GPU 且物理 GPU0 不得暴露；Retriever `/health` 必须 `ready` 且
  `vectors==21015324`（真实索引，非 fixture）；数据文件 SHA256 与 manifest 核对；
  评测问题与 smoke-train 交集必须为 0；
- **指标**：总体/分源 EM 与 success、answer 格式合规率、search 调用次数、
  检索状态分布（success / invalid_query / api_error / no_results）、无效查询率、
  无效动作率（混合/重复标签）、平均步数；离线复核分数（仅模型 actions 上
  skyRL EM，与训练环境评分同语义）与环境 reward 的吻合数；
- **产物**（写 `PROJECT3_RUN_DIR`，`.partial`→fsync→原子 rename）：
  `results.json`（参数、门禁结果、adapter/数据 SHA256、指标、峰值显存、
  `decoding_backend` 与 HF-vs-vLLM 差异声明）+ `episodes.jsonl`（逐 episode 证据）。

受管 wrapper `scripts/run_p3_eval_heldout.sh` 镜像训练脚本门禁：veRL commit pin
（exit 10）、路径（11）、loopback URL（12）、受管运行（13）、单卡 GPU1（14）、
补丁已应用（15）、EVAL_DATA 合法值（19）、adapter 目录（20），并以 `/health` 门禁后
exec 评测脚本。

## 5. 结果规范（results.json schema 摘要）

```json
{
  "kind": "p3-heldout-evaluation",
  "training": false,
  "decoding_backend": "hf-transformers-greedy",
  "adapter": {"path": "...", "adapter_model.safetensors": "sha256", "adapter_config.json": "sha256"},
  "data_files": {"path": "...", "sha256": "...", "hash_verified_against_manifest": true},
  "leakage": {"eval_questions": 32, "reference_questions": 8, "overlap": 0},
  "retriever_health": {"status": "ready", "vectors": 21015324},
  "metrics": {
    "overall": {"n": 32, "em": 0, "success": 0, "em_rate": 0.0, "success_rate": 0.0, "answer_compliance_rate": 0.0},
    "per_source": {"nq": {"n": 8, "em": 0, "success": 0, "em_rate": 0.0}},
    "action_stats": {"total_steps": 0, "invalid_actions": 0, "invalid_action_ratio": 0.0, "mixed_tag_steps": 0, "duplicate_tag_steps": 0},
    "retrieval": {"executed_searches": 0, "statuses": {}, "invalid_query_rate": 0.0, "api_error_rate": 0.0},
    "offline_rescore": {"matches": 0, "mismatches": 0}
  }
}
```

## 6. 验收门禁（评测 Run 后必须核对）

1. `metadata.env` 的 exit_code、物理 GPU、start/end；
2. stdout/stderr 无 traceback、无泄漏/门禁错误；
3. `results.json` 与 `episodes.jsonl` 原子存在、无 `.partial`、SHA256 记录；
4. 泄漏=0、health=21015324、hash 核对=true 已写入结果；
5. 退出后 GPU 回基线显存、无残留 PID/端口/Ray 进程（Retriever 用精确 tmux Ctrl-C 停止）。

## 7. 执行顺序与后续路线

```text
构建 heldout-32（CPU，已完成）→ CPU 测试（已完成，10 passed）
→ [GPU 阶段，另行批准] preflight → tmux 启动真实 Retriever → 6 个受管评测 Run
   （3 模型 × smoke-16 + 3 模型 × heldout-32，新 Run ID）→ 退出验收
→ 汇总对比表 + Wilson 置信区间
→ 有明确提升：verl/vLLM 原生评测复核 → 多 seed / 更大评测集
→ 无提升：排查 reward/prompt/rollout/group size/LR 配置，不直接跑 20 步
```
