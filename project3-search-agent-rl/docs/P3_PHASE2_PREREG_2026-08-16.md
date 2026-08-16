# P3 第二阶段预注册：官方宽松线 3B GRPO 复现训练（Step 0–50–100–300）

**预注册时间**：2026-08-16（本文件先于任何 Step 50+ 评测与 final 评测提交；
不批准启动 GPU 训练——Step 0–50 训练须另行单独批准）
**实验线**：official-loose（`docs/P3_EXPERIMENT_LINES_2026-08-15.md` §1 左列，
`run_p3_grpo_official_exp.sh` formal profile；`run_p3_eval_vllm_official.py` 官方宽松
语义评测）
**依据**：第一阶段官方模型验证 PASS（`P3_OFFICIAL_CHECKPOINT_PREREG_2026-08-15.md`，
McNemar p=0.0357，环境能观察 Search-R1 效应）→ 用户批准进入 3B 复现训练冻结阶段。
本预注册落实设计文档 §7 门禁结构（用户审阅拍板：Step 50/100 为**开发门禁**不设统计
判据；final-confirm512 为**唯一确认性检验**）。

## 0. 训练线冻结状态（本预注册固定引用）

- 训练入口：`scripts/run_p3_grpo_official_exp.sh` **formal** profile，fail-closed
  （PROJECT3_SEGMENT_STOP_STEP=50|100|300 必需；stop=50 禁止 resume；stop=100 必须
  resume 自 global_step_50；stop=300 必须 resume 自 global_step_100；其他组合报错
  exit 30–42）。
- patch 0006（`patches/0006-segment-stop-step-decoupled-schedule-horizon.patch`）：
  schedule horizon 恒 300 步（LR scheduler 长度），segment stop 为独立停止点；
  到达时完成当前 step + 保存 checkpoint（DataLoader/Optimizer/Scheduler/RNG）+ 正常
  return。三段 LR 曲线逐点一致（`tests/test_scheduler_continuity.py` 6/6 通过）。
- 三段 resolved config：`configs/p3_formal_segments_2026-08-16.json`；
  **training-invariant SHA256（三段必须相同）`2cc743a3cedbd957518717f7d47b0f1c3fe060abb07d92fe84b71cd270339674`**；
  full SHA：0-50 `4a472c90…`、50-100 `910f216c…`、100-300 `758326c6…`（run dir 为
  冻结占位，实际 run 由 run_managed 生成，per-run config_fp 记录于各 run 日志）。
- 训练不变量：lr=1e-6、warmup_steps=85（== int(0.285×300)）、kl low_var_kl 0.001、
  entropy 0、GRPO group_n=5、batch 66→330 samples、0.60/64/offloads=true、
  save_freq=50、seed 1234/1234、max_steps=2、history_length=2、topk=3、
  timeout=180、projection=official（宽松）。上游 pin `20bd331b…`，patches 0001–0006。

## 1. 假设（训练最终 checkpoint vs 基线）

- **H1**：Step 300 最终 checkpoint（官方宽松线 3B GRPO，全参数）在
  final-confirm512 上的 EM 高于 Qwen2.5-3B Base。
- **H0**：两者 EM 无差异（配对 discordant 方向无偏）。
- 训练成功与否由 final-confirm512 判定；dev 集数字不作终审（设计文档 §11）。

## 2. 固定条件（评测运行前全部固定，运行后不得更改）

| 项 | 值 |
|---|---|
| 基线（Step 0） | **复用**受管 Base 官方线 dev/最终结果：`p3-eval-official-confirm256-base3b-s0-20260815a`（dev official-confirm256-v1 EM **20/256**=7.81%，Wilson [5.11%, 11.76%]）；final-confirm512 上的 Base 结果在 final 评测窗口以同一 eval 入口受管补测（受管、SHA 在案），**不因中期结果修改** |
| dev 集 | `datasets/searchr1-official-confirm256-v1/heldout.parquet`（SHA `ffebf468e756…`，256 行，domain `searchr1-p3-official-confirm-v1`） |
| final 集（唯一确认性） | `datasets/searchr1-final-confirm512/heldout.parquet`（SHA `94b39266c2d9c54a55b4471e90daa493ab083a889d8f23510dadd8194b304ecc`，512 行，domain `searchr1-p3-final-confirm-v1`；manifest 泄漏=0：排除上游 train、smoke train/test、dev32、confirm256、official-confirm256-v1；配额 nq 128/hotpotqa 128/popqa 64/2wiki 64/triviaqa 64/musique 32/bamboogle 32） |
| 评测入口 | `scripts/run_p3_eval_vllm_official.py` + wrapper `run_p3_eval_vllm_official.sh`（官方宽松语义；Step 50/100/300 checkpoint 为 FSDP 分片 → 合并为 HF 权重后评测，转换细节执行阶段确认并记录，不改变本表判定规则） |
| 引擎/语义/检索 | 同第一阶段（vLLM 0.8.5.post1 V1=0 greedy、tokenizer 固定 Base、SearchMultiProcessEnv seed=0 group_n=1、max_steps=2/history_length=2/topk=3/timeout=180、真实 Wiki-18 21,015,324 向量 health 门禁、run_managed 受管 GPU1） |
| 运行 | 受管；Step 50/100 停训窗口内评测（训练与评测不并发） |

## 3. 门禁结构（预先固定）

1. **Step 50 开发门禁**（不设统计判据）：训练健康（正常退出、SEGMENT_STOP 标记、
   checkpoint global_step_50 完整、显存/CPU 回基线）+ 行为变化（搜索协议遵守率、
   平均检索次数 vs Base 方向性）+ dev EM 方向性（相对 Step 0）。通过 → 继续
   Step 50–100；异常 → 停训诊断（不自动调整参数/降级/重试）。
2. **Step 100 开发门禁**（不设统计判据）：趋势一致性（Step 50/100 方向一致或
   Step 100 更强）→ 继续 100–300；任一阶段异常 → 停训诊断。
3. **Step 300 final 确认性检验（唯一）**：final-confirm512 配对评测，McNemar
   三档判定（§4）；dev 集数字不作终审。**不因任何中期结果修改本判据。**

## 4. final 判定规则（三档，预先固定，评测后不得更改）

- **主指标**：EM（env reward ≥ 1.0，与第一阶段口径一致）。
- **主检验**：配对 McNemar 精确检验（双侧，discordant 方向精确二项 p）；Wilson
  95% CI（各自 EM 率）；discordant 明细（0→1、1→0、1→1、0→0）。

1. **PASS**：p < 0.05 **且** Step300 EM 严格 > Base EM
   → 官方宽松语义线在 3B 全参数上观察到确认性提升；批准后续行动（另行决策）。
2. **FAIL-TO-OBSERVE**：p < 0.05 **且** Step300 EM 严格 ≤ Base EM（显著负向或持平）
   → 强证据：该训练配置在本环境下无可观察正向效应；停止并诊断（配置/环境/数据
   方向），不直接重跑。
3. **INCONCLUSIVE**：p ≥ 0.05（无论方向）→ 无法确认显著差异；明确不作为训练失败
   或成功证据；结合 dev 门禁记录与次要指标形成方向，不直接批准/否决后续训练。
4. **报告义务**：无论结论，必须报告两侧 Wilson CI、McNemar p、discordant 明细、
   dev 门禁各阶段记录（含 run id/SHA）。
5. **设备/门禁失败**：任一 run 未通过退出验收 → 仅替换失败 run 重新受管运行，
   不重抽数据、不改规则。

## 5. 禁止事项（训练/评测前/中/后）

- 评测前不得查看 final-confirm512 题目（512 题）或任何模型输出；final 集封存
  （manifest+SHA 在案，构建 commit 见 §6），不评测、不挑题；
- 不得以任何中期结果调参、换模型、改语义/引擎/数据/门禁判据；
- 不得以 total_training_steps<300 或 warmup≠85 运行正式段（fail-closed 强制）；
- 不得把 smoke/resume 工程验证 checkpoint 接入正式训练（正式 Step 0 必须从
  Qwen2.5-3B Base 重启）；
- 若发现脚本/数据 bug：修复后**重新预注册并重跑全部**（不修补性改判据）。

## 6. final 数据 SHA 与构建 commit

- final-confirm512 heldout.parquet SHA256：`94b39266c2d9c54a55b4471e90daa493ab083a889d8f23510dadd8194b304ecc`
- manifest：`datasets/searchr1-final-confirm512/manifest.json`（输入 SHA/配额/泄漏统计；
  重建确定性已验证：同参数重建 → 同 SHA）
- 构建器：`scripts/build_p3_heldout_eval.py`（--domain `searchr1-p3-final-confirm-v1`
  --total 512 --extra-exclusions dev32/confirm256/official-confirm256-v1）
- 构建代码提交：`723cd78`（`freeze(p3): formal 3-segment freeze + patch 0006 +
  prereg`，含构建器、冻结 JSON、test_scheduler_continuity.py、patch 0006 与本预注册
  文件本体）。
