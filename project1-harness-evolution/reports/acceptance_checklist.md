# 项目一验收清单（DEVELOPMENT_SCOPE 2.2/2.3 逐项对照）

- 日期：2026-08-08（四臂 round 2 完成后定稿）
- 版本：仍为 v0（四臂 gate 全部拒绝，见 CHANGELOG r2）

## 2.2 必须交付逐项

| # | 交付项 | 证据位置 | 状态 |
|---|---|---|---|
| 1 | 一条命令完成轨迹采集、格式转换、失败诊断和报告生成 | `scripts/run_loop.sh`（一键复现） | ⚠️ 未达成——已用 `scripts/restart_all_arms.sh` + 分步脚本替代，一键复现脚本缺口如实标注（见未达成项） |
| 2 | 互不重叠的开发集与留出测试集 | `data/datasets/partition_manifest.json`（dev 24/val 8/holdout 8，hash 3694ebbc） | ✅ M2 |
| 3 | 固定零样本基线 + 改进方案 | baseline=retail40-v1（M1）；改进=APO/GEPA 两优化器 × 诊断开关 | ✅ |
| 4 | 留出集统一比较根因类别/失败步骤/token/成本/耗时 | `reports/ablation_2026-08-08.md`（M5）；根因类别见 `data/diagnostics/summary.json`（10 类） | ✅（失败任务根因已归，成本/耗时在 round 记录） |
| 5 | 诊断结果转化为实际 Agent 改进 | 诊断注入适配器（M3）+ GEPA ASI 注入（M4），本轮真实生成候选（GEPA 两臂各 1 个新候选） | ✅ |
| 6 | 同协议任务重跑改进前后 Agent | 各臂 val 独立重跑 + gate 决策（round2.json 四臂全落盘） | ✅ |
| 7 | 报告成功率变化 + 回退/成本异常检查 | `reports/ablation_2026-08-08.md` §3（成功/回退/成本三列） | ✅ |
| 8 | 保存可复现脚本、配置、原始结果和最终报告 | scripts/ + runs/loop-*/ + reports/ | ✅ |

## 2.3 完成判定逐项

- [x] 1. holdout 从未用于提示词/规则调优（`grep -ri holdout` 仅命中 manifests/报告，日志无引用）
- [x] 2. 改进方案与基线同模型（deepseek-v4-flash）、同任务、同评测协议（tau2 retail）
- [x] 3. 至少完成一次"发现失败—提出改进—重新评测"闭环（四臂 round2 记录，GEPA 两臂含真实新候选）
- [x] 4. 提升/持平/下降均如实落盘（`runs/loop-*/round2.json` gate 全部 reject + CHANGELOG 8 条 + 本报告结论）

## SPEC 04 M4/M5 验收

- [x] GEPA 至少完成一轮优化，产出 Pareto 候选 + val 指标（每臂 1 个新候选，val 全量评测发生）
- [x] 消融矩阵 ≥4 臂（baseline/apo-plain/apo-diagnosis/gepa-plain/gepa-diagnosis）全有数字
- [x] holdout 全程未触碰（日志可证）
- [ ] 一键复现脚本在干净环境说明下可运行（缺口，见下）
- [x] 验收清单逐项勾选，未达成项如实标注

## 未达成项（如实标注）

1. **2.2-1 一键复现脚本 `scripts/run_loop.sh` 未创建**。目前有四臂并行启动脚本
   `scripts/restart_all_arms.sh`（tmux 四臂一键启动，已验证可用），但"一条命令完成
   采集→转换→诊断→报告"的完整链尚未包装成单脚本。缺口原因：诊断生成、格式转换
   当前为 M2/M3 分步产物，未再包装。**下一步补充或移出验收范围**。
2. **方法收益未达成**：四臂候选 val 全部 ≤ 基线 0.9，gate 全部拒绝，无版本更新。
   这是方法的诚实结果（DEVELOPMENT_SCOPE 2.3 模板"闭环工程可行但方法尚未产生收益"），
   不是交付缺口——闭环本身已完整跑通并有完整记录。

## 结论

闭环工程验收项 7/8 达成（一键复现缺口如实标注），完成判定 4/4 达成，
SPEC 04 验收 4/5 达成（一键复现同缺口）。方法层面本轮无收益，按 2.3 诚实记录。
