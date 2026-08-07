# 项目一验收清单（DEVELOPMENT_SCOPE 2.2/2.3 逐项对照）

- 日期：2026-08-07
- 状态：待四臂完成回填数字后定稿

## 2.2 必须交付逐项

| # | 交付项 | 证据位置 | 状态 |
|---|---|---|---|
| 1 | 一条命令完成轨迹采集、格式转换、失败诊断和报告生成 | `scripts/run_loop.sh`（一键复现） | ⏳ 待测 |
| 2 | 互不重叠的开发集与留出测试集 | `data/datasets/partition_manifest.json`（dev/val/holdout 划分） | ✅ M2 |
| 3 | 固定零样本基线 + 改进方案 | baseline=retail40-v1（M1）；改进=APO/GEPA 两优化器 × 诊断开关 | ✅ |
| 4 | 留出集统一比较根因类别/失败步骤/token/成本/耗时 | `reports/ablation_*.md`（M5） | ⏳ |
| 5 | 诊断结果转化为实际 Agent 改进 | 诊断注入适配器（M3）+ GEPA ASI 注入（M4） | ✅ |
| 6 | 同协议任务重跑改进前后 Agent | 各臂 val 独立重跑 + gate 决策 | ⏳ 运行中 |
| 7 | 报告成功率变化 + 回退/成本异常检查 | `reports/ablation_*.md` 副作用列 | ⏳ |
| 8 | 保存可复现脚本、配置、原始结果和最终报告 | scripts/ + runs/loop-*/ + reports/ | ⏳ |

## 2.3 完成判定逐项

- [ ] 1. holdout 从未用于提示词/规则调优（日志可证：`grep holdout` 无命中）
- [ ] 2. 改进方案与基线同模型（deepseek-v4-flash）、同任务、同评测协议（tau2 retail）
- [ ] 3. 至少完成一次"发现失败—提出改进—重新评测"闭环（各臂 round1 记录）
- [ ] 4. 提升/持平/下降均如实落盘（`runs/loop-*/round1.json` gate 决策 + 本报告结论）

## SPEC 04 M4/M5 验收

- [ ] GEPA 至少完成一轮优化，产出 Pareto 候选 + val 指标
- [ ] 消融矩阵 ≥4 臂（baseline/apo-plain/apo-diagnosis/gepa-diagnosis）全有数字
- [ ] holdout 全程未触碰（日志可证）
- [ ] 一键复现脚本在干净环境说明下可运行
- [ ] 验收清单逐项勾选，未达成项如实标注

## 未达成项（如有，如实标注）

（待定）
