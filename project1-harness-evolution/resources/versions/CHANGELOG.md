# Harness 资源版本 CHANGELOG（M3 gate 自动维护）
格式: 每行一条 JSON 记录；版本递增 v1, v2, ...

{"version": null, "round": "gepa-plain-r1", "decision": "reject", "reason": "成功率回退（0.875 < 0.880）", "metrics": {"candidate_success_rate": 0.875, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": null}}
{"version": null, "round": "gepa-diagnosis-r1", "decision": "reject", "reason": "成功率回退（0.875 < 0.880）", "metrics": {"candidate_success_rate": 0.875, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": null}}
{"version": null, "round": "apo-plain-r1", "decision": "reject", "reason": "成功率回退（0.875 < 0.880）", "metrics": {"candidate_success_rate": 0.875, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": null}}
{"version": null, "round": "apo-diagnosis-r1", "decision": "reject", "reason": "通过（成功率 +10.00pp，成本 $0.0000）", "metrics": {"candidate_success_rate": 1.0, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": null}}
{"version": null, "round": "gepa-plain-r2", "decision": "reject", "reason": "成功率回退（0.750 < 0.880）", "metrics": {"candidate_success_rate": 0.75, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": 0.058}}
{"version": null, "round": "gepa-diagnosis-r2", "decision": "reject", "reason": "成功率回退（0.875 < 0.880）", "metrics": {"candidate_success_rate": 0.875, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": 0.058}}
{"version": null, "round": "apo-plain-r2", "decision": "reject", "reason": "成功率回退（0.750 < 0.880）", "metrics": {"candidate_success_rate": 0.75, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": 0.058}}
{"version": null, "round": "apo-diagnosis-r2", "decision": "reject", "reason": "成功率回退（0.875 < 0.880）", "metrics": {"candidate_success_rate": 0.875, "baseline_success_rate": 0.9, "candidate_cost": null, "baseline_cost": 0.058}}
{"version": null, "round": "apo-plain-r3", "decision": "reject", "reason": "未产生收益（0.875 ≤ 基线 0.875，持平/无提升）", "metrics": {"candidate_success_rate": 0.875, "baseline_success_rate": 0.875, "candidate_cost": null, "baseline_cost": 0.058}}
{"version": null, "round": "apo-diagnosis-r3", "decision": "reject", "reason": "未产生收益（0.875 ≤ 基线 0.875，持平/无提升）", "metrics": {"candidate_success_rate": 0.875, "baseline_success_rate": 0.875, "candidate_cost": null, "baseline_cost": 0.058}}
