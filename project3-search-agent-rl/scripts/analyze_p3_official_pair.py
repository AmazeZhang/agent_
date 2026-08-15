#!/usr/bin/env python3
"""Preregistered paired analysis: official Search-R1 3B GRPO vs Qwen2.5-3B Base
(official-loose line).

Implements exactly the statistics and the three-way decision fixed in
docs/P3_OFFICIAL_CHECKPOINT_PREREG_2026-08-15.md sections 3-4:

- 主指标 EM (env reward >= 1.0; official-loose, format_score=0.1 not counted)
- 主检验 exact two-sided McNemar (binomial on discordant direction)
- Wilson 95% CIs for both EM rates
- discordant detail (0->1, 1->0, 1->1, 0->0) with per-question records
- 次要指标 (descriptive only): per-source EM, executed searches, retrieval
  statuses, error-observation steps, format-scored (0.1) episodes, answer
  compliance, byte-identical / normalized edit distance between runs
- 判定: PASS / FAIL-TO-OBSERVE / INCONCLUSIVE per prereg section 4

Usage:
  analyze_p3_official_pair.py <base_run_id> <official_run_id> [--out-stem NAME]

Writes analysis/official-line/<stem>.md and .json. CPU-only.
"""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/media/imc/data/project3-search-agent-rl/runs")
OUT_DIR = PROJECT_ROOT / "analysis" / "official-line"


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return (centre - half, centre + half)


def em(episode: dict) -> int:
    return 1 if episode["reward"] >= 1.0 else 0


def step_pair_distance(base_steps: list[dict], official_steps: list[dict]) -> dict:
    n_steps = min(len(base_steps), len(official_steps))
    identical = 0
    lev_total = 0.0
    for i in range(n_steps):
        a = base_steps[i]["raw_action"]
        b = official_steps[i]["raw_action"]
        if a == b:
            identical += 1
        else:
            m, n = len(a), len(b)
            prev = list(range(n + 1))
            for i in range(1, m + 1):
                cur = [i] + [0] * n
                for j in range(1, n + 1):
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
                prev = cur
            lev_total += prev[n] / max(m, n)
    return {"aligned_steps": n_steps, "byte_identical": identical, "mean_norm_lev": lev_total / max(n_steps, 1)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_run", help="run id of the Qwen2.5-3B Base evaluation")
    parser.add_argument("official_run", help="run id of the official GRPO checkpoint evaluation")
    parser.add_argument("--out-stem", default="p3_official_pair_2026-08-15")
    args = parser.parse_args()

    BASE_RUN, OFFICIAL_RUN = args.base_run, args.official_run
    BASE = json.loads((RUNS / BASE_RUN / "results.json").read_text())
    OFF = json.loads((RUNS / OFFICIAL_RUN / "results.json").read_text())
    BASE_EPISODES = [json.loads(l) for l in (RUNS / BASE_RUN / "episodes.jsonl").open()]
    OFF_EPISODES = [json.loads(l) for l in (RUNS / OFFICIAL_RUN / "episodes.jsonl").open()]

    assert len(BASE_EPISODES) == len(OFF_EPISODES) == 256
    assert [e["question"] for e in BASE_EPISODES] == [e["question"] for e in OFF_EPISODES]
    assert BASE["line"] == OFF["line"] == "official-loose"

    b = [em(e) for e in BASE_EPISODES]
    o = [em(e) for e in OFF_EPISODES]
    b11 = sum(1 for x, y in zip(b, o) if x and y)
    b10 = sum(1 for x, y in zip(b, o) if x and not y)  # base won, official lost
    b01 = sum(1 for x, y in zip(b, o) if not x and y)  # base lost, official won
    b00 = sum(1 for x, y in zip(b, o) if not x and not y)
    n_disc = b10 + b01
    mcnemar_p = binomtest(b10, n_disc, 0.5, alternative="two-sided").pvalue if n_disc else 1.0

    base_ci = wilson_ci(sum(b), len(b))
    off_ci = wilson_ci(sum(o), len(o))

    # Three-way decision per prereg section 4 (fixed before evaluation).
    if mcnemar_p < 0.05 and sum(o) > sum(b):
        verdict = "PASS"
        reason = f"p={mcnemar_p:.4f} < 0.05 and official EM ({sum(o)}) > Base EM ({sum(b)}): environment can observe the Search-R1 effect"
    elif mcnemar_p < 0.05:
        verdict = "FAIL-TO-OBSERVE"
        reason = f"p={mcnemar_p:.4f} < 0.05 but official EM ({sum(o)}) <= Base EM ({sum(b)}): strong evidence the environment does not reproduce the official effect"
    else:
        verdict = "INCONCLUSIVE"
        reason = f"p={mcnemar_p:.4f} >= 0.05: no significant difference observed; explicitly NOT evidence of environment inconsistency (prereg section 4 rule 3)"

    sources = sorted({e["source"] for e in BASE_EPISODES})
    per_source = {}
    for source in sources:
        idx = [i for i, e in enumerate(BASE_EPISODES) if e["source"] == source]
        per_source[source] = {
            "n": len(idx),
            "base_em": sum(b[i] for i in idx),
            "official_em": sum(o[i] for i in idx),
        }

    secondary = {
        "base": {
            "searches": BASE["metrics"]["retrieval"]["executed_searches"],
            "retrieval_statuses": BASE["metrics"]["retrieval"]["statuses"],
            "error_observation_steps": BASE["metrics"]["action_stats"]["error_observation_steps"],
            "format_scored_episodes": BASE["metrics"]["action_stats"]["format_scored_episodes"],
            "total_steps": BASE["metrics"]["action_stats"]["total_steps"],
            "answer_compliance_rate": BASE["metrics"]["overall"]["answer_compliance_rate"],
        },
        "official": {
            "searches": OFF["metrics"]["retrieval"]["executed_searches"],
            "retrieval_statuses": OFF["metrics"]["retrieval"]["statuses"],
            "error_observation_steps": OFF["metrics"]["action_stats"]["error_observation_steps"],
            "format_scored_episodes": OFF["metrics"]["action_stats"]["format_scored_episodes"],
            "total_steps": OFF["metrics"]["action_stats"]["total_steps"],
            "answer_compliance_rate": OFF["metrics"]["overall"]["answer_compliance_rate"],
        },
    }

    dist = [step_pair_distance(b["steps"], o["steps"]) for b, o in zip(BASE_EPISODES, OFF_EPISODES)]
    gen_stats = {
        "episodes_with_identical_steps": sum(1 for d in dist if d["byte_identical"] == d["aligned_steps"] and d["aligned_steps"] > 0),
        "mean_norm_lev_over_episodes": sum(d["mean_norm_lev"] for d in dist) / len(dist),
        "total_aligned_steps": sum(d["aligned_steps"] for d in dist),
        "total_byte_identical_steps": sum(d["byte_identical"] for d in dist),
    }

    discordant = [
        {
            "index": i,
            "source": BASE_EPISODES[i]["source"],
            "question": BASE_EPISODES[i]["question"],
            "direction": "base_won_official_lost" if b[i] and not o[i] else "base_lost_official_won",
            "base_reward": BASE_EPISODES[i]["reward"],
            "official_reward": OFF_EPISODES[i]["reward"],
            "base_first_action": BASE_EPISODES[i]["steps"][0]["raw_action"][:160] if BASE_EPISODES[i]["steps"] else "",
            "official_first_action": OFF_EPISODES[i]["steps"][0]["raw_action"][:160] if OFF_EPISODES[i]["steps"] else "",
        }
        for i in range(len(b))
        if b[i] != o[i]
    ]

    result = {
        "schema_version": 1,
        "kind": "p3-official-checkpoint-preregistered-paired-comparison",
        "prereg": "docs/P3_OFFICIAL_CHECKPOINT_PREREG_2026-08-15.md",
        "line": "official-loose",
        "base_run": BASE_RUN,
        "official_run": OFFICIAL_RUN,
        "data_sha256": BASE["data_files"]["sha256"],
        "scripts": {
            "base_runtime_script_sha256": BASE["runtime_script_sha256"],
            "official_runtime_script_sha256": OFF["runtime_script_sha256"],
        },
        "primary": {
            "base_em": sum(b),
            "official_em": sum(o),
            "n": len(b),
            "base_em_rate": sum(b) / len(b),
            "official_em_rate": sum(o) / len(o),
            "base_wilson95_ci": base_ci,
            "official_wilson95_ci": off_ci,
            "discordant": {"b10_base_won": b10, "b01_official_won": b01, "b11_both": b11, "b00_neither": b00},
            "exact_twosided_mcnemar_p": mcnemar_p,
        },
        "decision": {
            "rule": "PASS if p<0.05 and official>base; FAIL-TO-OBSERVE if p<0.05 and official<=base; INCONCLUSIVE if p>=0.05 (prereg section 4)",
            "verdict": verdict,
            "reason": reason,
        },
        "per_source": per_source,
        "secondary": secondary,
        "generation_comparison": gen_stats,
        "discordant_detail": discordant,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / (args.out_stem + ".md")
    json_path = OUT_DIR / (args.out_stem + ".json")
    md.write_text(
        f"""# P3 官方模型验证：官方 Search-R1 3B GRPO vs Qwen2.5-3B Base（官方宽松语义线）

预注册：`docs/P3_OFFICIAL_CHECKPOINT_PREREG_2026-08-15.md`（先于任何评测提交）
数据：`searchr1-official-confirm256-v1` heldout.parquet（SHA `{result['data_sha256'][:16]}…`，
排除 dev32、旧 confirm256、训练集；构建确定重建一致）
语义：official-loose（raw action 直达 skyrl SearchEnv，无投影无惩罚，format_score=0.1）
后端：vLLM 0.8.5 V0 引擎 greedy；tokenizer 固定 Qwen2.5-3B Base（两模型输入 byte-identical）
运行：{BASE_RUN} / {OFFICIAL_RUN}（受管，GPU1，cleanup `physical_gpu=1 compute_processes=none`）

## 主指标（EM = env reward ≥ 1.0）

| 模型 | EM | 率 | Wilson 95% CI |
|---|---|---|---|
| Qwen2.5-3B Base | {sum(b)}/{len(b)} | {sum(b)/len(b):.4f} | [{base_ci[0]:.4f}, {base_ci[1]:.4f}] |
| 官方 Search-R1 3B GRPO | {sum(o)}/{len(o)} | {sum(o)/len(o):.4f} | [{off_ci[0]:.4f}, {off_ci[1]:.4f}] |

## 配对明细（discordant）

- 1→1（双方都对）：{b11}
- 0→0（双方都错）：{b00}
- 1→0（Base 对、官方错）：{b10}
- 0→1（Base 错、官方对）：{b01}

**精确双侧 McNemar p = {mcnemar_p:.6f}**（{b10}:{b01}，discordant n={n_disc}）

## 判定（预注册第 4 节，三档）

- **{verdict}**：{reason}
- PASS → 批准进入 3B 复现训练阶段（第二阶段门禁另行预注册）；
  FAIL-TO-OBSERVE → 停止训练计划，先诊断环境（检索质量对比优先）；
  INCONCLUSIVE → 不作为环境不一致证据，结合次要指标与后续诊断定方向。

## 次要指标（仅描述性）

| 指标 | Base | 官方 |
|---|---|---|
| 检索次数（成功/其余） | {secondary['base']['searches']}（{json.dumps(secondary['base']['retrieval_statuses'])}） | {secondary['official']['searches']}（{json.dumps(secondary['official']['retrieval_statuses'])}） |
| error observation 步 | {secondary['base']['error_observation_steps']}/{secondary['base']['total_steps']} | {secondary['official']['error_observation_steps']}/{secondary['official']['total_steps']} |
| format_scored（0.1）episode | {secondary['base']['format_scored_episodes']} | {secondary['official']['format_scored_episodes']} |
| answer_compliance rate | {secondary['base']['answer_compliance_rate']:.3f} | {secondary['official']['answer_compliance_rate']:.3f} |

分源 EM：{json.dumps(per_source, ensure_ascii=False)}

生成对比：{gen_stats['episodes_with_identical_steps']} 个 episode 逐字节一致，
全部对齐步 {gen_stats['total_byte_identical_steps']}/{gen_stats['total_aligned_steps']}，
平均归一化编辑距离 {gen_stats['mean_norm_lev_over_episodes']:.4f}。

## 声明边界（预注册第 7 节）

- 单 seed、greedy、官方宽松语义下的单次验证；判定"我们的评测链路能否观察官方
  训练效果"，不是官方模型在官方环境上的复现成绩；
- 本实验数字不与严格线或论文数字直接对照；
- discordant 逐题明细见随附 JSON（`{json_path.name}`）。
"""
    )
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote {md}")
    print(f"wrote {json_path}")
    print(f"EM base={sum(b)} official={sum(o)}  p={mcnemar_p:.6f}  verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
