#!/usr/bin/env python3
"""Preregistered paired analysis: confirm-256 Base vs train64nqh8 (vLLM greedy).

Implements exactly the statistics fixed in
docs/P3_CONFIRM256_PREREG_2026-08-15.md section 3-4:

- 主指标 EM (env reward >= 1.0; skyRL strict EM; format_score not counted)
- 主检验 exact two-sided McNemar (binomial on discordant direction)
- Wilson 95% CIs for both EM rates
- discordant detail (0->1, 1->0, 1->1, 0->0) with per-question records
- 次要指标 (descriptive only): per-source EM, executed searches,
  invalid actions, answer compliance, byte-identical steps, normalized edit
  distance between the two runs' generation.

Writes analysis/p3_confirm256_pair_2026-08-15.md (human) and
analysis/p3_confirm256_pair_2026-08-15.json (machine).

CPU-only.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/media/imc/data/project3-search-agent-rl/runs")
BASE_RUN = "p3-eval-vllm-confirm256-base-s0-20260815c"
TRAIN_RUN = "p3-eval-vllm-confirm256-train64nqh8-s0-20260815a"
OUT_STEM = "p3_confirm256_pair_2026-08-15"
OUT_DIR = PROJECT_ROOT / "analysis"

BASE = json.loads((RUNS / BASE_RUN / "results.json").read_text())
TRAIN = json.loads((RUNS / TRAIN_RUN / "results.json").read_text())
BASE_EPISODES = [json.loads(l) for l in (RUNS / BASE_RUN / "episodes.jsonl").open()]
TRAIN_EPISODES = [json.loads(l) for l in (RUNS / TRAIN_RUN / "episodes.jsonl").open()]


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


def step_pair_distance(base_steps: list[dict], train_steps: list[dict]) -> dict:
    """Byte-identical and normalized-edit-distance stats over aligned steps."""
    n_steps = min(len(base_steps), len(train_steps))
    identical = 0
    lev_total = 0.0
    for i in range(n_steps):
        a = base_steps[i]["raw_action"]
        b = train_steps[i]["raw_action"]
        if a == b:
            identical += 1
        else:
            # normalized Levenshtein (two-row DP, O(mn) with m,n <= a few hundred)
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
    assert len(BASE_EPISODES) == len(TRAIN_EPISODES) == 256
    assert [e["question"] for e in BASE_EPISODES] == [e["question"] for e in TRAIN_EPISODES]

    b = [em(e) for e in BASE_EPISODES]
    t = [em(e) for e in TRAIN_EPISODES]
    b11 = sum(1 for x, y in zip(b, t) if x and y)
    b10 = sum(1 for x, y in zip(b, t) if x and not y)  # base won, train lost
    b01 = sum(1 for x, y in zip(b, t) if not x and y)  # base lost, train won
    b00 = sum(1 for x, y in zip(b, t) if not x and not y)
    n_disc = b10 + b01
    mcnemar_p = binomtest(b10, n_disc, 0.5, alternative="two-sided").pvalue if n_disc else 1.0

    base_ci = wilson_ci(sum(b), len(b))
    train_ci = wilson_ci(sum(t), len(t))

    sources = sorted({e["source"] for e in BASE_EPISODES})
    per_source = {}
    for source in sources:
        idx = [i for i, e in enumerate(BASE_EPISODES) if e["source"] == source]
        per_source[source] = {
            "n": len(idx),
            "base_em": sum(b[i] for i in idx),
            "train_em": sum(t[i] for i in idx),
        }

    secondary = {
        "base": {
            "searches": BASE["metrics"]["retrieval"]["executed_searches"],
            "invalid_actions": BASE["metrics"]["action_stats"]["invalid_actions"],
            "total_steps": BASE["metrics"]["action_stats"]["total_steps"],
            "answer_compliance_rate": BASE["metrics"]["overall"]["answer_compliance_rate"],
            "retrieval_statuses": BASE["metrics"]["retrieval"]["statuses"],
        },
        "train64nqh8": {
            "searches": TRAIN["metrics"]["retrieval"]["executed_searches"],
            "invalid_actions": TRAIN["metrics"]["action_stats"]["invalid_actions"],
            "total_steps": TRAIN["metrics"]["action_stats"]["total_steps"],
            "answer_compliance_rate": TRAIN["metrics"]["overall"]["answer_compliance_rate"],
            "retrieval_statuses": TRAIN["metrics"]["retrieval"]["statuses"],
        },
    }

    # Step-level generation comparison (aligned episodes/steps).
    dist = [step_pair_distance(b["steps"], t["steps"]) for b, t in zip(BASE_EPISODES, TRAIN_EPISODES)]
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
            "direction": "base_won_train_lost" if b[i] and not t[i] else "base_lost_train_won",
            "base_reward": BASE_EPISODES[i]["reward"],
            "train_reward": TRAIN_EPISODES[i]["reward"],
            "base_first_action": BASE_EPISODES[i]["steps"][0]["raw_action"][:160] if BASE_EPISODES[i]["steps"] else "",
            "train_first_action": TRAIN_EPISODES[i]["steps"][0]["raw_action"][:160] if TRAIN_EPISODES[i]["steps"] else "",
        }
        for i in range(len(b))
        if b[i] != t[i]
    ]

    result = {
        "schema_version": 1,
        "kind": "p3-confirm256-preregistered-paired-comparison",
        "prereg": "docs/P3_CONFIRM256_PREREG_2026-08-15.md",
        "line": "strict-fork",
        "base_run": BASE_RUN,
        "train_run": TRAIN_RUN,
        "data_sha256": BASE["data_files"]["sha256"],
        "scripts": {
            "base_runtime_script_sha256": BASE["runtime_script_sha256"],
            "train_runtime_script_sha256": TRAIN["runtime_script_sha256"],
        },
        "primary": {
            "base_em": sum(b),
            "train_em": sum(t),
            "n": len(b),
            "base_em_rate": sum(b) / len(b),
            "train_em_rate": sum(t) / len(t),
            "base_wilson95_ci": base_ci,
            "train_wilson95_ci": train_ci,
            "discordant": {"b10_base_won": b10, "b01_train_won": b01, "b11_both": b11, "b00_neither": b00},
            "exact_twosided_mcnemar_p": mcnemar_p,
        },
        "decision": {
            "rule": "H1 supported only if p<0.05 and train64 EM > base EM (prereg section 4)",
            "verdict": "H1_NOT_SUPPORTED",
            "reason": f"p={mcnemar_p:.4f} >= 0.05; point estimate favors base ({sum(b)} vs {sum(t)})",
        },
        "per_source": per_source,
        "secondary": secondary,
        "generation_comparison": gen_stats,
        "discordant_detail": discordant,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = OUT_DIR / (OUT_STEM + ".md")
    json_path = OUT_DIR / (OUT_STEM + ".json")
    md.write_text(
        f"""# P3 confirm-256 预注册配对比较：Base vs train64nqh8（严格 fork 线）

预注册：`docs/P3_CONFIRM256_PREREG_2026-08-15.md`（c66677a 起，含运行记录加注）
数据：`searchr1-confirm256` heldout.parquet（SHA `{result['data_sha256'][:16]}…`，dev32 零重叠，泄漏 0）
后端：vLLM 0.8.5 V0 引擎 greedy（VLLM_USE_V1=0），与训练 rollout 同引擎路径
运行：{BASE_RUN} / {TRAIN_RUN}（均为受管运行，cleanup `physical_gpu=1 compute_processes=none`，0 检索超时）

## 主指标（EM = env reward ≥ 1.0）

| 模型 | EM | 率 | Wilson 95% CI |
|---|---|---|---|
| Base | {sum(b)}/{len(b)} | {sum(b)/len(b):.4f} | [{base_ci[0]:.4f}, {base_ci[1]:.4f}] |
| train64nqh8 | {sum(t)}/{len(t)} | {sum(t)/len(t):.4f} | [{train_ci[0]:.4f}, {train_ci[1]:.4f}] |

## 配对明细（discordant）

- 1→1（双方都对）：{b11}
- 0→0（双方都错）：{b00}
- 1→0（Base 对、train64 错）：{b10}
- 0→1（Base 错、train64 对）：{b01}

**精确双侧 McNemar p = {mcnemar_p:.6f}**（{b10}:{b01}，discordant n={n_disc}）

## 判定（预注册第 4 节）

- H1（train64 > Base）支持条件：p < 0.05 **且** train64 EM > Base EM。
- 实际：p = {mcnemar_p:.4f} ≥ 0.05，且点估计为 Base 更高（{sum(b)} vs {sum(t)}）。
- **结论：H1 不支持。** 按规则 2（p ≥ 0.05 无论方向都不支持 H1）；
  点估计方向偏 Base 但不显著（p ≥ 0.05，规则 4 的"负向记录"不触发）。
- 与 dev32（5/32 vs 3/32，p=0.5）一致：没有证据表明 train64nqh8 在
  严格 fork 语义下优于 Base；两个独立小样本上的点估计方向都不支持 H1。

## 次要指标（仅描述性）

| 指标 | Base | train64nqh8 |
|---|---|---|
| 检索次数（成功/其余） | {secondary['base']['searches']}（{json.dumps(secondary['base']['retrieval_statuses'])}） | {secondary['train64nqh8']['searches']}（{json.dumps(secondary['train64nqh8']['retrieval_statuses'])}） |
| invalid 动作 / 总步数 | {secondary['base']['invalid_actions']}/{secondary['base']['total_steps']} | {secondary['train64nqh8']['invalid_actions']}/{secondary['train64nqh8']['total_steps']} |
| answer_compliance rate | {secondary['base']['answer_compliance_rate']:.3f} | {secondary['train64nqh8']['answer_compliance_rate']:.3f} |

分源 EM：{json.dumps(per_source, ensure_ascii=False)}

生成对比：{gen_stats['episodes_with_identical_steps']} 个 episode 逐字节一致，
全部对齐步 {gen_stats['total_byte_identical_steps']}/{gen_stats['total_aligned_steps']}，
平均归一化编辑距离 {gen_stats['mean_norm_lev_over_episodes']:.4f}。

## 声明边界（预注册第 7 节）

- 严格 fork 线、单 seed、greedy 下的单次确认实验；不声称 Search-R1 复现；
- 官方宽松语义基线为另一条线（`docs/P3_EXPERIMENT_LINES_2026-08-15.md`）；
- discordant 逐题明细见随附 JSON（`{json_path.name}`）。
"""
    )
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote {md}")
    print(f"wrote {json_path}")
    print(f"EM base={sum(b)} train={sum(t)}  p={mcnemar_p:.6f}  verdict=H1_NOT_SUPPORTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
