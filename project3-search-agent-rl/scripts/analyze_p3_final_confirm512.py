#!/usr/bin/env python3
"""Analyze the final-confirm512 blind evaluation (three models, 512 paired items).

Reads episodes.jsonl from the three eval runs and computes:
  - per-model headline table (EM, compliance, search behaviour, reward dist)
  - paired Base vs Step300 McNemar exact two-sided p (confirmatory judgement)
  - Wilson 95% CI per model EM
  - discordant pairs (Base->Step300 0->1 / 1->0) with sources
  - exploratory mechanism indicators per the dated addendum
  - per-source EM
The official Search-R1 checkpoint is descriptive reference only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

BASE_DIR = Path(
    "/media/imc/data/project3-search-agent-rl/runs"
)
RUNS = {
    "Base": "p3-eval-final-confirm512-base-20260819a",
    "Step300": "p3-eval-final-confirm512-gs300-20260819a",
    "SearchR1": "p3-eval-final-confirm512-searchr1-20260819a",
}


def load_episodes(run_name: str) -> list[dict]:
    p = BASE_DIR / RUNS[run_name] / "episodes.jsonl"
    return [json.loads(l) for l in p.open()]


def episode_metrics(ep: dict) -> dict:
    steps = ep.get("steps", [])
    search_steps = [s for s in steps if s.get("executed_search")]
    statuses = [
        s.get("info", {}).get("retrieval", {}).get("status")
        for s in steps
        if s.get("executed_search") and isinstance(s.get("info", {}).get("retrieval"), dict)
    ]
    error_steps = [s for s in steps if s.get("error_observation")]
    return {
        "em": bool(ep.get("won") or ep.get("reward") == 1.0),
        "compliance": ep.get("reward", 0.0) >= 0.1,
        "reward": ep.get("reward", 0.0),
        "source": ep.get("source", "?"),
        "n_search": len(search_steps),
        "n_steps": len(steps),
        "one_step": len(steps) == 1,
        "statuses": statuses,
        "n_error": len(error_steps),
        "has_error_obs": any(s.get("error_observation") for s in steps),
    }


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(n01: int, n10: int) -> float:
    """Exact two-sided McNemar p = 2*P(X <= min(n01,n10)), X~Bin(n,0.5)."""
    n = n01 + n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * p)


def main() -> None:
    data = {name: [episode_metrics(e) for e in load_episodes(name)] for name in RUNS}
    n = len(data["Base"])
    assert all(len(v) == n for v in data.values()), "episode counts must match"
    print(f"paired episodes: {n}")

    # 1. headline per-model table
    print("\n=== per-model headline (n=512) ===")
    print(f"{'metric':<28}" + "".join(f"{name:>12}" for name in RUNS))
    for key, fmt in [
        ("EM", "{:>12}"),
        ("compliance", "{:>12.1%}"),
        ("search steps", "{:>12}"),
        ("1-step done", "{:>12.1%}"),
        ("error obs", "{:>12}"),
        ("steps total", "{:>12}"),
    ]:
        if key == "EM":
            vals = [sum(1 for m in data[name] if m["em"]) for name in RUNS]
            cells = "".join(f"{v}/{n} = {v/n:.1%}".rjust(12) for v in vals)
        elif key == "search steps":
            vals = [sum(m["n_search"] for m in data[name]) for name in RUNS]
            cells = "".join(f"{v}".rjust(12) for v in vals)
        elif key == "1-step done":
            vals = [sum(1 for m in data[name] if m["one_step"]) for name in RUNS]
            cells = "".join(f"{v/n:.1%}".rjust(12) for v in vals)
        elif key == "error obs":
            vals = [sum(m["n_error"] for m in data[name]) for name in RUNS]
            cells = "".join(f"{v}".rjust(12) for v in vals)
        elif key == "compliance":
            vals = [sum(1 for m in data[name] if m["compliance"]) for name in RUNS]
            cells = "".join(f"{v/n:.1%}".rjust(12) for v in vals)
        else:  # steps total
            vals = [sum(m["n_steps"] for m in data[name]) for name in RUNS]
            cells = "".join(f"{v}".rjust(12) for v in vals)
        print(f"{key:<28}{cells}")

    # status distribution (search steps only)
    print("\nsearch-step retrieval status distribution:")
    for name in RUNS:
        statuses = [s for m in data[name] for s in m["statuses"]]
        from collections import Counter
        c = Counter(statuses)
        print(f"  {name:<10} total={len(statuses):<4} " +
              " ".join(f"{k}:{v}" for k, v in sorted(c.items())))

    # reward distribution
    print("\nreward distribution (0.0 / 0.1 / 1.0):")
    for name in RUNS:
        c = Counter(m["reward"] for m in data[name])
        print(f"  {name:<10} {c.get(0.0,0)} / {c.get(0.1,0)} / {c.get(1.0,0)}")

    # 2. confirmatory: paired Base vs Step300
    base = data["Base"]
    step = data["Step300"]
    n01 = sum(1 for b, s in zip(base, step) if not b["em"] and s["em"])
    n10 = sum(1 for b, s in zip(base, step) if b["em"] and not s["em"])
    p_val = mcnemar_exact(n01, n10)
    print("\n=== CONFIRMATORY: Base vs Step300 (paired EM / McNemar) ===")
    print(f"0->1: {n01}, 1->0: {n10}, McNemar exact two-sided p = {p_val:.4f}")
    em_b = sum(1 for m in base if m["em"])
    em_s = sum(1 for m in step if m["em"])
    print(f"EM Base {em_b}/512 = {em_b/512:.2%}, Step300 {em_s}/512 = {em_s/512:.2%}")
    lo_b, hi_b = wilson_ci(em_b, n)
    lo_s, hi_s = wilson_ci(em_s, n)
    print(f"Wilson 95% CI Base:   [{lo_b:.1%}, {hi_b:.1%}]")
    print(f"Wilson 95% CI Step300:[{lo_s:.1%}, {hi_s:.1%}]")
    diff = em_s / n - em_b / n
    se = math.sqrt((em_s / n) * (1 - em_s / n) / n + (em_b / n) * (1 - em_b / n) / n)
    print(f"EM diff (Step300-Base): {diff:+.2%} (Wald 95% CI [{diff-1.96*se:+.1%}, {diff+1.96*se:+.1%}])")
    if em_s > em_b and p_val < 0.05:
        verdict = "PASS"
    elif p_val >= 0.05:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL-TO-OBSERVE"
    print(f"VERDICT: {verdict}")

    # 3. discordant pairs by source
    print("\ndiscordant pairs by source:")
    from collections import Counter
    up = Counter(s["source"] for b, s in zip(base, step) if not b["em"] and s["em"])
    down = Counter(b["source"] for b, s in zip(base, step) if b["em"] and not s["em"])
    print(f"  0->1: {dict(up)}")
    print(f"  1->0: {dict(down)}")

    # 4. per-source EM
    print("\nper-source EM (Base / Step300 / SearchR1):")
    for src in sorted({m["source"] for m in base}):
        vals = []
        for name in RUNS:
            ms = [m for m in data[name] if m["source"] == src]
            vals.append(f"{sum(1 for m in ms if m['em'])}/{len(ms)}")
        print(f"  {src:<16} " + "  ".join(f"{name}:{v}" for name, v in zip(RUNS, vals)))

    # 5. mechanism indicators (exploratory, per addendum)
    print("\n=== exploratory mechanism indicators (addendum) ===")
    for name in RUNS:
        ms = data[name]
        search_correct = sum(1 for m in ms if m["n_search"] > 0 and m["em"])
        search_wrong = sum(1 for m in ms if m["n_search"] > 0 and not m["em"])
        nosearch_correct = sum(1 for m in ms if m["n_search"] == 0 and m["em"])
        per_q = sum(m["n_search"] for m in ms) / len(ms)
        print(f"  {name:<10} search->correct: {search_correct}, search->wrong: {search_wrong}, "
              f"no-search->correct: {nosearch_correct}, search/q: {per_q:.3f}, "
              f"1-step rate: {sum(1 for m in ms if m['one_step'])/len(ms):.1%}")

    # 6. offline answer check coverage
    print("\noffline final-answer coverage (has_answer):")
    for name in RUNS:
        eps = load_episodes(name)
        c = sum(1 for e in eps if e.get("offline", {}).get("has_answer"))
        print(f"  {name:<10} {c}/512")

    with open(Path(__file__).parent.parent / "gates/final_confirm512_analysis_20260819a.json", "w") as f:
        json.dump({
            "n": n,
            "em": {name: sum(1 for m in data[name] if m["em"]) for name in RUNS},
            "mcnemar_01_10": [n01, n10],
            "mcnemar_p": p_val,
            "verdict": verdict,
            "wilson_ci": {"Base": [lo_b, hi_b], "Step300": [lo_s, hi_s]},
        }, f, indent=2)
    print("\nanalysis JSON written to gates/final_confirm512_analysis_20260819a.json")


if __name__ == "__main__":
    main()
