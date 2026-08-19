#!/usr/bin/env python3
"""Phase 4A diagnostic 2 analysis: 3 models x 4 conditions, paired.

Reads episodes.jsonl from the 12 counterfactual runs (runs/
p3-eval-counterfactual-{model}-{condition}-20260819a), joins by question, and
reports per model:
  - EM / compliance per condition (overall + per source)
  - paired McNemar exact two-sided p for real-top3 vs no-evidence,
    oracle vs no-evidence, real-top3 vs shuffled
  - oracle subset: EM on evidence-hit questions vs the same questions under
    no-evidence (paired); evidence-no-hit questions under oracle
  - delta table real-top3 - no-evidence, oracle - no-evidence,
    real-top3 - shuffled
Exploratory diagnostic, not confirmatory (prereg section 3).
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

RUNS_BASE = Path("/media/imc/data/project3-search-agent-rl/runs")
MODELS = ["base", "gs300", "searchr1"]
CONDITIONS = ["no-evidence", "real-top3", "oracle", "shuffled"]
RUN_SUFFIX = "20260819a"


def mcnemar_exact(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return min(1.0, 2.0 * p)


def load_run(model: str, condition: str) -> list[dict]:
    run = RUNS_BASE / f"p3-eval-counterfactual-{model}-{condition}-{RUN_SUFFIX}" / "episodes.jsonl"
    return [json.loads(l) for l in run.open()]


def main() -> None:
    eps: dict[str, dict[str, list[dict]]] = {}
    for m in MODELS:
        eps[m] = {}
        for c in CONDITIONS:
            eps[m][c] = load_run(m, c)
    n = len(eps[MODELS[0]][CONDITIONS[0]])
    assert all(len(eps[m][c]) == n for m in MODELS for c in CONDITIONS), "run sizes must match"

    def em(ep: dict) -> bool:
        return bool(ep["em_bool"])

    def comp(ep: dict) -> bool:
        return bool(ep["compliance"])

    out: dict = {"n": n}
    for m in MODELS:
        print(f"\n=== model {m} (n={n}) ===")
        # headline EM/compliance per condition
        row_em = [sum(1 for e in eps[m][c] if em(e)) for c in CONDITIONS]
        row_comp = [sum(1 for e in eps[m][c] if comp(e)) for c in CONDITIONS]
        print(f"{'condition':<14}" + "".join(f"{c:>14}" for c in CONDITIONS))
        print(f"{'EM':<14}" + "".join(f"{v}/{n} = {v/n:.1%}".rjust(14) for v in row_em))
        print(f"{'compliance':<14}" + "".join(f"{v/n:.1%}".rjust(14) for v in row_comp))

        # per-source EM
        print("per-source EM (no-evidence / real-top3 / oracle / shuffled):")
        for src in sorted({e["source"] for e in eps[m]["no-evidence"]}):
            vals = [f"{sum(1 for e in eps[m][c] if e['source'] == src and em(e))}/{sum(1 for e in eps[m][c] if e['source'] == src)}"
                    for c in CONDITIONS]
            print(f"  {src:<16} " + "  ".join(vals))

        # paired deltas + McNemar
        print("paired deltas (McNemar exact two-sided):")
        pairs = [
            ("real-top3 - no-evidence", "real-top3", "no-evidence"),
            ("oracle - no-evidence", "oracle", "no-evidence"),
            ("real-top3 - shuffled", "real-top3", "shuffled"),
        ]
        deltas = {}
        for label, c1, c0 in pairs:
            n01 = sum(1 for a, b in zip(eps[m][c1], eps[m][c0]) if em(a) and not em(b))
            n10 = sum(1 for a, b in zip(eps[m][c1], eps[m][c0]) if not em(a) and em(b))
            p = mcnemar_exact(n01, n10)
            deltas[label] = {"n01": n01, "n10": n10, "p": p}
            print(f"  {label:<26} 0->1: {n01}, 1->0: {n10}, p = {p:.4f}")

        # oracle subset: evidence-hit vs no-evidence paired; no-hit oracle EM
        oracle = eps[m]["oracle"]
        noev = eps[m]["no-evidence"]
        hit = [(o, ne) for o, ne in zip(oracle, noev) if o["oracle_hit"]]
        nohit = [o for o in oracle if not o["oracle_hit"]]
        hit_em = sum(1 for o, _ in hit if em(o))
        hit_noev_em = sum(1 for _, ne in hit if em(ne))
        n01 = sum(1 for o, ne in hit if em(o) and not em(ne))
        n10 = sum(1 for o, ne in hit if not em(o) and em(ne))
        p_hit = mcnemar_exact(n01, n10)
        print(f"oracle evidence-hit questions: {len(hit)}/{n}  "
              f"EM oracle={hit_em}/{len(hit)} = {hit_em/len(hit):.1%}  "
              f"no-evidence={hit_noev_em}/{len(hit)} = {hit_noev_em/len(hit):.1%}  "
              f"McNemar p={p_hit:.4f}")
        print(f"oracle evidence-NO-hit questions: {len(nohit)}/{n}, EM = {sum(1 for o in nohit if em(o))}/{len(nohit)}")
        out[m] = {
            "em": {c: row_em[i] for i, c in enumerate(CONDITIONS)},
            "compliance": {c: row_comp[i] for i, c in enumerate(CONDITIONS)},
            "deltas": deltas,
            "oracle": {
                "evidence_hit": len(hit),
                "em_on_hit": f"{hit_em}/{len(hit)}",
                "em_on_hit_no_evidence": f"{hit_noev_em}/{len(hit)}",
                "mcnemar_p_on_hit": p_hit,
                "em_on_nohit": f"{sum(1 for o in nohit if em(o))}/{len(nohit)}",
            },
        }

    out_path = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/p3_diag2_counterfactual_analysis_20260819.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
