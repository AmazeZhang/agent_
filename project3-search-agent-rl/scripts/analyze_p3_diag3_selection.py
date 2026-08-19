#!/usr/bin/env python3
"""Phase 4A diagnostic 3: search selection bias (pure CPU).

Inputs:
  - dev256 episodes of Base / official Search-R1 / Step300
    (runs/p3-eval-official-confirm256-{base3b,official3b,gs300}-*)
  - diag2 oracle runs (runs/p3-eval-counterfactual-{model}-oracle-*)

Outputs:
  - search vs no-search question subsets: source composition, question token
    length (Base tokenizer), multi-hop share (2wikimultihopqa/hotpotqa/musique)
  - Base direct-answer EM on the official model's searched subset
  - official model's searched subset EM under oracle evidence (diag2 slice)
  - attribution material for search->correct = 0 (selection bias vs retriever
    failure vs evidence-usage failure); not a confirmatory claim (prereg s4).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

DATA_ROOT = Path("/media/imc/data/project3-search-agent-rl")
RUNS_BASE = DATA_ROOT / "runs"
TOKENIZER = DATA_ROOT / "models/Qwen2.5-3B"
MULTI_HOP = {"2wikimultihopqa", "hotpotqa", "musique"}

MODEL_EVAL_RUNS = {
    "Base": "p3-eval-official-confirm256-base3b-s0-20260815a",
    "SearchR1": "p3-eval-official-confirm256-official3b-s0-20260815a",
    "Step300": "p3-eval-official-confirm256-gs300-20260819a",  # GPU1 managed backfill
}
COUNTERFACTUAL_RUN = "p3-eval-counterfactual-{model}-oracle-20260819a"
OUT_PATH = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/p3_diag3_selection_20260819.json")


def load_episodes(run: str) -> list[dict]:
    p = RUNS_BASE / run / "episodes.jsonl"
    return [json.loads(l) for l in p.open()]


def searched_questions(eps: list[dict]) -> set[str]:
    out = set()
    for e in eps:
        if any(s.get("executed_search") for s in e.get("steps", [])):
            out.add(e["question"])
    return out


def em_bool(ep: dict) -> bool:
    """EM from the formal-line episode: env reward >= 1.0 (exact match)."""
    return float(ep.get("reward", 0.0)) >= 1.0


def main() -> None:
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    episodes = {m: load_episodes(r) for m, r in MODEL_EVAL_RUNS.items()}
    eps_by_q = {m: {e["question"]: e for e in eps} for m, eps in episodes.items()}
    all_q = list(eps_by_q["Base"])
    print(f"dev256 paired: {len(all_q)}")

    out: dict = {}
    for m in MODEL_EVAL_RUNS:
        eps = episodes[m]
        s_q = searched_questions(eps)
        n_s = len(s_q)
        n_ns = len(all_q) - n_s
        src_s = Counter(e["source"] for e in eps if e["question"] in s_q)
        src_ns = Counter(e["source"] for e in eps if e["question"] not in s_q)
        len_s = [len(tok.encode(q)) for q in s_q]
        len_ns = [len(tok.encode(q)) for q in all_q if q not in s_q]
        multi_s = sum(1 for e in eps if e["question"] in s_q and e["source"] in MULTI_HOP)
        multi_ns = sum(1 for e in eps if e["question"] not in s_q and e["source"] in MULTI_HOP)
        em_s = sum(1 for e in eps if e["question"] in s_q and em_bool(e))
        em_ns = sum(1 for e in eps if e["question"] not in s_q and em_bool(e))
        row = {
            "n_search": n_s,
            "n_nosearch": n_ns,
            "source_search": dict(src_s),
            "source_nosearch": dict(src_ns),
            "q_len_search_mean": float(sum(len_s) / len(len_s)) if len_s else None,
            "q_len_nosearch_mean": float(sum(len_ns) / len(len_ns)) if len_ns else None,
            "multihop_share_search": multi_s / n_s if n_s else None,
            "multihop_share_nosearch": multi_ns / n_ns if n_ns else None,
            "em_search": f"{em_s}/{n_s}",
            "em_nosearch": f"{em_ns}/{n_ns}",
        }
        out[m] = row
        print(f"\n=== {m}: search={n_s}, no-search={n_ns} ===")
        print(f"  source search: {dict(src_s)}")
        print(f"  source no-search: {dict(src_ns)}")
        print(f"  q-len mean search={row['q_len_search_mean']:.1f} no-search={row['q_len_nosearch_mean']:.1f}")
        print(f"  multi-hop share search={row['multihop_share_search']:.2f} no-search={row['multihop_share_nosearch']:.2f}")
        print(f"  EM search={em_s}/{n_s}, EM no-search={em_ns}/{n_ns}")

    # Base direct-answer EM on the SearchR1 searched subset (selection bias)
    sr1_searched = searched_questions(episodes["SearchR1"])
    base_on_sr1 = [eps_by_q["Base"][q] for q in sr1_searched if q in eps_by_q["Base"]]
    em = sum(1 for e in base_on_sr1 if em_bool(e))
    out["base_em_on_searchr1_searched"] = f"{em}/{len(base_on_sr1)}"
    print(f"\nBase direct EM on SearchR1-searched subset: {em}/{len(base_on_sr1)}")
    # and on Step300 searched subset
    s3_searched = searched_questions(episodes["Step300"])
    base_on_s3 = [eps_by_q["Base"][q] for q in s3_searched if q in eps_by_q["Base"]]
    em3 = sum(1 for e in base_on_s3 if em_bool(e))
    out["base_em_on_step300_searched"] = f"{em3}/{len(base_on_s3)}"
    print(f"Base direct EM on Step300-searched subset: {em3}/{len(base_on_s3)}")

    # SearchR1 searched subset under oracle evidence (diag2 slice)
    oracle = {m: {e["question"]: e for e in load_episodes(COUNTERFACTUAL_RUN.format(model=m))} for m in
              ("base", "gs300", "searchr1")}
    for m in ("searchr1", "gs300", "base"):
        model_key = {"searchr1": "SearchR1", "gs300": "Step300", "base": "Base"}[m]
        searched = searched_questions(episodes[model_key])
        hit = [oracle[m][q] for q in searched if q in oracle[m] and oracle[m][q]["oracle_hit"]]
        em_o = sum(1 for e in hit if e["em_bool"])
        out[f"oracle_em_on_{m}_searched_subset"] = f"{em_o}/{len(hit)}"
        print(f"oracle-condition EM on {model_key}-searched subset (evidence-hit qs): {em_o}/{len(hit)}")

    out["note"] = "Exploratory selection-bias material (prereg s4): searched questions being harder is selection bias, not proof search is useless."
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {OUT_PATH}")


if __name__ == "__main__":
    main()
