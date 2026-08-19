#!/usr/bin/env python3
"""Phase 4A diagnostic 1: retriever ceiling audit (pure CPU).

Part A: for each dev256 question, use the raw question as query, retrieve
Top-3 and Top-10, and report HTTP success/timeout/errors, lexical answer
recall (normalized alias substring in concatenated docs) at Top-1/3/10,
per-source stats, score distribution and latency p50/p95/p99.
Part B: audit real search queries from existing dev256 episodes of the
available models (Base, official Search-R1; Step300 added once its dev256
episodes exist) and classify query-invalid / success-no-evidence /
success-with-evidence.

Lexical answer hit is an automated proxy metric only, not semantic relevance.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

RETRIEVE_URL = "http://127.0.0.1:18080/retrieve"
HEALTH_URL = "http://127.0.0.1:18080/health"
DATA_DIR = Path("/media/imc/data/project3-search-agent-rl")
DEV256 = DATA_DIR / "datasets/searchr1-official-confirm256-v1/heldout.parquet"
RUNS_DIR = DATA_DIR / "runs"
CKPT = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/diag1_partA_checkpoint.jsonl")
OUT_PATH = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/p3_diag1_retriever_audit_20260819.json")
MODEL_RUNS = {
    "Base": "p3-eval-official-confirm256-base3b-s0-20260815a",
    "SearchR1": "p3-eval-official-confirm256-official3b-s0-20260815a",
    "Step300": None,  # dev256 episodes generated in a later GPU1 managed run
}


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = re.sub(r"[\s\W_]+", "", s)  # fold whitespace and punctuation
    return s


def retrieve(query: str, topk: int, timeout: float = 180.0) -> dict:
    payload = json.dumps({"query": query, "topk": topk, "return_scores": True}).encode()
    req = Request(RETRIEVE_URL, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
        latency = time.monotonic() - t0
        docs = body["result"][0]
        return {
            "ok": True,
            "latency": latency,
            "docs": [
                {"id": d["document"]["id"], "contents": d["document"]["contents"], "score": d.get("score")}
                for d in docs
            ],
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "latency": time.monotonic() - t0, "docs": []}


def alias_hit_in_docs(answers: list[str], docs: list[dict]) -> bool:
    """Normalized answer alias substring in concatenated doc text."""
    norm_answers = [norm_text(a) for a in answers if a]
    if not norm_answers:
        return False
    blob = " ".join(d["contents"] for d in docs)
    nblob = norm_text(blob)
    return any(a and a in nblob for a in norm_answers)


def main() -> None:
    df = pd.read_parquet(DEV256)
    # resume: rows already retrieved in a previous run are reloaded from the
    # checkpoint file (one JSON object per line, in question order); a row is
    # complete when both top3_ok and top10_ok are present.
    ckpt_rows: dict[str, dict] = {}
    if CKPT.exists():
        for line in CKPT.open():
            r = json.loads(line)
            if "top3_ok" in r and "top10_ok" in r:
                # rows written before the recall1 fix lack these keys; recompute
                if "top3_recall1" not in r:
                    r["top3_recall1"] = alias_hit_in_docs(r["answers"], r.get("top3_docs", [])[:1])
                    r["top10_recall1"] = alias_hit_in_docs(r["answers"], r.get("top10_docs", [])[:1])
                ckpt_rows[r["question"]] = r
        print(f"checkpoint resume: {len(ckpt_rows)}/{len(df)} questions cached", flush=True)

    rows = []
    for _, r in df.iterrows():
        q = r["extra_info"]["question"]
        answers = list(r["reward_model"]["ground_truth"]["target"])
        src = r["data_source"]
        if q in ckpt_rows:
            rows.append(ckpt_rows[q])
            continue
        row = {"question": q, "answers": answers, "source": src}
        for topk in (3, 10):
            res = retrieve(q, topk)
            row[f"top{topk}_ok"] = res["ok"]
            row[f"top{topk}_err"] = res.get("error")
            row[f"top{topk}_lat"] = res.get("latency")
            row[f"top{topk}_docs"] = res.get("docs", [])
            docs = res.get("docs", [])
            row[f"top{topk}_score"] = [d["score"] for d in docs if d["score"] is not None]
            row[f"top{topk}_recall"] = alias_hit_in_docs(answers, docs)
            row[f"top{topk}_recall1"] = alias_hit_in_docs(answers, docs[:1])
        rows.append(row)
        with CKPT.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        sys.stdout.write(f"\r{len(rows)}/256 queries done")
    print()

    # ---- Part A summary ----
    print("\n=== A. question-as-query retrieval (n=256) ===")
    lat3 = [r["top3_lat"] for r in rows if r["top3_ok"]]
    lat10 = [r["top10_lat"] for r in rows if r["top10_ok"]]
    for topk, lat in (("top3", lat3), ("top10", lat10)):
        lat_sorted = sorted(lat)
        n = len(lat_sorted)
        p50 = lat_sorted[n // 2]
        p95 = lat_sorted[int(n * 0.95)]
        p99 = lat_sorted[int(n * 0.99)]
        print(f"  {topk}: ok={len(lat)}/256, latency p50={p50:.3f}s p95={p95:.3f}s p99={p99:.3f}s")
    errs = Counter(r["top10_err"] for r in rows if not r["top10_ok"])
    print(f"  errors (top10): {dict(errs) or 'none'}")

    print("\n  lexical answer recall (all / per source):")
    src_order = ["nq", "hotpotqa", "popqa", "2wikimultihopqa", "triviaqa", "musique", "bamboogle"]
    hdr = f"{'source':<14}" + "".join(f"top{k}".rjust(8) for k in (1, 3, 10))
    print("  " + hdr)
    for src in src_order:
        sub = [r for r in rows if r["source"] == src]
        n = len(sub)
        cells = [f"{sum(1 for r in sub if r[f'top{k}_recall' if k > 1 else 'top3_recall1'])}/{n}".rjust(8) for k in (1, 3, 10)]
        print(f"  {src:<14}" + "".join(cells))
    n = len(rows)
    all_cells = [f"{sum(1 for r in rows if r['top3_recall1'])}/{n}".rjust(8),
                 f"{sum(1 for r in rows if r['top3_recall'])}/{n}".rjust(8),
                 f"{sum(1 for r in rows if r['top10_recall'])}/{n}".rjust(8)]
    print(f"  {'ALL':<14}" + "".join(all_cells))

    scores = [s for r in rows for s in r["top10_score"]]
    if scores:
        ss = sorted(scores)
        quant = lambda p: ss[min(len(ss) - 1, int(p * len(ss)))]
        print(f"  doc score distribution (top10): min={quant(0):.4f} p50={quant(0.5):.4f} "
              f"p95={quant(0.95):.4f} max={quant(1):.4f}")

    # ---- Part B: real query audit ----
    print("\n=== B. real search query audit (dev256 episodes) ===")
    summary = {}
    for model, run in MODEL_RUNS.items():
        if run is None:
            continue
        ep_path = RUNS_DIR / run / "episodes.jsonl"
        recs = [json.loads(l) for l in ep_path.open()]
        statuses = Counter()
        classes = Counter()
        queries = []
        search_steps = []  # one entry per executed search step (for len + sampling)
        for r in recs:
            ep_answers = list(r.get("answers") or [])
            for s in r.get("steps", []):
                if not s.get("executed_search"):
                    continue
                search_steps.append(s)
                info = s.get("info", {})
                status = info.get("retrieval", {}).get("status") if isinstance(info.get("retrieval"), dict) else None
                statuses[status or "no-status"] += 1
                # extract query from postprocessed_action <search>...</search>
                pa = info.get("postprocessed_action", "")
                m = re.search(r"<search>(.*?)</search>", pa, re.DOTALL)
                q = m.group(1).strip() if m else ""
                queries.append(q)
                if status in ("invalid_query", "api_error", "no_results") or not q:
                    classes["invalid_or_failed"] += 1
                elif status == "success":
                    res = retrieve(q, 10)
                    if not res["ok"]:
                        classes["retriever_error_on_recheck"] += 1
                    else:
                        hit = alias_hit_in_docs(ep_answers, res["docs"])
                        classes["success_with_evidence" if hit else "success_no_evidence"] += 1
                else:
                    classes["other"] += 1
        summary[model] = {
            "search_steps": len(search_steps),
            "statuses": dict(statuses),
            "classes": dict(classes),
            "empty_query": sum(1 for q in queries if not q),
        }
        print(f"  {model}: search_steps={len(search_steps)}, statuses={dict(statuses)}")
        print(f"          classes={dict(classes)}, empty_query={summary[model]['empty_query']}")

    # build a clean JSON without the giant doc payloads
    out = {
        "dev256_sha": "ffebf468e756a673da267f5830cfc67f2e9c4dc44ec41c979a389c1efebfff60",
        "part_a": {
        "n": n,
        "ok_top3": len(lat3),
        "ok_top10": len(lat10),
        "latency_top3": {"p50": sorted(lat3)[len(lat3) // 2], "p95": sorted(lat3)[int(len(lat3) * 0.95)], "p99": sorted(lat3)[int(len(lat3) * 0.99)]},
        "latency_top10": {"p50": sorted(lat10)[len(lat10) // 2], "p95": sorted(lat10)[int(len(lat10) * 0.95)], "p99": sorted(lat10)[int(len(lat10) * 0.99)]},
        "recall": {
            "top1": sum(1 for r in rows if r["top3_recall1"]),
            "top3": sum(1 for r in rows if r["top3_recall"]),
            "top10": sum(1 for r in rows if r["top10_recall"]),
        },
        "recall_by_source": {
            src: {
                "n": len([r for r in rows if r["source"] == src]),
                "top1": sum(1 for r in rows if r["source"] == src and r["top3_recall1"]),
                "top3": sum(1 for r in rows if r["source"] == src and r["top3_recall"]),
                "top10": sum(1 for r in rows if r["source"] == src and r["top10_recall"]),
            } for src in src_order
        },
        "scores_top10": {"min": quant(0), "p50": quant(0.5), "p95": quant(0.95), "max": quant(1)} if scores else None,
        },
    }
    out["part_b"] = summary
    out_path = Path("/home/imc/yzy/agent/project3-search-agent-rl/gates/p3_diag1_retriever_audit_20260819.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
