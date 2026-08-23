#!/usr/bin/env python3
"""P3 Search-aware clean v2: strict per-question paired statistics (CPU-only).

Pairs the per-question binary outcomes (EM/won) of confirm-256 eval runs by
question_id (0..255, sorted, identical ID space, same data SHA) and computes:

  - contingency: 1->1, 0->0, control-correct/v2-wrong, control-wrong/v2-correct
  - net gain (b-c convention: b = control-correct/v2-wrong, c = control-wrong/v2-correct)
  - exact two-sided McNemar p (binomial on discordant pairs, min(b,c) tail doubled)
  - Wilson 95% CI for both sides
  - per-source discordant counts
  - per-question behavior breakdown (search/answer/correct/invalid/max-steps/mixed-round)

Pairs: v2_step5 vs step0 | v2_step5 vs grpo10 | v2_step5 vs gigpo10 | grpo10 vs step0.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/p3_v2_paired_stats.py \
      --out gates/p3_v2_paired_stats_20260823.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

RUNS = {
    "v2_step5": "p3-eval-v2-behavior-gs5-confirm256-20260823a",
    "step0": "p3-eval-upstream-clean-step0-instruct-confirm256-20260820a",
    "grpo10": "p3-eval-upstream-clean-grpo10-confirm256-20260820a",
    "gigpo10": "p3-eval-upstream-clean-gigpo10-confirm256-20260820b",
}
PAIRS = [("v2_step5", "step0"), ("v2_step5", "grpo10"), ("v2_step5", "gigpo10"), ("grpo10", "step0")]
ENV_MAX_STEPS = 4


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def mcnemar_exact_two_sided(b, c):
    """Exact two-sided McNemar p on discordant pair counts b, c (Bin(n, 0.5))."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(p, 1.0)


def load_run(name, data_root):
    base = Path(data_root) / "project3-search-agent-rl" / "runs" / RUNS[name]
    rows = [json.loads(l) for l in (base / "episodes.jsonl").open()]
    rj = json.load((base / "results.json").open())
    return rows, rj


def is_search_step(step):
    info = step.get("info", {})
    return info.get("tool_name") == "search"


def is_answer_step(step):
    # answer rounds are non-tool actions: tool_name is None, the projected
    # action carries the <answer>...</answer> tag (search_projection semantics)
    info = step.get("info", {})
    return str(info.get("postprocessed_action", "")).startswith("<answer>")


def analyze_episode(ep):
    steps = ep["steps"]
    searched = any(is_search_step(s) for s in steps)
    answered_env = any(is_answer_step(s) for s in steps)
    # results.json "answer compliance" uses the offline extraction: an
    # <answer> tag anywhere in the concatenated raw actions (may be a draft
    # inside a mixed round whose projection chose search). 232 vs 189 for the
    # v2 Step5 run is a real behavior signal (drafted-but-never-committed).
    has_answer_offline = bool(ep.get("offline", {}).get("has_answer", False))
    invalid_q = any(
        is_search_step(s)
        and (s.get("info", {}).get("retrieval_failed", False)
             or bool((s.get("info", {}).get("retrieval") or {}).get("api_request_error")))
        for s in steps
    )
    max_steps_exhausted = (not answered_env) and len(steps) >= ENV_MAX_STEPS
    mixed_rounds = [
        (i, s.get("raw_action", "")) for i, s in enumerate(steps)
        if "<search>" in (s.get("raw_action") or "") and "<answer>" in (s.get("raw_action") or "")
    ]
    return {
        "qid": ep["question_id"], "source": ep["source"], "won": bool(ep["won"]),
        "searched": searched, "answered_env": answered_env, "has_answer_offline": has_answer_offline,
        "invalid_query": invalid_q,
        "max_steps_exhausted": max_steps_exhausted, "n_steps": len(steps),
        "mixed_rounds": len(mixed_rounds),
        "projection_wins": {"search": sum(1 for _, a in mixed_rounds if "<search>" in a and a.find("<search>") < a.find("<answer>")),
                            "answer": sum(1 for _, a in mixed_rounds if "<answer>" in a and a.find("<answer>") < a.find("<search>"))},
        "drafted_answer_never_committed": has_answer_offline and not answered_env,
        "termination_offline": "answered" if has_answer_offline else (
            "max_steps_exhausted" if (not answered_env) and len(steps) >= ENV_MAX_STEPS else (
                "ended_no_answer_committed" if not answered_env else "answered_env")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/media/imc/data")
    ap.add_argument("--out", default="gates/p3_v2_paired_stats_20260823.json")
    args = ap.parse_args()

    data = {}
    for name in RUNS:
        rows, rj = load_run(name, args.data_root)
        eps = [analyze_episode(e) for e in rows]
        by_qid = {e["qid"]: e for e in eps}
        data[name] = {
            "n": len(rows),
            "data_sha256": rj["data_files"]["sha256"],
            "qids": [e["qid"] for e in eps],
            "by_qid": by_qid,
        }
        assert len(by_qid) == 256 and len(set(by_qid)) == 256, f"{name}: bad qid space"
        assert data[name]["qids"] == sorted(data[name]["qids"]), f"{name}: order not sorted"

    # --- consistency gates ---
    sha = {name: data[name]["data_sha256"] for name in RUNS}
    assert len(set(sha.values())) == 1, f"data SHA mismatch: {sha}"
    qid_sets = {name: set(data[name]["by_qid"]) for name in RUNS}
    assert all(s == qid_sets["v2_step5"] for s in qid_sets.values()), "qid set mismatch"

    report = {
        "kind": "p3-v2-paired-stats",
        "created_at": "2026-08-23",
        "data_sha256": sha["v2_step5"],
        "qid_space": "0..255 sorted, unique per run",
        "n_per_run": 256,
        "pairs": {},
        "behavior": {},
        "integrity": {
            "id_sets_equal": True,
            "data_sha_identical": True,
            "order_sorted_identical": True,
            "runs": {name: {"run_id": RUNS[name], "data_sha256": sha[name]} for name in RUNS},
        },
    }

    # --- behavior breakdown (all runs, question level) ---
    for name in RUNS:
        eps = list(data[name]["by_qid"].values())
        searched = [e for e in eps if e["searched"]]
        answered_off = [e for e in eps if e["has_answer_offline"]]
        answered_env = [e for e in eps if e["answered_env"]]
        sa_off = [e for e in searched if e["has_answer_offline"]]
        sc = [e for e in searched if e["won"]]
        nsc = [e for e in eps if (not e["searched"]) and e["won"]]
        sba_off = [e for e in searched if not e["has_answer_offline"]]
        no_a_off = [e for e in eps if not e["has_answer_offline"]]
        report["behavior"][name] = {
            "em": sum(e["won"] for e in eps),
            "searched": len(searched),
            "answered_offline": len(answered_off),
            "answered_env_committed": len(answered_env),
            "searched_and_answered_offline": len(sa_off),
            "searched_and_correct": len(sc),
            "no_search_and_correct": len(nsc),
            "searched_but_no_answer_offline": len(sba_off),
            "no_search_and_no_answer_offline": sum(1 for e in eps if (not e["searched"]) and (not e["has_answer_offline"])),
            "drafted_answer_never_committed": sum(e["drafted_answer_never_committed"] for e in eps),
            "max_steps_exhausted": sum(e["max_steps_exhausted"] for e in eps),
            "invalid_query_questions": sum(e["invalid_query"] for e in eps),
            "mixed_round_actions": sum(e["mixed_rounds"] for e in eps),
            "projection_search_first": sum(e["projection_wins"]["search"] for e in eps),
            "projection_answer_first": sum(e["projection_wins"]["answer"] for e in eps),
            "search_rate": round(len(searched) / 256, 4),
            "answer_compliance_offline": round(len(answered_off) / 256, 4),
            "search_to_answer_offline": round(len(sa_off) / len(searched), 4) if searched else None,
            "search_to_correct": round(len(sc) / len(searched), 4) if searched else None,
            "no_search_to_correct": round(len(nsc) / max(1, 256 - len(searched)), 4),
            "avg_steps": round(sum(e["n_steps"] for e in eps) / 256, 4),
            "unanswered_termination_offline": {
                "max_steps_exhausted": sum(e["termination_offline"] == "max_steps_exhausted" for e in no_a_off),
                "ended_no_answer_committed": sum(e["termination_offline"] == "ended_no_answer_committed" for e in no_a_off),
                "searched_but_no_answer_offline": len(sba_off),
                "of_which_invalid_query": sum(e["invalid_query"] for e in sba_off),
                "of_which_drafted_but_never_committed": sum(e["drafted_answer_never_committed"] for e in no_a_off),
            },
        }
        # 24-unanswered detail (v2_step5 only, full per-question classification)
        if name == "v2_step5":
            detail = []
            for e in no_a_off:
                detail.append({
                    "qid": e["qid"], "source": e["source"],
                    "searched": e["searched"], "invalid_query": e["invalid_query"],
                    "max_steps_exhausted": e["max_steps_exhausted"],
                    "n_steps": e["n_steps"],
                    "termination_offline": e["termination_offline"],
                    "drafted_never_committed": e["drafted_answer_never_committed"],
                })
            report["behavior"][name]["unanswered_questions_detail"] = detail

    # --- paired McNemar ---
    for a_name, b_name in PAIRS:
        A, B = data[a_name]["by_qid"], data[b_name]["by_qid"]
        cells = {"a11": 0, "a00": 0, "b10": 0, "c01": 0}  # b10: B correct, A wrong; c01: A correct, B wrong
        per_source = {}
        for qid in sorted(A):
            a_won, b_won = A[qid]["won"], B[qid]["won"]
            src = A[qid]["source"]
            key = "a11" if (a_won and b_won) else "a00" if (not a_won and not b_won) else ("b10" if b_won else "c01")
            cells[key] += 1
            per_source.setdefault(src, {"a11": 0, "a00": 0, "b10": 0, "c01": 0})[key] += 1
        n_disc = cells["b10"] + cells["c01"]
        report["pairs"][f"{b_name}->{a_name}"] = {
            "control": b_name, "v2_or_target": a_name,
            "contingency": cells,
            "net_gain": cells["c01"] - cells["b10"],
            "discordant_n": n_disc,
            "mcnemar_exact_two_sided_p": round(mcnemar_exact_two_sided(cells["b10"], cells["c01"]), 6),
            "wilson95": {
                b_name: wilson_ci(sum(1 for e in B.values() if e["won"]), 256),
                a_name: wilson_ci(sum(1 for e in A.values() if e["won"]), 256),
            },
            "per_source_discordants": {
                src: {"control_correct_target_wrong": v["b10"], "control_wrong_target_correct": v["c01"], "discordant": v["b10"] + v["c01"]}
                for src, v in sorted(per_source.items())
            },
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    tmp.replace(out)

    # --- console summary ---
    print("== integrity ==")
    print(f"  data_sha256={sha['v2_step5']}  qid 0..255 sorted unique across all runs")
    print("\n== behavior (question level, n=256) ==")
    for name in RUNS:
        b = report["behavior"][name]
        print(f"  {name:9s} em={b['em']:3d} searched={b['searched']:3d} ans_off={b['answered_offline']:3d} "
              f"sc={b['searched_and_correct']:3d} nsc={b['no_search_and_correct']:3d} "
              f"sba_off={b['searched_but_no_answer_offline']:3d} maxst={b['max_steps_exhausted']:3d} invalid={b['invalid_query_questions']:3d} "
              f"mixed={b['mixed_round_actions']}")
    print("\n== paired McNemar (target vs control) ==")
    for key, p in report["pairs"].items():
        c = p["contingency"]
        print(f"  {key:18s} 1->1={c['a11']:3d} 0->0={c['a00']:3d} ctlR/tgtW={c['b10']:3d} ctlW/tgtR={c['c01']:3d} "
              f"net={p['net_gain']:+d}  McNemar exact 2-sided p={p['mcnemar_exact_two_sided_p']:.6f}")
    print("\n== v2_step5 unanswered termination ==")
    ut = report["behavior"]["v2_step5"]["unanswered_termination_offline"]
    print(f"  {ut}")


if __name__ == "__main__":
    main()
