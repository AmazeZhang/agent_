#!/usr/bin/env python3
"""P3 counterfactual retrieval eval: strict per-question paired statistics.

Pairs the per-question outcomes of the SAME merged model (v2 Step5) on the
SAME official-confirm256-v1 set, greedy temperature=0, GPU1, by question_id
(0..255, sorted, identical ID space, same data SHA):

  real       -- p3-eval-v2-behavior-gs5-confirm256-20260823a (untouched evidence)
  shuffled   -- evidence replaced by REAL docs of question (i+17) mod 256
  no-evidence-- fixed neutral envelope, no retriever call

Outputs per comparison (real vs each counterfactual):
  - contingency 1->1 / 0->0 / real-correct/cf-wrong / real-wrong/cf-correct
  - net gain, exact two-sided McNemar p, Wilson 95% CIs
  - real-only correct vs counterfactual-only correct
  - how many of the real searched-and-correct (69) flip under the condition
  - evidence-change-but-answer-unchanged (identical committed <answer> text)
  - search rate / search->answer / search->correct / no-search->correct /
    searched-and-correct abs / compliance / avg steps / unanswered reasons
  - per-source discordants
  - integrity: ID sets, data SHA, order, pre-registration mapping SHA check

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/p3_v2_counterfactual_stats.py \
      --out gates/p3_v2_counterfactual_stats_20260823.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

RUNS = {
    "real": "p3-eval-v2-behavior-gs5-confirm256-20260823a",
    "shuffled": "p3-eval-v2-behavior-gs5-confirm256-shuffled-20260823a",
    "no-evidence": "p3-eval-v2-behavior-gs5-confirm256-noevidence-20260823a",
}
CONDITIONS = ("shuffled", "no-evidence")
ENV_MAX_STEPS = 4
EXPECTED_MAPPING_SHA = "93363b6730795ca4608cc2e88212126f366f0f261239014cbac5226f3166c480"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def mcnemar_exact_two_sided(b, c):
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
    return step.get("info", {}).get("tool_name") == "search"


def is_answer_step(step):
    info = step.get("info", {})
    return str(info.get("postprocessed_action", "")).startswith("<answer>")


def committed_answer_text(ep):
    """The <answer>...</answer> text of the env-committed answer round, if any."""
    for step in ep["steps"]:
        info = step.get("info", {})
        if str(info.get("postprocessed_action", "")).startswith("<answer>"):
            act = str(info.get("postprocessed_action", ""))
            if "<answer>" in act:
                return act[act.index("<answer>") + len("<answer>"):]
    return None


def analyze_episode(ep):
    steps = ep["steps"]
    searched = any(is_search_step(s) for s in steps)
    answered_env = any(is_answer_step(s) for s in steps)
    has_answer_offline = bool(ep.get("offline", {}).get("has_answer", False))
    invalid_q = any(
        is_search_step(s)
        and (s.get("info", {}).get("retrieval_failed", False)
             or bool((s.get("info", {}).get("retrieval") or {}).get("api_request_error")))
        for s in steps
    )
    # evidence-change: the docs returned differ from what the REAL run saw on
    # the same question (only meaningful in the counterfactual runs)
    doc_ids = [
        sorted((s.get("info", {}).get("retrieval") or {}).get("document_ids") or [])
        for s in steps if is_search_step(s)
    ]
    return {
        "qid": ep["question_id"], "source": ep["source"], "won": bool(ep["won"]),
        "searched": searched, "answered_env": answered_env,
        "has_answer_offline": has_answer_offline, "invalid_query": invalid_q,
        "n_steps": len(steps),
        "max_steps_exhausted": (not answered_env) and len(steps) >= ENV_MAX_STEPS,
        "termination": ("answered" if has_answer_offline else (
            "max_steps_exhausted" if (not answered_env) and len(steps) >= ENV_MAX_STEPS else
            ("ended_no_answer_committed" if not answered_env else "answered_env"))),
        "answer_text": committed_answer_text(ep),
        "search_doc_ids": doc_ids,
        "avg_steps": len(steps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/media/imc/data")
    ap.add_argument("--out", default="gates/p3_v2_counterfactual_stats_20260823.json")
    args = ap.parse_args()

    data = {}
    for name in RUNS:
        rows, rj = load_run(name, args.data_root)
        eps = [analyze_episode(e) for e in rows]
        by_qid = {e["qid"]: e for e in eps}
        data[name] = {
            "n": len(rows),
            "data_sha256": rj["data_files"]["sha256"],
            "by_qid": by_qid,
            "metrics": rj["metrics"],
        }
        assert len(by_qid) == 256 and len(set(by_qid)) == 256, f"{name}: bad qid space"
        assert data[name]["by_qid"] is not None
    shas = {n: data[n]["data_sha256"] for n in RUNS}
    assert len(set(shas.values())) == 1, f"data SHA mismatch: {shas}"

    # pre-registration mapping SHA (from the two counterfactual run dirs)
    prereg_shas = {}
    for cond in CONDITIONS:
        p = (Path(args.data_root) / "project3-search-agent-rl" / "runs" /
             RUNS[cond] / "retrieval_condition_preregistration.json")
        d = json.load(p.open())
        assert d["condition"] == cond and d["n_questions"] == 256 and d["shuffle_step"] == 17
        prereg_shas[cond] = d["mapping_sha256"]
        assert prereg_shas[cond] == EXPECTED_MAPPING_SHA, f"{cond}: mapping SHA mismatch"

    report = {
        "kind": "p3-v2-counterfactual-stats",
        "created_at": "2026-08-23",
        "data_sha256": shas["real"],
        "qid_space": "0..255 sorted, unique per run (strict per-question pairing)",
        "model": "p3-v2-behavior-gs5-merged-20260823d (same merged model, greedy temp=0, GPU1)",
        "mapping": "shuffled: evidence of q replaced by REAL docs of (q+17) mod 256; "
                   "real retrieval executes first, non-success kept verbatim; "
                   "no-evidence: fixed neutral envelope, no HTTP",
        "preregistration_mapping_sha256": prereg_shas,
        "conditions": {},
        "pairs": {},
        "integrity": {
            "id_sets_equal": True,
            "data_sha_identical": True,
            "mapping_sha_verified": True,
        },
    }

    # --- per-condition behavior ---
    for name in RUNS:
        eps = list(data[name]["by_qid"].values())
        searched = [e for e in eps if e["searched"]]
        answered_off = [e for e in eps if e["has_answer_offline"]]
        sc = [e for e in searched if e["won"]]
        nsc = [e for e in eps if (not e["searched"]) and e["won"]]
        no_a = [e for e in eps if not e["has_answer_offline"]]
        report["conditions"][name] = {
            "run_id": RUNS[name],
            "em": sum(e["won"] for e in eps),
            "em_rate": round(sum(e["won"] for e in eps) / 256, 4),
            "wilson95": wilson_ci(sum(e["won"] for e in eps), 256),
            "searched": len(searched),
            "search_rate": round(len(searched) / 256, 4),
            "answered_offline": len(answered_off),
            "answered_env_committed": sum(e["answered_env"] for e in eps),
            "search_to_answer": round(len([e for e in searched if e["has_answer_offline"]]) / max(1, len(searched)), 4),
            "search_to_correct": round(len(sc) / max(1, len(searched)), 4),
            "no_search_to_correct": round(len(nsc) / max(1, 256 - len(searched)), 4),
            "searched_and_correct_abs": len(sc),
            "no_search_and_correct_abs": len(nsc),
            "avg_steps": round(sum(e["avg_steps"] for e in eps) / 256, 4),
            "invalid_query_questions": sum(e["invalid_query"] for e in eps),
            "max_steps_exhausted": sum(e["max_steps_exhausted"] for e in eps),
            "unanswered_termination": {
                "max_steps_exhausted": sum(e["termination"] == "max_steps_exhausted" for e in no_a),
                "ended_no_answer_committed": sum(e["termination"] == "ended_no_answer_committed" for e in no_a),
                "drafted_but_never_committed": sum(e["has_answer_offline"] and not e["answered_env"] for e in eps),
            },
            "metrics_search_status_counts": data[name]["metrics"]["search_behavior_v2"].get("search_status_counts"),
        }

    # --- paired comparisons real vs each counterfactual ---
    for cond in CONDITIONS:
        A = data["real"]["by_qid"]
        B = data[cond]["by_qid"]
        cells = {"a11": 0, "a00": 0, "b10": 0, "c01": 0}  # b10: cf correct, real wrong; c01: real correct, cf wrong
        per_source = {}
        real_only = []
        cf_only = []
        real_sc_flip = {"still_correct": 0, "flipped": 0}
        answer_unchanged = 0
        answer_changed = 0
        for qid in sorted(A):
            a, b = A[qid], B[qid]
            src = a["source"]
            key = "a11" if (a["won"] and b["won"]) else "a00" if (not a["won"] and not b["won"]) else ("b10" if b["won"] else "c01")
            cells[key] += 1
            per_source.setdefault(src, {"a11": 0, "a00": 0, "b10": 0, "c01": 0})[key] += 1
            if a["won"] and not b["won"]:
                real_only.append(qid)
            if b["won"] and not a["won"]:
                cf_only.append(qid)
            if a["searched"] and a["won"]:
                if b["won"]:
                    real_sc_flip["still_correct"] += 1
                else:
                    real_sc_flip["flipped"] += 1
            # evidence-change-but-answer-unchanged: both runs committed an
            # answer with the same text (counterfactual evidence differs by
            # construction whenever the real search succeeded)
            if a["answered_env"] and b["answered_env"] and a["answer_text"] == b["answer_text"]:
                answer_unchanged += 1
            elif a["answered_env"] or b["answered_env"]:
                answer_changed += 1
        n_disc = cells["b10"] + cells["c01"]
        report["pairs"][f"real_vs_{cond}"] = {
            "control": "real", "target": cond,
            "contingency": cells,
            "net_gain": cells["c01"] - cells["b10"],
            "discordant_n": n_disc,
            "mcnemar_exact_two_sided_p": round(mcnemar_exact_two_sided(cells["b10"], cells["c01"]), 8),
            "wilson95": {
                "real": wilson_ci(cells["a11"] + cells["c01"], 256),
                cond: wilson_ci(cells["a11"] + cells["b10"], 256),
            },
            "real_only_correct": len(real_only),
            f"{cond}_only_correct": len(cf_only),
            "real_only_correct_qids": real_only,
            f"{cond}_only_correct_qids": cf_only,
            "real_searched_and_correct_flips": real_sc_flip,
            "evidence_change_answer_unchanged": answer_unchanged,
            "evidence_change_answer_changed": answer_changed,
            "per_source_discordants": {
                src: {"real_correct_cf_wrong": v["c01"], "real_wrong_cf_correct": v["b10"],
                      "discordant": v["c01"] + v["b10"]}
                for src, v in sorted(per_source.items())
            },
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    tmp.replace(out)

    # --- console ---
    print("== conditions (n=256, same model/seed/prompt, greedy) ==")
    for name in RUNS:
        c = report["conditions"][name]
        print(f"  {name:12s} em={c['em']:3d} ({c['em_rate']:.3f}) wilson={c['wilson95']} "
              f"searched={c['searched']} s2a={c['search_to_answer']:.3f} s2c={c['search_to_correct']:.3f} "
              f"sc_abs={c['searched_and_correct_abs']} nsc_abs={c['no_search_and_correct_abs']} "
              f"ans_off={c['answered_offline']} ans_env={c['answered_env_committed']} "
              f"steps={c['avg_steps']} invalid={c['invalid_query_questions']} maxst={c['max_steps_exhausted']}")
    print("\n== paired (real vs counterfactual, strict per-question) ==")
    for key, p in report["pairs"].items():
        c = p["contingency"]
        print(f"  {key:18s} 1->1={c['a11']:3d} 0->0={c['a00']:3d} realR/cfW={c['c01']:3d} realW/cfR={c['b10']:3d} "
              f"net={p['net_gain']:+d}  McNemar exact 2-sided p={p['mcnemar_exact_two_sided_p']:.8f}")
        print(f"      real-only correct={p['real_only_correct']}  {p['target']}-only correct={p[f'{p[chr(116)+chr(97)+chr(114)+chr(103)+chr(101)+chr(116)]}_only_correct']}")
        print(f"      real searched-and-correct: still correct {p['real_searched_and_correct_flips']['still_correct']}, flipped {p['real_searched_and_correct_flips']['flipped']}")
        print(f"      evidence-change answer-unchanged={p['evidence_change_answer_unchanged']} changed={p['evidence_change_answer_changed']}")


if __name__ == "__main__":
    main()
