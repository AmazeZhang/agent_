#!/usr/bin/env python3
"""P3 v2 ten-step: five-way per-question paired statistics (CPU-only).

Five confirm-256 eval runs on official-confirm256-v1 (identical question ID
space 0..255, same data SHA, greedy temp=0, GPU1, seed=0, max_steps=4):

  step0      -- clean Step0 (2026-08-20 line)
  grpo10     -- clean GRPO10 (2026-08-20 line)
  gigpo10    -- clean GiGPO10 (2026-08-20 line)
  v2_gs5     -- fresh v2 10-step run, merged at global_step_5
  v2_gs10    -- fresh v2 10-step run, merged at global_step_10

For every ordered pair: contingency (1->1, 0->0, control-wrong/target-correct,
control-correct/target-wrong), net gain, exact two-sided McNemar p, Wilson 95%
CIs, per-source discordants. Behavior breakdown for all five runs (EM, search
rate, search->answer, search->correct, searched-and-correct abs, no-search->
correct, compliance, avg steps, unanswered termination, invalid, mixed rounds).
Step5->Step10 trajectory: per-question gained/lost/maintained cross-tabulated
with search behavior change.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/p3_v2_five_way_stats.py \
      --fresh-gs5 p3-eval-v2-tenstep-gs5-confirm256-20260823a \
      --fresh-gs10 p3-eval-v2-tenstep-gs10-confirm256-20260823a \
      --out gates/p3_v2_five_way_stats_20260823.json
"""
import argparse
import json
import math
from pathlib import Path

DEFAULT_RUNS = {
    "step0": "p3-eval-upstream-clean-step0-instruct-confirm256-20260820a",
    "grpo10": "p3-eval-upstream-clean-grpo10-confirm256-20260820a",
    "gigpo10": "p3-eval-upstream-clean-gigpo10-confirm256-20260820b",
}
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
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) * 0.5 ** n for i in range(k + 1))
    return min(p, 1.0)


def is_search_step(step):
    return step.get("info", {}).get("tool_name") == "search"


def is_answer_step(step):
    info = step.get("info", {})
    return str(info.get("postprocessed_action", "")).startswith("<answer>")


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
    n_search_steps = sum(1 for s in steps if is_search_step(s))
    # true redundant search: search round whose returned doc ids were all
    # already seen in earlier search rounds of the same episode
    seen_doc_ids = set()
    n_redundant = 0
    for s in steps:
        if is_search_step(s):
            doc_ids = set((s.get("info", {}).get("retrieval") or {}).get("document_ids") or [])
            if doc_ids and doc_ids <= seen_doc_ids:
                n_redundant += 1
            seen_doc_ids.update(doc_ids)
    return {
        "qid": ep["question_id"], "source": ep["source"], "won": bool(ep["won"]),
        "searched": searched, "answered_env": answered_env,
        "has_answer_offline": has_answer_offline, "invalid_query": invalid_q,
        "n_steps": len(steps), "n_search_steps": n_search_steps,
        "n_redundant_searches": n_redundant,
        "max_steps_exhausted": (not answered_env) and len(steps) >= ENV_MAX_STEPS,
        "termination_offline": "answered" if has_answer_offline else (
            "max_steps_exhausted" if (not answered_env) and len(steps) >= ENV_MAX_STEPS else (
                "ended_no_answer_committed" if not answered_env else "answered_env")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/media/imc/data")
    ap.add_argument("--fresh-gs5", required=True, help="fresh v2 ten-step run merged at gs5 (eval run id)")
    ap.add_argument("--fresh-gs10", required=True, help="fresh v2 ten-step run merged at gs10 (eval run id)")
    ap.add_argument("--out", default="gates/p3_v2_five_way_stats_20260823.json")
    args = ap.parse_args()

    runs = dict(DEFAULT_RUNS)
    runs["v2_gs5"] = args.fresh_gs5
    runs["v2_gs10"] = args.fresh_gs10
    ORDER = ["step0", "grpo10", "gigpo10", "v2_gs5", "v2_gs10"]

    data = {}
    for name in ORDER:
        base = Path(args.data_root) / "project3-search-agent-rl" / "runs" / runs[name]
        rows = [json.loads(l) for l in (base / "episodes.jsonl").open()]
        rj = json.load((base / "results.json").open())
        eps = [analyze_episode(e) for e in rows]
        by_qid = {e["qid"]: e for e in eps}
        data[name] = {
            "run_id": runs[name],
            "data_sha256": rj["data_files"]["sha256"],
            "qids": [e["qid"] for e in eps],
            "by_qid": by_qid,
        }
        assert len(by_qid) == 256 and len(set(by_qid)) == 256, f"{name}: bad qid space"
        assert data[name]["qids"] == sorted(data[name]["qids"]), f"{name}: order not sorted"

    shas = {name: data[name]["data_sha256"] for name in ORDER}
    assert len(set(shas.values())) == 1, f"data SHA mismatch: {shas}"
    qid_sets = {name: set(data[name]["by_qid"]) for name in ORDER}
    assert all(s == qid_sets["step0"] for s in qid_sets.values()), "qid set mismatch"

    report = {
        "kind": "p3-v2-five-way-stats",
        "created_at": "2026-08-23",
        "data_sha256": shas["step0"],
        "qid_space": "0..255 sorted, unique per run (strict per-question pairing)",
        "n_per_run": 256,
        "runs": {name: {"run_id": runs[name]} for name in ORDER},
        "behavior": {},
        "pairs": {},
        "step5_to_step10": {},
        "integrity": {"id_sets_equal": True, "data_sha_identical": True,
                      "order_sorted_identical": True},
    }

    # --- behavior ---
    for name in ORDER:
        eps = list(data[name]["by_qid"].values())
        searched = [e for e in eps if e["searched"]]
        answered_off = [e for e in eps if e["has_answer_offline"]]
        answered_env = [e for e in eps if e["answered_env"]]
        sc = [e for e in searched if e["won"]]
        nsc = [e for e in eps if (not e["searched"]) and e["won"]]
        no_a = [e for e in eps if not e["has_answer_offline"]]
        report["behavior"][name] = {
            "em": sum(e["won"] for e in eps),
            "em_rate": round(sum(e["won"] for e in eps) / 256, 4),
            "wilson95": wilson_ci(sum(e["won"] for e in eps), 256),
            "searched": len(searched),
            "search_rate": round(len(searched) / 256, 4),
            "answered_offline": len(answered_off),
            "answered_env_committed": len(answered_env),
            "search_to_answer": round(len([e for e in searched if e["has_answer_offline"]]) / max(1, len(searched)), 4),
            "search_to_correct": round(len(sc) / max(1, len(searched)), 4),
            "no_search_to_correct": round(len(nsc) / max(1, 256 - len(searched)), 4),
            "searched_and_correct_abs": len(sc),
            "no_search_and_correct_abs": len(nsc),
            "searched_but_no_answer_offline": sum(e["searched"] and not e["has_answer_offline"] for e in eps),
            "avg_steps": round(sum(e["n_steps"] for e in eps) / 256, 4),
            "avg_search_steps": round(sum(e["n_search_steps"] for e in eps) / 256, 4),
            "true_redundant_searches": sum(e["n_redundant_searches"] for e in eps),
            "search_rounds_distribution": dict(sorted({
                k: sum(1 for e in eps if e["n_search_steps"] == k)
                for k in sorted({e["n_search_steps"] for e in eps})
            }.items())),
            "invalid_query_questions": sum(e["invalid_query"] for e in eps),
            "max_steps_exhausted": sum(e["max_steps_exhausted"] for e in eps),
            "unanswered_termination_offline": {
                "max_steps_exhausted": sum(e["termination_offline"] == "max_steps_exhausted" for e in no_a),
                "ended_no_answer_committed": sum(e["termination_offline"] == "ended_no_answer_committed" for e in no_a),
            },
        }

    # --- all ordered pairs ---
    for a in ORDER:
        for b in ORDER:
            if a == b:
                continue
            A, B = data[a]["by_qid"], data[b]["by_qid"]
            cells = {"a11": 0, "a00": 0, "b10": 0, "c01": 0}  # b10: B correct A wrong; c01: A correct B wrong
            per_source = {}
            for qid in sorted(A):
                x, y = A[qid], B[qid]
                src = x["source"]
                key = "a11" if (x["won"] and y["won"]) else "a00" if (not x["won"] and not y["won"]) else ("b10" if y["won"] else "c01")
                cells[key] += 1
                per_source.setdefault(src, {"a11": 0, "a00": 0, "b10": 0, "c01": 0})[key] += 1
            report["pairs"][f"{a}_vs_{b}"] = {
                "control": a, "target": b,
                "contingency": cells,
                "net_gain": cells["c01"] - cells["b10"],
                "discordant_n": cells["b10"] + cells["c01"],
                "mcnemar_exact_two_sided_p": round(mcnemar_exact_two_sided(cells["b10"], cells["c01"]), 8),
                "wilson95": {
                    a: wilson_ci(cells["a11"] + cells["c01"], 256),
                    b: wilson_ci(cells["a11"] + cells["b10"], 256),
                },
                "per_source_discordants": {
                    src: {"a_correct_b_wrong": v["c01"], "a_wrong_b_correct": v["b10"],
                          "discordant": v["c01"] + v["b10"]}
                    for src, v in sorted(per_source.items())
                },
            }

    # --- Step5 -> Step10 trajectory (fresh v2 line only) ---
    A, B = data["v2_gs5"]["by_qid"], data["v2_gs10"]["by_qid"]
    traj = {"gained": [], "lost": [], "maintained_correct": [], "maintained_wrong": []}
    for qid in sorted(A):
        x, y = A[qid], B[qid]
        if x["won"] and not y["won"]:
            traj["lost"].append(qid)
        elif y["won"] and not x["won"]:
            traj["gained"].append(qid)
        elif x["won"]:
            traj["maintained_correct"].append(qid)
        else:
            traj["maintained_wrong"].append(qid)
    report["step5_to_step10"] = {
        "gained": traj["gained"],
        "lost": traj["lost"],
        "maintained_correct": traj["maintained_correct"],
        "maintained_wrong": traj["maintained_wrong"],
        "n_gained": len(traj["gained"]),
        "n_lost": len(traj["lost"]),
        "n_maintained_correct": len(traj["maintained_correct"]),
        "n_maintained_wrong": len(traj["maintained_wrong"]),
        "net": len(traj["gained"]) - len(traj["lost"]),
        "search_behavior_change": {
            "gained_questions": [{"qid": q, "s5_searched": A[q]["searched"], "s10_searched": B[q]["searched"],
                                  "s5_steps": A[q]["n_steps"], "s10_steps": B[q]["n_steps"]} for q in traj["gained"]],
            "lost_questions": [{"qid": q, "s5_searched": A[q]["searched"], "s10_searched": B[q]["searched"],
                                "s5_steps": A[q]["n_steps"], "s10_steps": B[q]["n_steps"]} for q in traj["lost"]],
        },
        "mcnemar_exact_two_sided_p": round(
            mcnemar_exact_two_sided(len(traj["lost"]), len(traj["gained"])), 8),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    tmp.replace(out)

    print("== behavior (n=256, greedy temp=0, seed=0, GPU1) ==")
    for name in ORDER:
        b = report["behavior"][name]
        print(f"  {name:8s} em={b['em']:3d} ({b['em_rate']:.3f}) wilson={b['wilson95']} "
              f"searched={b['searched']:3d} s2a={b['search_to_answer']:.3f} s2c={b['search_to_correct']:.3f} "
              f"sc_abs={b['searched_and_correct_abs']:3d} nsc_abs={b['no_search_and_correct_abs']:3d} "
              f"ans_off={b['answered_offline']:3d} ans_env={b['answered_env_committed']:3d} "
              f"steps={b['avg_steps']:.3f} invalid={b['invalid_query_questions']} "
              f"maxst={b['max_steps_exhausted']}")
    print("\n== paired (strict per-question) ==")
    for key, p in sorted(report["pairs"].items()):
        c = p["contingency"]
        print(f"  {key:22s} 1->1={c['a11']:3d} 0->0={c['a00']:3d} "
              f"{p['control'][:6]}R/{p['target'][:6]}W={c['c01']:3d} {p['control'][:6]}W/{p['target'][:6]}R={c['b10']:3d} "
              f"net={p['net_gain']:+d}  p={p['mcnemar_exact_two_sided_p']:.8f}")
    tr = report["step5_to_step10"]
    print(f"\n== Step5 -> Step10: gained={tr['n_gained']} lost={tr['n_lost']} "
          f"maintained_correct={tr['n_maintained_correct']} maintained_wrong={tr['n_maintained_wrong']} "
          f"net={tr['net']:+d} McNemar p={tr['mcnemar_exact_two_sided_p']:.8f}")


if __name__ == "__main__":
    main()
