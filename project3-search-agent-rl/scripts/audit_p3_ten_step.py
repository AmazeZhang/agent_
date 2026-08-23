#!/usr/bin/env python3
"""P3 v2 ten-step: per-step training audit with the 10 abort conditions (CPU-only).

Reads one step's rollout artifacts of the fresh v2 GRPO10 run
(rollouts/<N>.audit.jsonl + stdout/stderr) and checks:

  C1 search trajectory rate < 70% for 2 consecutive steps
  C2 useful-search (evidence_hit) declining 3 consecutive steps by >50% of Step1
  C3 invalid query rate > 10%
  C4 true-redundant penalty again the largest-magnitude reward component
  C5 duplicate identity ((traj_uid, env_step) repeated) != 0
  C6 reward-component-sum == trajectory return == advantage broadcast mismatch
  C7 OOM / NaN / Xid / GPU-drop / retriever-timeout signatures in logs
  C8 config fingerprint change vs the launch-time fingerprint
  C9 GPU0/5 used (metadata.env physical_gpu_ids / CUDA_VISIBLE_DEVICES gate)
  C10 NCCL collective divergence or worker loss in logs

Audit semantics (validated against the completed 5-step v2 run):
  - one audit record per (trajectory, env_step); traj_uid = trajectory identity
  - uid = GRPO group id (shared across the n=5 rollouts of a group) -- NOT a
    duplicate-identity signal
  - search_v1.status != None marks a search step record
  - search_v1_episode.<comp>_c is the per-episode reward component (cents),
    identical on every record of the trajectory
  - answer completion = per-trajectory answer_reward_c != 0 (committed
    <answer> round; format_reward_c is 0 by design: format_score=0.0)

Per-step stats: search trajectory rate, useful-search, searched and correct
(evidence_credit), closed-book, valid/invalid query, new-document search,
true redundant, search-round distribution, reward components (c), per-trajectory
advantage, answer completion, duplicate identity, padding count.

State (multi-step conditions) accumulates in --state <json>.

Usage:
  CUDA_VISIBLE_DEVICES='' python3 scripts/audit_p3_ten_step.py \
      --run /media/imc/data/project3-search-agent-rl/runs/<run-id> \
      --state gates/p3_ten_step_audit_20260823a.json \
      --expected-config-sha d727b64f7c1c235e1d070637d9af498a02b1b89868bce088afdc19b814358402
"""
import argparse
import json
import re
from pathlib import Path

COMPONENTS = ("answer_reward_c", "format_reward_c", "evidence_hit_reward_c",
              "searched_correct_bonus_c", "invalid_penalty_c", "redundant_penalty_c")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def audit_step(run: Path, step: int) -> dict | None:
    audit = load_jsonl(run / "rollouts" / f"{step}.audit.jsonl")
    if not audit:
        return None

    # --- per-trajectory grouping ---
    trajs: dict[str, dict] = {}
    n_search_records = 0
    n_invalid = 0
    n_evidence_hit = 0
    n_evidence_credit = 0
    n_redundant = 0
    dup_identity: list[tuple] = []
    padding = 0
    seen_pairs: set[tuple] = set()
    for r in audit:
        md = r["metadata"]
        tid = md["traj_uid"]
        pair = (tid, md["env_step"])
        if pair in seen_pairs:
            dup_identity.append(pair)
        seen_pairs.add(pair)
        if md.get("is_padding"):
            padding += 1
        t = trajs.setdefault(tid, {
            "searched": False, "n_search_steps": 0,
            "answer_committed": False, "components": {},
            "adv": [], "record_count": 0,
        })
        t["record_count"] += 1
        sv = md.get("search_v1") or {}
        if sv.get("status") is not None:
            t["searched"] = True
            t["n_search_steps"] += 1
            n_search_records += 1
            if sv.get("invalid_or_error"):
                n_invalid += 1
            if sv.get("evidence_hit"):
                n_evidence_hit += 1
            if sv.get("evidence_credit"):
                n_evidence_credit += 1
            if sv.get("redundant_search"):
                n_redundant += 1
        eps = md.get("search_v1_episode") or {}
        if eps.get("answer_reward_c", 0) != 0:
            t["answer_committed"] = True
        for k in COMPONENTS:
            if eps:
                t["components"][k] = eps.get(k, 0)
        t["adv"].append(float(r["trajectory_advantage"]))

    n_traj = len(trajs)
    search_trajs = [t for t in trajs.values() if t["searched"]]
    n_sc = sum(1 for t in trajs.values() if t["searched"] and t["components"].get("evidence_hit_reward_c", 0))
    # searched-and-correct: evidence_credit / searched_correct_bonus_c nonzero
    n_sc_bonus = sum(1 for t in trajs.values() if t["components"].get("searched_correct_bonus_c", 0))
    n_committed = sum(1 for t in trajs.values() if t["answer_committed"])
    n_closed = n_traj - len(search_trajs)
    round_dist = {}
    for t in trajs.values():
        k = str(t["n_search_steps"])
        round_dist[k] = round_dist.get(k, 0) + 1

    # per-traj advantage consistency: nonzero values within a trajectory must agree
    adv_violations = []
    for tid, t in trajs.items():
        nz = sorted({round(a, 6) for a in t["adv"] if abs(a) > 1e-9})
        if len(nz) > 1:
            adv_violations.append(tid)
    pos_adv = sum(1 for t in trajs.values() if max(t["adv"]) > 0)
    total_adv = sum(max(t["adv"]) for t in trajs.values())

    # reward components (per-episode, so per-trajectory sums)
    comp_totals = {k: 0 for k in COMPONENTS}
    for t in trajs.values():
        for k in COMPONENTS:
            comp_totals[k] += t["components"].get(k, 0)
    nz_comp = {k: v for k, v in comp_totals.items() if v != 0}
    largest_abs = max(comp_totals, key=lambda k: abs(comp_totals[k])) if any(comp_totals.values()) else None

    return {
        "step": step,
        "n_records": len(audit),
        "n_trajectories": n_traj,
        "duplicate_identity_count": len(dup_identity),
        "padding_records": padding,
        "search_trajectory_rate": round(len(search_trajs) / n_traj, 4),
        "search_records": n_search_records,
        "useful_search_rate": round(n_evidence_hit / max(1, n_search_records), 4),
        "evidence_hit_records": n_evidence_hit,
        "evidence_credit_records": n_evidence_credit,
        "searched_correct_bonus_trajectories": n_sc_bonus,
        "invalid_search_rate": round(n_invalid / max(1, n_search_records), 4),
        "invalid_or_error_records": n_invalid,
        "true_redundant_rate": round(n_redundant / max(1, n_search_records), 4),
        "true_redundant_records": n_redundant,
        "new_document_search_records": n_search_records - n_redundant,
        "search_rounds_distribution": round_dist,
        "closed_book_episodes": n_closed,
        "answer_committed_rate": round(n_committed / n_traj, 4),
        "reward_components_cents": nz_comp,
        "largest_abs_component": largest_abs,
        "trajectory_advantage_mismatch_uids": adv_violations,
        "trajectories_with_positive_advantage": pos_adv,
        "trajectory_advantage_sum": round(total_adv, 4),
    }


def scan_logs(run: Path) -> dict:
    out = {}
    txt = ""
    for fn in ("stdout.log", "stderr.log"):
        p = run / fn
        if p.exists():
            txt += p.read_text(encoding="utf-8", errors="replace")
    out["nan_in_loss"] = bool(re.search(r"(?i)\b(nan|inf)\b", txt))
    out["oom"] = bool(re.search(r"(?i)out of memory|cuda oom|CUDA_OUT_OF_MEMORY", txt))
    out["xid"] = bool(re.search(r"(?i)xid|Xid 13|Xid 31|Xid 74|Xid 79", txt))
    out["gpu_drop"] = bool(re.search(r"(?i)device.*(lost|removed|disappeared)|cuda error:.*(unavailable|device)", txt))
    out["nccl"] = bool(re.search(r"(?i)nccl.*(error|diverg|timeout|check failed)|collective.*(mismatch|diverg)|NCCL_ERR", txt))
    out["worker_loss"] = bool(re.search(r"(?i)worker.*(died|failed|lost|terminated)|ray.*(worker.*crash|actor.*dead)|TaskRunner.*(error|exception)", txt))
    out["retriever_timeout"] = bool(re.search(r"(?i)retriev.*(timeout|timed out|connection.*fail)|Search API call failed", txt))
    m = re.search(r"resolved_config_sha256=([0-9a-f]{64})", txt)
    out["resolved_config_sha256"] = m.group(1) if m else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="managed run dir")
    ap.add_argument("--state", type=Path, default=None, help="state json (multi-step conditions)")
    ap.add_argument("--expected-config-sha", default=None, help="C8: fingerprint from launch")
    ap.add_argument("--expected-gpus", default="1,2,3,4,6,7", help="C9: allowed physical GPUs")
    args = ap.parse_args()

    run = args.run
    state = {}
    if args.state and args.state.exists():
        state = json.load(args.state.open())

    # C9: GPU gate from managed metadata
    meta = run / "metadata.env"
    gpus = ""
    if meta.exists():
        for line in meta.open():
            line = line.strip()
            if line.startswith("physical_gpu_ids="):
                gpus = line.split("=", 1)[1]
            elif line.startswith("CUDA_VISIBLE_DEVICES="):
                gpus = line.split("=", 1)[1]
    allowed = set(args.expected_gpus.split(","))
    actual = set(g.strip() for g in gpus.split(",") if g.strip()) if gpus else set()
    c9 = {"actual": gpus, "allowed": args.expected_gpus,
          "violation": bool(actual - allowed)}

    # C8: config fingerprint
    logs = scan_logs(run)
    c8 = {"launch_fingerprint": args.expected_config_sha,
          "seen_fingerprint": logs["resolved_config_sha256"],
          "violation": bool(args.expected_config_sha and logs["resolved_config_sha256"]
                            and args.expected_config_sha != logs["resolved_config_sha256"])}

    # C7/C10 from logs
    c7 = {k: logs[k] for k in ("nan_in_loss", "oom", "xid", "gpu_drop", "retriever_timeout")}
    c10 = {k: logs[k] for k in ("nccl", "worker_loss")}

    # audit every step whose audit file exists
    steps = sorted({int(p.stem.split(".")[0]) for p in (run / "rollouts").glob("*.audit.jsonl")})
    step_stats = {}
    for s in steps:
        st = audit_step(run, s)
        if st is not None:
            step_stats[s] = st

    def srate(s):
        st = step_stats.get(s)
        if not st:
            return None
        return st["search_trajectory_rate"]

    c1 = {"consecutive_steps_below_70pct": [], "violation": False}
    for s in sorted(step_stats):
        r = srate(s)
        if r is not None and r < 0.7:
            c1["consecutive_steps_below_70pct"].append(s)
        else:
            c1["consecutive_steps_below_70pct"] = []
    c1["violation"] = len(c1["consecutive_steps_below_70pct"]) >= 2

    c2 = {"violation": False, "note": None}
    s1 = step_stats.get(1)
    if s1 and s1["useful_search_rate"] is not None and s1["useful_search_rate"] > 0:
        base = s1["useful_search_rate"]
        decline = 0
        for s in sorted(step_stats):
            st = step_stats[s]
            if st["useful_search_rate"] is None:
                continue
            if st["useful_search_rate"] < base * 0.5:
                decline += 1
            else:
                decline = 0
            if decline >= 3:
                c2["violation"] = True
                c2["note"] = f"useful-search < 50% of Step1 ({base:.3f}) for 3 consecutive steps ending at {s}"
                break

    c3 = {"violation": False, "worst": None}
    for s in sorted(step_stats):
        st = step_stats[s]
        if st["invalid_search_rate"] > 0.10:
            c3["violation"] = True
            c3["worst"] = {"step": s, "rate": st["invalid_search_rate"]}
            break

    c4 = {"violation": False, "steps": []}
    for s in sorted(step_stats):
        st = step_stats[s]
        if st["largest_abs_component"] == "redundant_penalty_c" and st["reward_components_cents"].get("redundant_penalty_c", 0) < 0:
            c4["steps"].append(s)
            c4["violation"] = True

    c5 = {"violation": False, "steps": []}
    for s in sorted(step_stats):
        if step_stats[s]["duplicate_identity_count"] != 0:
            c5["steps"].append(s)
            c5["violation"] = True

    c6 = {"violation": False, "steps": [], "note": "advantage broadcast checked in-process by v2-0006 (fail-closed); audit re-checks per-traj consistency"}
    for s in sorted(step_stats):
        if step_stats[s]["trajectory_advantage_mismatch_uids"]:
            c6["steps"].append(s)
            c6["violation"] = True

    report = {
        "kind": "p3-ten-step-per-step-audit",
        "run_id": run.name,
        "conditions": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6,
                       "C7": c7, "C8": c8, "C9": c9, "C10": c10},
        "step_stats": step_stats,
        "any_violation": c1["violation"] or c2["violation"] or c3["violation"] or c4["violation"]
                         or c5["violation"] or c6["violation"] or any(c7.values())
                         or c8["violation"] or c9["violation"] or any(c10.values()),
    }
    if args.state:
        state["last_audit"] = report
        args.state.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.state.with_suffix(args.state.suffix + ".partial")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(args.state)

    print(f"== per-step audit: {run.name} ==")
    for s in sorted(step_stats):
        st = step_stats[s]
        print(f"  step {s}: traj={st['n_trajectories']} rec={st['n_records']} "
              f"search_traj_rate={st['search_trajectory_rate']:.3f} "
              f"useful={st['useful_search_rate']:.3f} invalid={st['invalid_search_rate']:.3f} "
              f"redundant={st['true_redundant_rate']:.3f} "
              f"ans_committed={st['answer_committed_rate']:.3f} closed={st['closed_book_episodes']} "
              f"dup_identity={st['duplicate_identity_count']} pad={st['padding_records']} "
              f"pos_adv={st['trajectories_with_positive_advantage']} adv_sum={st['trajectory_advantage_sum']:.3f} "
              f"largest_comp={st['largest_abs_component']}")
    print("== conditions ==")
    for k, v in report["conditions"].items():
        flag = "VIOLATION" if (v.get("violation") or any(x for x in v.values() if x is True)) else "ok"
        print(f"  {k}: {flag}  {json.dumps(v, ensure_ascii=False)[:180]}")
    print(f"any_violation={report['any_violation']}")


if __name__ == "__main__":
    main()
