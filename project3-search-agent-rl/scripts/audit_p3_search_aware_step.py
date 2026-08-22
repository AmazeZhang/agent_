"""CPU-only per-step audit for a Search-aware GRPO training run.

Reads one step's rollout audit (rollouts/<step>.audit.jsonl, produced by the
P3 patch 0008 v1 trajectory-return + traj-audit machinery) and emits:

  - per-step reward 8-component aggregates (group/episode level)
  - search rate / effective query rate / invalid rate (per trajectory)
  - answer compliance (terminal record has an answer)
  - trajectory advantage distribution by trajectory type
    (useful-search / search-no-evidence / closed-book / direct-answer)
  - search-direct-answer trajectory count (searched and answered in-episode)

No GPU, no training code paths, no network access.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

COMPONENTS = (
    "answer_reward_c",
    "format_reward_c",
    "evidence_hit_reward_c",
    "searched_correct_bonus_c",
    "invalid_penalty_c",
    "redundant_penalty_c",
    "answer_leak_penalty_c",
)


def _recover_traj_adv(records: list[dict]) -> dict[str, float | None]:
    """Recover each traj_uid's true trajectory advantage from the audit's
    max() view: advantages[index].max() == traj_adv for positive adv, and for
    negative adv it is visible only on full-mask records (0 otherwise)."""
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    out: dict[str, float | None] = {}
    for uid, rs in by_uid.items():
        nz = {r["trajectory_advantage"] for r in rs if abs(r["trajectory_advantage"]) > 1e-9}
        if len(nz) > 1:
            raise ValueError(f"traj_uid {uid} shows multiple advantage values {nz}")
        out[uid] = next(iter(nz)) if nz else None  # None => adv <= 0 (all-masked view)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_path", type=Path, help="rollouts/<step>.audit.jsonl")
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.audit_path, encoding="utf-8")]
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    uid_adv = _recover_traj_adv(records)

    # per-uid (trajectory) classification
    comp_agg = Counter()
    n_search_traj = 0
    n_valid_search_traj = 0
    n_invalid_search_traj = 0
    n_searches = 0
    n_valid_searches = 0
    n_answer_compliant = 0
    n_direct_answer_after_search = 0
    adv_by_type: dict[str, list[float]] = defaultdict(list)
    type_counts: Counter[str] = Counter()
    group_comp_agg = Counter()

    for uid, rs in by_uid.items():
        ep = rs[0]["metadata"]["search_v1_episode"]
        for c in COMPONENTS:
            comp_agg[c] += ep[c]
        adv = uid_adv[uid]
        statuses = [r["metadata"]["search_v1"].get("status") for r in rs]
        searched = any(r["metadata"]["search_v1"].get("query") for r in rs)
        terminal = [r["metadata"]["search_v1"] for r in rs if r["metadata"]["search_v1"].get("terminal")]
        has_answer = bool(terminal) and (terminal[0].get("r_answer_total") is not None)
        if has_answer:
            n_answer_compliant += 1
        if searched:
            n_search_traj += 1
            n_valid_search_traj += sum(1 for s in statuses if s == "success")
            n_invalid_search_traj += sum(1 for s in statuses if s == "invalid_query")
            n_searches += sum(1 for s in statuses if s in ("success", "invalid_query", "api_error", "processing_error", "no_results"))
            n_valid_searches += sum(1 for s in statuses if s == "success")
            if has_answer:
                n_direct_answer_after_search += 1
        # trajectory type
        sv1s = [r["metadata"]["search_v1"] for r in rs]
        useful = any(s.get("evidence_credit") for s in sv1s) or ep["searched_correct_bonus_c"] > 0
        if useful:
            t = "useful_search"
        elif searched:
            t = "search_no_evidence"
        elif has_answer:
            t = "closed_book"
        else:
            t = "no_answer"
        if adv is not None:
            adv_by_type[t].append(adv)
        type_counts[t] += 1
        # group dict (per uid record, dedup)
        g = rs[0]["metadata"]["search_v1_group"]
        for c in COMPONENTS:
            group_comp_agg[c] += g[c]

    # group-level component sums need dedup per group, not per uid-record:
    group_comp = Counter()
    seen_groups: set[str] = set()
    for r in records:
        gid = r["metadata"]["search_v1_group"]["uid"]
        if gid in seen_groups:
            continue
        seen_groups.add(gid)
        g = r["metadata"]["search_v1_group"]
        for c in COMPONENTS:
            group_comp[c] += g[c]

    n_traj = len(by_uid)
    report = {
        "n_trajectories": n_traj,
        "n_records": len(records),
        "search_rate_traj": n_search_traj / n_traj,
        "n_search_trajectories": n_search_traj,
        "n_search_calls": n_searches,
        "effective_query_rate": n_valid_searches / n_searches if n_searches else None,
        "invalid_search_calls": n_invalid_search_traj,
        "invalid_rate": n_invalid_search_traj / n_searches if n_searches else None,
        "answer_compliance": n_answer_compliant / n_traj,
        "n_direct_answer_after_search": n_direct_answer_after_search,
        "reward_8_components_episode_sum": dict(comp_agg),
        "reward_8_components_group_sum": dict(group_comp),
        "trajectory_type_counts": dict(type_counts),
        "trajectory_advantage_by_type": {
            t: {
                "n_known": len(vals),
                "mean": sum(vals) / len(vals) if vals else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "n_positive": sum(1 for v in vals if v > 0),
            }
            for t, vals in sorted(adv_by_type.items())
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
