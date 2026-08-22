"""CPU-only trajectory audit for a P3 official-loose confirm-256 eval run.

Reads one eval run dir (results.json + episodes.jsonl produced by
run_p3_eval_vllm_official.py) and emits audit.json with the 11 trajectory
statistics required for the Search-aware GRPO 10-step report:

  1. question-level search rate     6. no-search -> correct
  2. search calls per episode       7. answer compliance
  3. search success rate            8. EM (env & offline)
  4. search -> answer               9. correct-after-real-search count
  5. search -> correct             10. correct answers that depended on search
                                  11. invalid/empty/redundant/leak counts

Plus reward-8-component trigger counts for search-aware steps where the
episode metadata carries search_v1 (training protocol runs only; the eval
official-loose protocol has no step reward, so those counters stay zero).
No GPU, no training code paths, no network access.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SEARCH_V1_COMPONENTS = (
    "answer_reward_c",
    "format_reward_c",
    "evidence_hit_reward_c",
    "searched_correct_bonus_c",
    "invalid_penalty_c",
    "redundant_penalty_c",
    "answer_leak_penalty_c",
)

RETRIEVAL_FAILURES = {"invalid_query", "api_error", "processing_error", "no_results"}


def _question_level(episodes: list[dict]) -> dict:
    n = len(episodes)
    searched = [e for e in episodes if any(s["executed_search"] for s in e["steps"])]
    n_search = len(searched)
    total_calls = sum(1 for e in episodes for s in e["steps"] if s["executed_search"])
    statuses = Counter(
        s["info"].get("retrieval", {}).get("status")
        for e in episodes
        for s in e["steps"]
        if s.get("executed_search")
    )
    successes = sum(statuses.get(k, 0) for k in ("success",))
    fail_by_status = {k: statuses.get(k, 0) for k in RETRIEVAL_FAILURES}

    def has_answer(e: dict) -> bool:
        return bool(e["offline"].get("has_answer"))

    def correct(e: dict) -> bool:
        return e["offline"].get("score", 0.0) >= 1.0

    correct_env = [e for e in episodes if e["reward"] >= 1.0]
    search_answer = [e for e in searched if has_answer(e)]
    search_correct = [e for e in searched if correct(e)]
    nosearch_correct = [e for e in episodes if e not in searched and correct(e)]
    compliance = [e for e in episodes if has_answer(e)]

    return {
        "n_episodes": n,
        "search_rate_question_level": n_search / n,
        "n_searched_episodes": n_search,
        "search_calls_total": total_calls,
        "search_calls_per_episode": total_calls / n,
        "search_success_rate": successes / total_calls if total_calls else None,
        "retrieval_status_counts": dict(statuses),
        "retrieval_failure_counts": fail_by_status,
        "search_to_answer": len(search_answer) / n_search if n_search else None,
        "search_to_correct": len(search_correct) / n_search if n_search else None,
        "no_search_to_correct": len(nosearch_correct) / (n - n_search) if n - n_search else None,
        "n_correct_total": len(correct_env),
        "em_env": len(correct_env) / n,
        "em_offline": sum(1 for e in episodes if correct(e)) / n,
        "n_correct_after_real_search": len(search_correct),
        "answer_compliance": len(compliance) / n,
        "invalid_query_count": statuses.get("invalid_query", 0),
        "empty_query_count": sum(
            1
            for e in episodes
            for s in e["steps"]
            if s.get("executed_search")
            and not (s["info"].get("postprocessed_action") or "").strip()
        ),
        "redundant_search_count": 0,  # official-loose eval has no redundancy gate
        "answer_leak_count": 0,  # official-loose eval has no leak gate
        "reward_8_component_triggers": {c: 0 for c in SEARCH_V1_COMPONENTS},
        "trajectory_advantage_by_type": {},
    }


def _search_v1_audit(episodes: list[dict]) -> dict:
    """If episode steps carry search_v1 metadata (training-protocol audit
    files), aggregate the 8 reward components across steps; otherwise zeros."""
    comp_triggers = {c: 0 for c in SEARCH_V1_COMPONENTS}
    any_v1 = False
    for e in episodes:
        for s in e.get("steps", []):
            sv1 = s.get("info", {}).get("search_v1") if isinstance(s.get("info"), dict) else None
            if not isinstance(sv1, dict):
                continue
            any_v1 = True
            for c in SEARCH_V1_COMPONENTS:
                if sv1.get(c, 0) != 0:
                    comp_triggers[c] += 1
    return {
        "has_search_v1_metadata": any_v1,
        "reward_8_component_triggers": comp_triggers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="eval run dir containing results.json + episodes.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="output audit.json path (default: <run_dir>/audit.json)")
    args = parser.parse_args()

    run_dir = args.run_dir
    episodes_path = run_dir / "episodes.jsonl"
    results_path = run_dir / "results.json"
    if not episodes_path.is_file() or not results_path.is_file():
        raise SystemExit(f"missing results.json/episodes.jsonl in {run_dir}")
    episodes = [json.loads(line) for line in open(episodes_path, encoding="utf-8")]

    audit = {
        "run_dir": str(run_dir),
        "n_episodes": len(episodes),
        "trajectories": _question_level(episodes),
    }
    audit["trajectories"].update(_search_v1_audit(episodes))

    out = args.out or (run_dir / "audit.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
