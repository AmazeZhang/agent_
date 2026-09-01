#!/usr/bin/env python3
"""Pair Aware-v2 and external StepSearch retrieval behaviour on smoke-16."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from searchr1_repro.search_v2_reward import evidence_hit_in_docs, valid_aliases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aware-run", type=Path, required=True)
    parser.add_argument("--stepsearch-run", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    result = json.loads((run_dir / "results.json").read_text())
    episodes = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
    if len(episodes) != 16:
        raise ValueError(f"expected 16 episodes in {run_dir}, got {len(episodes)}")
    return result, episodes


def query(step: dict) -> str | None:
    retrieval = (step.get("info") or {}).get("retrieval") or {}
    value = retrieval.get("query")
    return value if isinstance(value, str) and value.strip() else None


def episode_view(episode: dict) -> dict:
    aliases = valid_aliases(episode["answers"])
    calls = []
    for step in episode["steps"]:
        value = query(step)
        if value is None:
            continue
        retrieval = (step.get("info") or {}).get("retrieval") or {}
        status = retrieval.get("status")
        observation = step.get("observation") or ""
        hit = status in {"success", "no_results"} and evidence_hit_in_docs(observation, aliases)
        calls.append({"query": value, "status": status, "evidence_hit": bool(hit)})
    return {
        "question_id": episode["question_id"],
        "question": episode["question"],
        "search_calls": len(calls),
        "evidence_hit_calls": sum(call["evidence_hit"] for call in calls),
        "evidence_hit_question": any(call["evidence_hit"] for call in calls),
        "queries": [call["query"] for call in calls],
        "em": int(episode["offline"]["score"] >= 1.0),
        "has_answer": bool(episode["offline"]["has_answer"]),
        "final_answer": episode["offline"]["final_answer"],
    }


def main() -> int:
    args = parse_args()
    aware_result, aware_episodes = load_run(args.aware_run)
    step_result, step_episodes = load_run(args.stepsearch_run)
    aware = {row["question_id"]: row for row in map(episode_view, aware_episodes)}
    step = {row["question_id"]: row for row in map(episode_view, step_episodes)}
    if set(aware) != set(step):
        raise ValueError("question_id sets differ")
    if aware_result["data_files"]["sha256"] != step_result["data_files"]["sha256"]:
        raise ValueError("dataset hashes differ")

    paired = []
    categories = {"both_hit": [], "stepsearch_only_hit": [], "aware_only_hit": [], "neither_hit": []}
    for question_id in sorted(aware):
        a, s = aware[question_id], step[question_id]
        if a["question"] != s["question"]:
            raise ValueError(f"question mismatch at id {question_id}")
        if a["evidence_hit_question"] and s["evidence_hit_question"]:
            category = "both_hit"
        elif s["evidence_hit_question"]:
            category = "stepsearch_only_hit"
        elif a["evidence_hit_question"]:
            category = "aware_only_hit"
        else:
            category = "neither_hit"
        categories[category].append(question_id)
        paired.append({"question_id": question_id, "question": a["question"], "category": category,
                       "aware": a, "stepsearch": s})

    aware_q_hits = sum(row["evidence_hit_question"] for row in aware.values())
    step_q_hits = sum(row["evidence_hit_question"] for row in step.values())
    aware_em = sum(row["em"] for row in aware.values())
    step_em = sum(row["em"] for row in step.values())
    eligible = step_q_hits - aware_q_hits >= 2 and step_em >= aware_em
    report = {
        "kind": "p3-aware-stepsearch-smoke16-retrieval-screen",
        "dataset_sha256": aware_result["data_files"]["sha256"],
        "aware_run": str(args.aware_run),
        "stepsearch_run": str(args.stepsearch_run),
        "aggregate_metrics": {
            "aware": aware_result["metrics"],
            "stepsearch": step_result["metrics"],
        },
        "question_evidence_hits": {"aware": aware_q_hits, "stepsearch": step_q_hits,
                                   "difference_stepsearch_minus_aware": step_q_hits - aware_q_hits},
        "paired_categories": categories,
        "em": {"aware": aware_em, "stepsearch": step_em},
        "decision": {
            "eligible_for_bounded_aware_experiment": eligible,
            "rule": "StepSearch gains >=2 evidence-hit questions and does not reduce EM",
        },
        "per_question": paired,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    rows = [
        "# Aware-v2 vs StepSearch smoke-16 retrieval screen",
        "",
        f"Dataset SHA256: `{report['dataset_sha256']}`",
        "",
        "| Metric | Aware-v2 | StepSearch |",
        "|---|---:|---:|",
        f"| Evidence-hit questions | {aware_q_hits}/16 | {step_q_hits}/16 |",
        f"| Evidence-hit calls | {aware_result['metrics']['search_behavior_v2']['evidence_hit_searches']}/{aware_result['metrics']['search_behavior_v2']['total_search_calls']} | {step_result['metrics']['search_behavior_v2']['evidence_hit_searches']}/{step_result['metrics']['search_behavior_v2']['total_search_calls']} |",
        f"| Multi-hop episodes | {aware_result['metrics']['search_behavior_v2']['multi_hop_episodes']}/16 | {step_result['metrics']['search_behavior_v2']['multi_hop_episodes']}/16 |",
        f"| True redundant searches | {aware_result['metrics']['search_behavior_v2']['true_redundant_searches']} | {step_result['metrics']['search_behavior_v2']['true_redundant_searches']} |",
        f"| Answer compliance | {aware_result['metrics']['overall']['answer_compliance']}/16 | {step_result['metrics']['overall']['answer_compliance']}/16 |",
        f"| EM | {aware_em}/16 | {step_em}/16 |",
        "",
        "## Paired evidence-hit sets",
        "",
    ]
    for name, ids in categories.items():
        rows.append(f"- `{name}` ({len(ids)}): {ids}")
    rows += ["", "## Aware-only evidence-hit cases", ""]
    for item in paired:
        if item["category"] != "aware_only_hit":
            continue
        rows += [
            f"- Q{item['question_id']}: {item['question']}",
            f"  - Aware query: `{item['aware']['queries'][0]}`",
            f"  - StepSearch queries: {json.dumps(item['stepsearch']['queries'], ensure_ascii=False)}",
        ]
    rows += [
        "",
        "## Decision",
        "",
        f"Eligible for a bounded Aware mechanism experiment: **{'yes' if eligible else 'no'}**.",
        "The models and prompt protocols differ, so this is a descriptive mechanism screen, not a causal algorithm comparison.",
    ]
    args.output_md.write_text("\n".join(rows) + "\n")
    print(json.dumps({"question_evidence_hits": report["question_evidence_hits"],
                      "paired_categories": categories, "em": report["em"],
                      "decision": report["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
