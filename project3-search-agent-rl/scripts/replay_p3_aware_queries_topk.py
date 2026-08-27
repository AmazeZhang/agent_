#!/usr/bin/env python3
"""CPU-only replay of final Aware queries at top-k=5.

One HTTP call per unique query retrieves top-5 from the already-running,
identity-checked Wiki-18 service. The first three document IDs must exactly
match the stored top-3 episode metadata. The script then compares the frozen
v2 answer-alias evidence proxy at k=3 and k=5. It does not run the policy,
generate answers, estimate EM, or modify the Retriever.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from searchr1_repro.search_v2_reward import evidence_hit_in_docs, valid_aliases


DEFAULT_RUN = "p3-eval-aware-v2-seed2026-gs10-confirm256-20260824a"
EXPECTED_TOP3_HIT_CALLS = 147
EXPECTED_TOP3_HIT_QUESTIONS = 127
EXPECTED_VECTORS = 21_015_324


def request_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def retrieve(url: str, query: str, timeout: float) -> list[dict[str, Any]]:
    payload = request_json(
        url,
        {"query": query, "topk": 5, "return_scores": True},
        timeout,
    )
    rows = payload.get("result")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list):
        raise RuntimeError(f"unexpected response envelope for query {query!r}")
    if len(rows[0]) != 5:
        raise RuntimeError(f"expected 5 hits, got {len(rows[0])} for query {query!r}")
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/media/imc/data")
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--url", default="http://127.0.0.1:18080/retrieve")
    parser.add_argument("--health-url", default="http://127.0.0.1:18080/health")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be in 1..8 for this bounded replay")
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out}")

    health = request_json(args.health_url, timeout=args.timeout)
    expected_health = {
        "status": "ready",
        "index_class": "IndexFlatIP",
        "dimension": 768,
        "vectors": EXPECTED_VECTORS,
        "corpus_rows": EXPECTED_VECTORS,
        "max_concurrent_queries": 64,
    }
    if health != expected_health:
        raise RuntimeError(f"Retriever health mismatch: {health}")

    run_dir = Path(args.data_root) / "project3-search-agent-rl" / "runs" / args.run
    episode_path = run_dir / "episodes.jsonl"
    episodes = [json.loads(line) for line in episode_path.open()]
    if len(episodes) != 256 or {ep["question_id"] for ep in episodes} != set(range(256)):
        raise RuntimeError("expected qid 0..255 exactly once")

    calls: list[dict[str, Any]] = []
    for episode in episodes:
        for step in episode["steps"]:
            info = step.get("info") or {}
            retrieval = info.get("retrieval") or {}
            if info.get("tool_name") == "search" and retrieval.get("status") == "success":
                calls.append(
                    {
                        "qid": int(episode["question_id"]),
                        "answers": episode["answers"],
                        "correct": bool(episode.get("reward", 0) >= 1.0),
                        "query": retrieval["query"],
                        "stored_top3_ids": [str(value) for value in retrieval.get("document_ids") or []],
                        "stored_top3_text": step.get("observation") or "",
                    }
                )
    if len(calls) != 328 or any(len(call["stored_top3_ids"]) != 3 for call in calls):
        raise RuntimeError("expected 328 successful calls with exactly three stored document IDs")

    for call in calls:
        call["stored_top3_hit"] = evidence_hit_in_docs(
            call["stored_top3_text"], valid_aliases(call["answers"])
        )
    # A top-3 hit remains a hit at top-5, so only replay historical misses.
    # This cuts requests while keeping the one preregistered metric exact.
    calls_needing_extras = [call for call in calls if not call["stored_top3_hit"]]
    unique_queries = list(dict.fromkeys(call["query"] for call in calls_needing_extras))
    responses: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(retrieve, args.url, query, args.timeout): query for query in unique_queries}
        completed = 0
        for future in as_completed(futures):
            query = futures[future]
            responses[query] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(unique_queries):
                print(f"completed {completed}/{len(unique_queries)} unique queries", flush=True)

    canonical_responses = [
        {
            "query": query,
            "hits": [
                {
                    "id": str(hit["document"]["id"]),
                    "contents_sha256": hashlib.sha256(hit["document"]["contents"].encode()).hexdigest(),
                    "score": hit["score"],
                }
                for hit in responses[query]
            ],
        }
        for query in sorted(responses)
    ]
    response_bundle_sha256 = hashlib.sha256(
        json.dumps(canonical_responses, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    top3_hit_calls = 0
    top5_hit_calls = 0
    top3_hit_qids: set[int] = set()
    top5_hit_qids: set[int] = set()
    call_level_new_hit_qids: set[int] = set()
    live_prefix_drifts: list[dict[str, Any]] = []
    insufficient_new_docs: list[dict[str, Any]] = []
    per_qid: dict[int, dict[str, Any]] = {
        int(ep["question_id"]): {
            "correct": bool(ep.get("reward", 0) >= 1.0),
            "top3_hit": False,
            "top5_hit": False,
        }
        for ep in episodes
    }
    for call in calls:
        aliases = valid_aliases(call["answers"])
        hit3 = bool(call["stored_top3_hit"])
        hit5 = hit3
        if not hit3:
            hits = responses[call["query"]]
            returned_ids = [str(hit["document"]["id"]) for hit in hits]
            if returned_ids[:3] != call["stored_top3_ids"]:
                live_prefix_drifts.append(
                    {
                        "qid": call["qid"],
                        "query": call["query"],
                        "stored": call["stored_top3_ids"],
                        "replayed": returned_ids[:3],
                    }
                )
            stored_ids = set(call["stored_top3_ids"])
            new_hits = [
                hit for hit in hits if str(hit["document"]["id"]) not in stored_ids
            ][:2]
            if len(new_hits) != 2:
                insufficient_new_docs.append(
                    {"qid": call["qid"], "query": call["query"], "new_docs": len(new_hits)}
                )
            augmented_docs = call["stored_top3_text"] + "\n" + "\n".join(
                hit["document"]["contents"] for hit in new_hits
            )
            hit5 = evidence_hit_in_docs(augmented_docs, aliases)
        if hit3:
            top3_hit_calls += 1
            top3_hit_qids.add(call["qid"])
            per_qid[call["qid"]]["top3_hit"] = True
        if hit5:
            top5_hit_calls += 1
            top5_hit_qids.add(call["qid"])
            per_qid[call["qid"]]["top5_hit"] = True
        if hit5 and not hit3:
            call_level_new_hit_qids.add(call["qid"])

    top3_baseline_reproduced = (
        top3_hit_calls == EXPECTED_TOP3_HIT_CALLS
        and len(top3_hit_qids) == EXPECTED_TOP3_HIT_QUESTIONS
    )
    if insufficient_new_docs or not top3_baseline_reproduced:
        failure = {
            "kind": "p3-aware-seed2026-anchored-topk-replay-failure",
            "live_prefix_drifts": live_prefix_drifts,
            "insufficient_new_docs": insufficient_new_docs,
            "top3_hit_calls": top3_hit_calls,
            "top3_hit_questions": len(top3_hit_qids),
            "expected_top3_hit_calls": EXPECTED_TOP3_HIT_CALLS,
            "expected_top3_hit_questions": EXPECTED_TOP3_HIT_QUESTIONS,
            "response_bundle_sha256": response_bundle_sha256,
        }
        failure_path = args.out.with_suffix(args.out.suffix + ".failure.json")
        failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n")
        raise RuntimeError(
            f"hard anchored replay gate failed: insufficient_new_docs={len(insufficient_new_docs)}, "
            f"top3_baseline_reproduced={top3_baseline_reproduced}; evidence={failure_path}"
        )

    searched_qids = {call["qid"] for call in calls}
    newly_hit_question_qids = top5_hit_qids - top3_hit_qids
    incorrect_no_hit_qids = {
        qid for qid in searched_qids if not per_qid[qid]["correct"] and not per_qid[qid]["top3_hit"]
    }
    rescued_in_incorrect = incorrect_no_hit_qids & newly_hit_question_qids
    report = {
        "kind": "p3-aware-seed2026-existing-query-topk3-vs-topk5-replay",
        "run": args.run,
        "protocol": {
            "policy_or_answer_generation": False,
            "same_existing_queries": True,
            "retrieval_only": True,
            "workers": args.workers,
            "historical_top3_anchor": "stored document bodies from episodes.jsonl",
            "top5_construction": (
                "historical top-3 plus the first two replayed documents whose IDs are not in historical top-3"
            ),
            "claim_boundary": (
                "This measures evidence-proxy recall for two extra candidates only; it does not estimate EM "
                "and is not an exact reconstruction of historical top-5 when a live rank boundary drifts."
            ),
        },
        "retriever": {
            "health": health,
            "listener_pid_recorded_by_operator": 1355816,
            "response_bundle_sha256": response_bundle_sha256,
        },
        "integrity": {
            "episodes_sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
            "questions": len(episodes),
            "successful_search_calls": len(calls),
            "historical_top3_miss_calls_replayed": len(calls_needing_extras),
            "unique_queries_replayed": len(unique_queries),
            "api_failures": 0,
            "live_prefix_drift_records": len(live_prefix_drifts),
            "live_prefix_drift_examples": live_prefix_drifts,
            "insufficient_new_docs": 0,
            "top3_baseline_reproduced": top3_baseline_reproduced,
            "anchor_note": (
                "Live prefix drift is recorded but cannot rewrite the historical top-3 baseline."
            ),
        },
        "results": {
            "searched_questions": len(searched_qids),
            "top3": {
                "evidence_hit_calls": top3_hit_calls,
                "evidence_hit_questions": len(top3_hit_qids),
                "question_hit_rate_given_search": len(top3_hit_qids) / len(searched_qids),
            },
            "top5": {
                "evidence_hit_calls": top5_hit_calls,
                "evidence_hit_questions": len(top5_hit_qids),
                "question_hit_rate_given_search": len(top5_hit_qids) / len(searched_qids),
            },
            "delta_top5_minus_top3": {
                "evidence_hit_calls": top5_hit_calls - top3_hit_calls,
                "evidence_hit_questions": len(top5_hit_qids) - len(top3_hit_qids),
                "question_hit_rate_pp": 100 * (len(top5_hit_qids) - len(top3_hit_qids)) / len(searched_qids),
                "newly_hit_qids": sorted(newly_hit_question_qids),
                "call_level_new_hit_qids_already_hit_on_another_step": sorted(
                    call_level_new_hit_qids & top3_hit_qids
                ),
            },
            "incorrect_searched_without_top3_evidence": len(incorrect_no_hit_qids),
            "those_newly_hit_at_top5": len(rescued_in_incorrect),
            "those_still_without_evidence_at_top5": len(incorrect_no_hit_qids - rescued_in_incorrect),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    partial.replace(args.out)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
