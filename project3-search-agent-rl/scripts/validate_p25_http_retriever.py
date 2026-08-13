#!/usr/bin/env python3
"""Validate the localhost Search-R1 retrieval HTTP contract on fixed questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080/retrieve")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(8, 16), required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("validation URL must use the local loopback host")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    records = [record for record in records if record.get("split") == "test"][: args.count]
    if len(records) != args.count:
        raise ValueError(f"expected {args.count} test records, found {len(records)}")

    traces = []
    latencies = []
    errors = []
    full_document_answer_hits = 0
    session = requests.Session()
    for record in records:
        started = time.monotonic()
        try:
            response = session.post(
                args.url,
                json={"query": record["question"], "topk": args.topk, "return_scores": True},
                timeout=args.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            latency = time.monotonic() - started
            hits = payload["result"][0]
            if len(hits) != args.topk:
                raise ValueError(f"expected {args.topk} hits, got {len(hits)}")
            for hit in hits:
                if set(hit) != {"document", "score"}:
                    raise ValueError(f"unexpected hit keys: {sorted(hit)}")
                if not isinstance(hit["document"].get("id"), str) or not isinstance(
                    hit["document"].get("contents"), str
                ):
                    raise ValueError("document is missing string id/contents")
            answer_ranks = [
                rank
                for rank, hit in enumerate(hits, start=1)
                if any(str(answer).casefold() in hit["document"]["contents"].casefold() for answer in record["answers"])
            ]
            if answer_ranks:
                full_document_answer_hits += 1
            latencies.append(latency)
            traces.append(
                {
                    "record": record,
                    "latency_seconds": latency,
                    "answer_string_ranks": answer_ranks,
                    "hits": [
                        {
                            "rank": rank,
                            "document_id": hit["document"]["id"],
                            "score": hit["score"],
                            "contents_preview": hit["document"]["contents"][:500],
                        }
                        for rank, hit in enumerate(hits, start=1)
                    ],
                }
            )
            print(f"record={record['id']} latency_seconds={latency:.3f} status=success", flush=True)
        except Exception as exc:
            errors.append({"record_id": record["id"], "type": type(exc).__name__, "message": str(exc)})
            print(f"record={record['id']} status=error type={type(exc).__name__} message={exc}", flush=True)

    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "warning": "Real wiki-18 HTTP integration diagnostic; not a model-quality benchmark.",
        "parameters": {"url": args.url, "count": args.count, "topk": args.topk, "timeout": args.timeout},
        "metrics": {
            "requests": args.count,
            "successes": len(traces),
            "errors": len(errors),
            "answer_string_in_full_document_queries": full_document_answer_hits,
            "latency_p50_seconds": statistics.median(latencies) if latencies else None,
            "latency_p95_seconds": percentile(latencies, 0.95) if latencies else None,
            "latency_max_seconds": max(latencies) if latencies else None,
        },
        "errors": errors,
        "traces": traces,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print(f"sha256={hashlib.sha256(args.output.read_bytes()).hexdigest()}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
