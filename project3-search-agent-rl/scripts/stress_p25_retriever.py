#!/usr/bin/env python3
"""CPU-only stress test for the project retriever HTTP service.

Loads the real /retrieve endpoint with N concurrent client threads and reports
latency percentiles, throughput, timeouts and errors. Used to pick the server's
--max-concurrent-queries value for the 6-GPU GRPO training burst (up to 330
concurrent env retrievals), per docs/P3_PHASE2_OFFICIAL_TRAIN_DESIGN_2026-08-15.md.

Gates:
  - CPU-only: refuses to run with any CUDA_VISIBLE_DEVICES set
  - health gate: vectors must be 21,015,324 (real Wiki-18 index)

Exit code: 0 if no timeouts and no errors and health survives; 1 otherwise
(JSON report is always written to --out).

Usage:
  stress_p25_retriever.py --concurrency 128 --requests-per-worker 20 \
      --url http://127.0.0.1:18080/retrieve --out /tmp/stress.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Mix of realistic NQ/HotpotQA-style query lengths (load only, no evaluation).
QUERIES = [
    "who wrote the song imagine",
    "what channel is celebrity big brother on in the usa",
    "how many bones are in the adult human body",
    "who is the founder of microsoft",
    "when was the first iphone released",
    "what is the capital of new zealand",
    "who painted the mona lisa",
    "what year did world war ii end",
    "which country hosted the 2016 summer olympics",
    "what is the tallest mountain in africa",
    "how does photosynthesis convert sunlight into chemical energy",
    "what was the primary cause of the fall of the roman empire",
    "who discovered the structure of dna and in what year",
    "what are the main differences between mitosis and meiosis",
    "list the planets of the solar system in order from the sun",
    "who wrote the american declaration of independence and why was it written",
    "what is the economic impact of the industrial revolution on britain",
    "explain how the circulatory system and the respiratory system interact",
    "who was the first emperor of china and how did he unify the country",
    "what evidence supports the theory of continental drift proposed by wegener",
]


def _post(url: str, payload: dict, timeout: float) -> tuple[int, float]:
    started = time.monotonic()
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, time.monotonic() - started
    except urllib.error.HTTPError as error:
        return error.code, time.monotonic() - started
    except Exception as error:  # noqa: BLE001 - network errors are data here
        raise RuntimeError(f"request failed: {error!r}") from error


def _health(health_url: str) -> dict:
    with urllib.request.urlopen(health_url, timeout=5) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080/retrieve")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests-per-worker", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("/tmp/p3_retriever_stress.json"))
    args = parser.parse_args()

    if args.concurrency < 1 or args.requests_per_worker < 1:
        parser.error("concurrency and requests-per-worker must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise SystemExit("stress test is CPU-only; unset CUDA_VISIBLE_DEVICES")

    health_url = args.url.rsplit("/", 1)[0] + "/health"
    before = _health(health_url)
    if before.get("status") != "ready" or before.get("vectors") != 21_015_324:
        raise SystemExit(f"health gate failed: {before}")
    print(f"health before: {before}", flush=True)

    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(worker_id: int):
        for i in range(args.requests_per_worker):
            query = QUERIES[(worker_id + i) % len(QUERIES)]
            try:
                status, elapsed = _post(args.url, {"query": query, "topk": 3}, args.timeout)
            except RuntimeError as error:
                with lock:
                    errors.append(str(error))
                return
            with lock:
                latencies.append(elapsed)
                if status != 200:
                    errors.append(f"http {status}")

    started = time.monotonic()
    threads = [
        threading.Thread(target=worker, args=(i,), daemon=True)
        for i in range(args.concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.monotonic() - started

    after = _health(health_url)
    latencies.sort()
    n = len(latencies)
    percentiles = {
        "p50": statistics.median(latencies) if n else None,
        "p90": latencies[min(n - 1, int(n * 0.90))] if n else None,
        "p95": latencies[min(n - 1, int(n * 0.95))] if n else None,
        "p99": latencies[min(n - 1, int(n * 0.99))] if n else None,
        "max": latencies[-1] if n else None,
    }
    report = {
        "config": {
            "concurrency": args.concurrency,
            "requests_per_worker": args.requests_per_worker,
            "timeout_seconds": args.timeout,
            "total_requests": args.concurrency * args.requests_per_worker,
        },
        "health_before": before,
        "health_after": after,
        "wall_seconds": wall,
        "successful_responses": n,
        "errors": errors,
        "throughput_req_per_s": n / wall if wall > 0 else 0.0,
        "latency_seconds": percentiles,
        "verdict": "OK" if not errors and after.get("status") == "ready" else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["verdict"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
