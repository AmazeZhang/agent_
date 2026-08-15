#!/usr/bin/env python3
"""Launch the project CPU retriever on the IPv4 loopback interface only."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import faiss
import torch
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from searchr1_repro.cpu_dense_retriever import CpuDenseRetriever, create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--default-topk", type=int, default=3)
    parser.add_argument("--max-topk", type=int, default=10)
    # Global /retrieve concurrency cap (queueing, not rejection). Protects the
    # CPU index from burst load (e.g. 330 GRPO envs). See create_app.
    parser.add_argument("--max-concurrent-queries", type=int, default=32)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU service requires CUDA_VISIBLE_DEVICES='' or '-1'")
    if not (1024 <= args.port <= 65535):
        parser.error("port must be between 1024 and 65535")
    if min(args.threads, args.default_topk, args.max_topk) < 1 or args.default_topk > args.max_topk:
        parser.error("invalid thread or top-k limits")

    faiss.omp_set_num_threads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = args.resource_root.resolve()
    prepared = root / "prepared"
    started = time.monotonic()
    retriever = CpuDenseRetriever.load(
        index_path=prepared / "e5_Flat.index",
        corpus_path=prepared / "wiki-18.jsonl",
        offsets_path=prepared / "wiki-18.offsets.npy",
        model_path=root / "model" / "e5-base-v2",
    )
    print(
        json.dumps(
            {
                "status": "loaded",
                "host": "127.0.0.1",
                "port": args.port,
                "threads": args.threads,
                "index_class": type(retriever.index).__name__,
                "dimension": int(retriever.index.d),
                "vectors": int(retriever.index.ntotal),
                "load_seconds": time.monotonic() - started,
            }
        ),
        flush=True,
    )
    app = create_app(
        retriever,
        default_topk=args.default_topk,
        max_topk=args.max_topk,
        max_concurrent_queries=args.max_concurrent_queries,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
