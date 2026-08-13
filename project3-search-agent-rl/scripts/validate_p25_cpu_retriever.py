#!/usr/bin/env python3
"""Validate the prepared Search-R1 FAISS index and run bounded CPU retrieval."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


EXPECTED_DIMENSION = 768
EXPECTED_VECTORS = 21015324


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_test_questions(path: Path, count: int) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    test = [record for record in records if record.get("split") == "test"]
    if len(test) < count:
        raise ValueError(f"requested {count} test records, found {len(test)}")
    return test[:count]


def encode_queries(model_path: Path, questions: list[str]) -> tuple[np.ndarray, float]:
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval()
    inputs = tokenizer(
        [f"query: {question}" for question in questions],
        max_length=256,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = model(**inputs, return_dict=True)
        embeddings = mean_pool(output.last_hidden_state, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    array = embeddings.cpu().numpy().astype(np.float32, order="C")
    return array, time.monotonic() - started


def scan_corpus(path: Path, wanted: set[int], progress_every: int) -> tuple[dict[int, dict], dict]:
    started = time.monotonic()
    found: dict[int, dict] = {}
    rows = 0
    alignment_errors: list[dict] = []
    first_id = None
    last_id = None
    with path.open("rb") as handle:
        for line_index, raw_line in enumerate(handle):
            record = json.loads(raw_line)
            record_id = record.get("id")
            if line_index == 0:
                first_id = record_id
            last_id = record_id
            if record_id != str(line_index) and len(alignment_errors) < 10:
                alignment_errors.append({"line_index": line_index, "id": record_id})
            if line_index in wanted:
                found[line_index] = record
            rows = line_index + 1
            if progress_every and rows % progress_every == 0:
                print(f"corpus_rows_checked={rows}", flush=True)
    return found, {
        "rows": rows,
        "first_id": first_id,
        "last_id": last_id,
        "alignment_error_examples": alignment_errors,
        "all_ids_match_line_index": not alignment_errors,
        "wanted_documents_found": len(found),
        "wanted_documents_expected": len(wanted),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("CPU validation requires CUDA_VISIBLE_DEVICES='' or '-1'")
    if args.count < 1 or args.topk < 1 or args.threads < 1:
        parser.error("count, topk, and threads must be positive")

    root = args.resource_root.resolve()
    prepared = root / "prepared"
    prepare_manifest_path = prepared / "prepare-complete.json"
    prepare_manifest = json.loads(prepare_manifest_path.read_text())
    index_path = prepared / "e5_Flat.index"
    corpus_path = prepared / "wiki-18.jsonl"
    model_path = root / "model" / "e5-base-v2"
    output_path = args.output or prepared / "cpu-validation.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite validation output: {output_path}")

    faiss.omp_set_num_threads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    load_started = time.monotonic()
    index = faiss.read_index(str(index_path))
    load_seconds = time.monotonic() - load_started
    index_metadata = {
        "class": type(index).__name__,
        "dimension": int(index.d),
        "vectors": int(index.ntotal),
        "is_trained": bool(index.is_trained),
        "metric_type": int(index.metric_type),
        "load_seconds": load_seconds,
    }
    print(json.dumps({"index": index_metadata}, indent=2), flush=True)
    if index.d != EXPECTED_DIMENSION or index.ntotal != EXPECTED_VECTORS:
        raise RuntimeError(f"unexpected FAISS shape: d={index.d}, ntotal={index.ntotal}")
    if prepare_manifest["corpus"]["rows"] != index.ntotal:
        raise RuntimeError("prepare manifest corpus rows do not match FAISS vectors")

    records = load_test_questions(args.records, args.count)
    embeddings, encode_seconds = encode_queries(model_path, [record["question"] for record in records])
    search_started = time.monotonic()
    scores, indices = index.search(embeddings, args.topk)
    search_seconds = time.monotonic() - search_started
    wanted = {int(value) for value in indices.reshape(-1) if value >= 0}
    documents, corpus_validation = scan_corpus(corpus_path, wanted, progress_every=1_000_000)
    if corpus_validation["rows"] != index.ntotal:
        raise RuntimeError("actual corpus row count does not match FAISS vectors")
    if not corpus_validation["all_ids_match_line_index"]:
        raise RuntimeError("corpus IDs are not aligned with zero-based line positions")
    if corpus_validation["wanted_documents_found"] != corpus_validation["wanted_documents_expected"]:
        raise RuntimeError("not all retrieved FAISS IDs mapped to corpus documents")

    retrievals = []
    for row, record in enumerate(records):
        hits = []
        for rank, (doc_index, score) in enumerate(zip(indices[row].tolist(), scores[row].tolist()), start=1):
            document = documents[int(doc_index)]
            hits.append(
                {
                    "rank": rank,
                    "index": int(doc_index),
                    "score": float(score),
                    "document_id": document["id"],
                    "contents_preview": document["contents"][:500],
                }
            )
        retrievals.append({"record": record, "hits": hits})

    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "warning": "Real wiki-18 retrieval integration check; not a model-quality benchmark.",
        "environment": {
            "faiss": getattr(faiss, "__version__", "unknown"),
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "threads": args.threads,
        },
        "parameters": {
            "resource_root": str(root),
            "records": str(args.records.resolve()),
            "count": args.count,
            "topk": args.topk,
        },
        "index": index_metadata,
        "query_encoding_seconds": encode_seconds,
        "search_seconds": search_seconds,
        "corpus_validation": corpus_validation,
        "elapsed_seconds": time.monotonic() - started,
        "retrievals": retrievals,
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "retrievals"}, indent=2), flush=True)
    print(f"output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
