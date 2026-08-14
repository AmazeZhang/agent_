#!/usr/bin/env python3
"""Deterministic train-64 builder for the P3 data-enlargement round.

Selects 64 training rows with fixed per-source quotas via ascending
SHA256("searchr1-p3-train64-v1\\0source\\0normalized_question") + original
row index as tiebreaker, excluding rows whose normalized question appears
in smoke-train (8 rows) or heldout-32 (32 rows, leakage guard: train-64
must never overlap the heldout eval set).

Pool split (audited 2026-08-14): upstream train.parquet contains only nq +
hotpotqa; the other five sources exist only in upstream test.parquet.
Hybrid pools maximize train-split provenance:
  - train pool (upstream train.parquet): nq 16, hotpotqa 16
  - test pool (upstream test.parquet): popqa 8, 2wikimultihopqa 8,
    triviaqa 8, musique 4, bamboogle 4
Both pools exclude smoke + heldout normalized questions; the shared
`seen` set dedupes across pools too (same question in both pools).

Output: datasets/searchr1-train64/train.parquet (schema identical to
smoke train.parquet) + manifest.json (source hashes, quotas, leakage
assertions, output SHA256).

Rebuilding must reproduce the same SHA256 (determinism).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

SELECTION_DOMAIN = "searchr1-p3-train64-v1"
TRAIN_POOL_QUOTAS: dict[str, int] = {
    "nq": 16,
    "hotpotqa": 16,
}
TEST_POOL_QUOTAS: dict[str, int] = {
    "popqa": 8,
    "2wikimultihopqa": 8,
    "triviaqa": 8,
    "musique": 4,
    "bamboogle": 4,
}
assert sum(TRAIN_POOL_QUOTAS.values()) + sum(TEST_POOL_QUOTAS.values()) == 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def stable_key(source: str, question: str, index: int) -> tuple[str, int]:
    payload = f"{SELECTION_DOMAIN}\0{source}\0{normalize_question(question)}".encode()
    return hashlib.sha256(payload).hexdigest(), index


def normalized_question_set(frame: pd.DataFrame) -> set[str]:
    return {normalize_question(str(item["question"])) for item in frame["env_kwargs"]}


def select_rows(
    frame: pd.DataFrame,
    quotas: dict[str, int],
    seen: set[str],
    duplicate_skips: list[str],
) -> list[int]:
    """Select quota rows per source; `seen` (normalized questions) mutates in
    place and dedupes across pools. Returns positional row indexes."""
    selected: list[int] = []
    for source, quota in quotas.items():
        candidates: list[tuple[tuple[str, int], int, str]] = []
        for index, row in frame[frame["data_source"] == source].iterrows():
            question = str(row["env_kwargs"]["question"])
            normalized = normalize_question(question)
            if normalized in seen:
                continue
            candidates.append((stable_key(source, question, int(index)), int(index), normalized))
        candidates.sort()
        source_selected = 0
        for _, index, normalized in candidates:
            if normalized in seen:
                duplicate_skips.append(normalized)
                continue
            selected.append(index)
            seen.add(normalized)
            source_selected += 1
            if source_selected == quota:
                break
        if source_selected != quota:
            raise RuntimeError(f"could only select {source_selected}/{quota} rows for {source}")
    return selected


def to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def record_id(pool: str, row: pd.Series) -> str:
    original_index = int(row["extra_info"]["index"])
    return f"train64-{pool}-{row['data_source']}-{original_index}"


def write_atomic(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    with partial.open("ab") as stream:
        stream.flush()
        import os

        os.fsync(stream.fileno())
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True, help="upstream train.parquet (nq+hotpotqa pool)")
    parser.add_argument("--test-source", type=Path, required=True, help="upstream test.parquet (other five sources pool)")
    parser.add_argument("--smoke-train", type=Path, required=True, help="smoke train.parquet (excluded)")
    parser.add_argument("--heldout", type=Path, required=True, help="heldout-32 heldout.parquet (excluded, leakage guard)")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory (train64)")
    args = parser.parse_args()

    for path in (args.train_source, args.test_source, args.smoke_train, args.heldout):
        if not path.is_file():
            raise SystemExit(f"required source missing: {path}")

    train = pd.read_parquet(args.train_source)
    test = pd.read_parquet(args.test_source)
    smoke_train = pd.read_parquet(args.smoke_train)
    heldout = pd.read_parquet(args.heldout)

    smoke_questions = normalized_question_set(smoke_train)
    heldout_questions = normalized_question_set(heldout)

    seen: set[str] = set(smoke_questions | heldout_questions)
    duplicate_skips: list[str] = []

    train_indexes = select_rows(train, TRAIN_POOL_QUOTAS, seen, duplicate_skips)
    test_indexes = select_rows(test, TEST_POOL_QUOTAS, seen, duplicate_skips)
    selected = pd.concat(
        [train.loc[train_indexes], test.loc[test_indexes]], ignore_index=True
    )
    if len(selected) != 64:
        raise RuntimeError(f"selected {len(selected)} rows, expected 64")

    actual_quotas = selected.groupby("data_source").size().to_dict()
    expected_quotas = {**TRAIN_POOL_QUOTAS, **TEST_POOL_QUOTAS}
    if actual_quotas != expected_quotas:
        raise RuntimeError(f"quota mismatch: {actual_quotas}")

    selected_questions = normalized_question_set(selected)
    overlap_smoke = selected_questions & smoke_questions
    overlap_heldout = selected_questions & heldout_questions
    if overlap_smoke or overlap_heldout:
        raise RuntimeError(
            f"leakage: smoke_overlap={len(overlap_smoke)} heldout_overlap={len(overlap_heldout)}"
        )
    if len(selected_questions) != 64:
        raise RuntimeError(f"internal duplicate questions: {64 - len(selected_questions)}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "train.parquet"
    selected.to_parquet(parquet_path, index=False)

    output_sha = sha256_file(parquet_path)
    pool_of_row = []
    for _ in range(len(train_indexes)):
        pool_of_row.append("train")
    for _ in range(len(test_indexes)):
        pool_of_row.append("test")
    records = [
        {
            "id": record_id(pool_of_row[i], selected.iloc[i]),
            "data_source": str(selected.iloc[i]["data_source"]),
            "question": str(selected.iloc[i]["env_kwargs"]["question"]),
            "original_index": int(selected.iloc[i]["extra_info"]["index"]),
            "pool": pool_of_row[i],
            "question_sha256": hashlib.sha256(
                normalize_question(str(selected.iloc[i]["env_kwargs"]["question"])).encode()
            ).hexdigest(),
        }
        for i in range(len(selected))
    ]

    manifest = {
        "claim_boundary": (
            "64-row deterministic training subset for the P3 data-enlargement "
            "round; not itself a quality claim."
        ),
        "selection": (
            "per-source quota, ascending SHA256(searchr1-p3-train64-v1\\0source\\0"
            "normalized_question), original row index as tiebreaker; excludes "
            "smoke-train (8) and heldout-32 (32) normalized questions; "
            "cross-pool dedupe on normalized question"
        ),
        "pool_split": (
            "upstream train.parquet holds only nq+hotpotqa; the other five "
            "sources exist only in upstream test.parquet (audited 2026-08-14). "
            "nq/hotpotqa selected from the train pool; popqa/2wikimultihopqa/"
            "triviaqa/musique/bamboogle from the test pool."
        ),
        "quotas": expected_quotas,
        "code_revisions": {"root": "<set-by-builder>"},
        "inputs": {
            "train_pool": {"path": str(args.train_source.resolve()), "sha256": sha256_file(args.train_source), "rows": len(train)},
            "test_pool": {"path": str(args.test_source.resolve()), "sha256": sha256_file(args.test_source), "rows": len(test)},
            "smoke_train": {"path": str(args.smoke_train.resolve()), "sha256": sha256_file(args.smoke_train), "rows": len(smoke_train)},
            "heldout": {"path": str(args.heldout.resolve()), "sha256": sha256_file(args.heldout), "rows": len(heldout)},
        },
        "leakage": {
            "smoke_train_normalized_overlap": len(overlap_smoke),
            "heldout_normalized_overlap": len(overlap_heldout),
            "internal_duplicate_questions": 64 - len(selected_questions),
        },
        "outputs": {
            "train": {"path": str(parquet_path.resolve()), "rows": 64, "sha256": output_sha},
            "records": {"path": str((out_dir / "records.jsonl").resolve()), "rows": 64},
        },
    }
    write_atomic(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    with open(out_dir / "records.jsonl", "w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")

    print(f"selected 64 rows -> {parquet_path}")
    print(f"output sha256: {output_sha}")
    print(f"leakage: smoke={len(overlap_smoke)} heldout={len(overlap_heldout)}")
    print(f"quotas: {actual_quotas}")
    print(f"cross-pool duplicate skips: {len(duplicate_skips)}")


if __name__ == "__main__":
    main()
