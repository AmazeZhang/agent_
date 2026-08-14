#!/usr/bin/env python3
"""Build a deterministic, leakage-free held-out evaluation set for P3.

Samples 32 questions from the upstream Search-R1 test split (51,713 rows),
excluding every normalized question that appears in the upstream train split
(169,615 rows, covering the 10 cross-split overlaps found by the P1 audit),
the smoke train split (8 rows) and the smoke test split (16 rows). Selection
is per-source quota via ascending SHA256("searchr1-p3-eval-v1\\0source\\
0normalized_question"), so rebuilding yields byte-identical outputs.

The output is a real held-out question set scored against the real Wiki-18
retriever. It is still small-sample (32 rows) preliminary evidence; it does not
by itself establish Search-R1 reproduction or generalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


HELDOUT_QUOTAS = {
    "nq": 8,
    "hotpotqa": 8,
    "popqa": 4,
    "2wikimultihopqa": 4,
    "triviaqa": 4,
    "musique": 2,
    "bamboogle": 2,
}
EXPECTED_TOTAL = sum(HELDOUT_QUOTAS.values())  # 32

SELECTION_DOMAIN = "searchr1-p3-eval-v1"
UPSTREAM_REPO = "PeterJinGo/nq_hotpotqa_train"
UPSTREAM_REVISION = "b7d80abfee334a7a91cb377544f09180d58b34f6"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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
    frame: pd.DataFrame, quotas: dict[str, int], excluded_questions: set[str]
) -> pd.DataFrame:
    selected: list[int] = []
    seen = set(excluded_questions)
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
                continue
            selected.append(index)
            seen.add(normalized)
            source_selected += 1
            if source_selected == quota:
                break
        if source_selected != quota:
            raise RuntimeError(f"could only select {source_selected}/{quota} rows for {source}")
    return frame.loc[selected].copy()


def to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def record_id(row: pd.Series) -> str:
    original_index = int(row["extra_info"]["index"])
    return f"heldout-{row['data_source']}-{original_index}"


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
    parser.add_argument("--test-source", type=Path, required=True, help="upstream test.parquet")
    parser.add_argument("--train-source", type=Path, required=True, help="upstream train.parquet (leakage exclusion)")
    parser.add_argument("--smoke-train", type=Path, required=True, help="smoke train.parquet (leakage exclusion)")
    parser.add_argument("--smoke-test", type=Path, required=True, help="smoke test.parquet (coverage exclusion)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.test_source, args.train_source, args.smoke_train, args.smoke_test):
        if not path.is_file():
            raise SystemExit(f"input parquet missing: {path}")

    output_parquet = args.output_dir / "heldout.parquet"
    if output_parquet.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_parquet}")

    test_frame = pd.read_parquet(args.test_source)
    train_frame = pd.read_parquet(args.train_source)
    smoke_train = pd.read_parquet(args.smoke_train)
    smoke_test = pd.read_parquet(args.smoke_test)

    # Leakage exclusion: every normalized question trained on (upstream train,
    # smoke train) is banned. Smoke test rows are additionally excluded from the
    # pool so the held-out coverage is maximized (16 + 32 distinct questions).
    upstream_train_questions = normalized_question_set(train_frame)
    smoke_train_questions = normalized_question_set(smoke_train)
    smoke_test_questions = normalized_question_set(smoke_test)
    banned = upstream_train_questions | smoke_train_questions
    pool_exclusions = banned | smoke_test_questions

    heldout = select_rows(test_frame, HELDOUT_QUOTAS, pool_exclusions)

    if len(heldout) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} rows, got {len(heldout)}")
    heldout_questions = normalized_question_set(heldout)
    if not heldout_questions.isdisjoint(banned):
        raise RuntimeError("leakage: held-out questions overlap trained questions")
    if not heldout_questions.isdisjoint(smoke_test_questions):
        raise RuntimeError("held-out questions overlap smoke test questions")
    actual_quotas = heldout.groupby("data_source").size().to_dict()
    if actual_quotas != HELDOUT_QUOTAS:
        raise RuntimeError(f"quota mismatch: {actual_quotas}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout.to_parquet(output_parquet, index=False)

    records = []
    for _, row in heldout.iterrows():
        env_kwargs = to_builtin(row["env_kwargs"])
        records.append(
            {
                "id": record_id(row),
                "split": "heldout",
                "source": str(row["data_source"]),
                "original_index": int(row["extra_info"]["index"]),
                "question": env_kwargs["question"],
                "answers": env_kwargs["ground_truth"]["target"],
            }
        )
    records_path = args.output_dir / "records.jsonl"
    write_atomic(
        records_path,
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
    )

    root_revision = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    manifest = {
        "schema_version": 1,
        "purpose": "held-out evaluation set for Step 0/Step 2/Step 5 same-condition comparison",
        "claim_boundary": (
            "small-sample (32 rows) preliminary evidence; not by itself a Search-R1 "
            "reproduction or a generalization claim"
        ),
        "upstream_dataset": {
            "repo_id": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
            "license": "not declared in the Hugging Face dataset metadata; manual verification required",
        },
        "code_revisions": {
            "root": root_revision,
        },
        "selection_algorithm": (
            f"per-source quota, ascending SHA256({SELECTION_DOMAIN}\\0source\\0normalized_question), "
            "excluding upstream-train and smoke-train normalized questions"
        ),
        "selection_is_order_independent": True,
        "quotas": HELDOUT_QUOTAS,
        "leakage": {
            "upstream_train_normalized_overlap": len(heldout_questions & upstream_train_questions),
            "smoke_train_normalized_overlap": len(heldout_questions & smoke_train_questions),
            "smoke_test_normalized_overlap": len(heldout_questions & smoke_test_questions),
            "pool_duplicates_removed": 0,
        },
        "inputs": {
            "test": {"path": str(args.test_source.resolve()), "sha256": sha256_file(args.test_source), "rows": len(test_frame)},
            "train": {"path": str(args.train_source.resolve()), "sha256": sha256_file(args.train_source), "rows": len(train_frame)},
            "smoke_train": {"path": str(args.smoke_train.resolve()), "sha256": sha256_file(args.smoke_train), "rows": len(smoke_train)},
            "smoke_test": {"path": str(args.smoke_test.resolve()), "sha256": sha256_file(args.smoke_test), "rows": len(smoke_test)},
        },
        "outputs": {
            "heldout": {"path": str(output_parquet.resolve()), "sha256": sha256_file(output_parquet), "rows": len(heldout)},
            "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path), "rows": len(records)},
        },
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    write_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        f"created {len(heldout)} held-out rows at {args.output_dir} "
        f"(train overlap {len(heldout_questions & banned)})"
    )


if __name__ == "__main__":
    main()
