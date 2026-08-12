#!/usr/bin/env python3
"""Build deterministic Search-R1 P1 smoke splits and a protocol-only corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


TRAIN_QUOTAS = {"nq": 4, "hotpotqa": 4}
TEST_QUOTAS = {
    "nq": 2,
    "hotpotqa": 2,
    "popqa": 3,
    "2wikimultihopqa": 3,
    "triviaqa": 2,
    "musique": 2,
    "bamboogle": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def stable_key(source: str, question: str, index: int) -> tuple[str, int]:
    payload = f"searchr1-p1-v1\0{source}\0{normalize_question(question)}".encode()
    return hashlib.sha256(payload).hexdigest(), index


def select_rows(frame: pd.DataFrame, quotas: dict[str, int], excluded_questions: set[str]) -> pd.DataFrame:
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


def record_id(split: str, row: pd.Series) -> str:
    original_index = int(row["extra_info"]["index"])
    return f"{split}-{row['data_source']}-{original_index}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {split: args.source_dir / f"{split}.parquet" for split in ("train", "test")}
    frames = {split: pd.read_parquet(path) for split, path in source_paths.items()}

    train = select_rows(frames["train"], TRAIN_QUOTAS, set())
    train_questions = {normalize_question(str(item["question"])) for item in train["env_kwargs"]}
    test = select_rows(frames["test"], TEST_QUOTAS, train_questions)

    output_paths = {"train": args.output_dir / "train.parquet", "test": args.output_dir / "test.parquet"}
    train.to_parquet(output_paths["train"], index=False)
    test.to_parquet(output_paths["test"], index=False)

    records = []
    documents = []
    for split, selected in (("train", train), ("test", test)):
        for _, row in selected.iterrows():
            rid = record_id(split, row)
            env_kwargs = to_builtin(row["env_kwargs"])
            answers = env_kwargs["ground_truth"]["target"]
            records.append(
                {
                    "id": rid,
                    "split": split,
                    "source": str(row["data_source"]),
                    "original_index": int(row["extra_info"]["index"]),
                    "question": env_kwargs["question"],
                    "answers": answers,
                }
            )
            answer_text = "; ".join(map(str, answers))
            documents.append(
                {
                    "id": f"fixture-{rid}",
                    "title": f"P1 fixture evidence for {rid}",
                    "text": f"Question: {env_kwargs['question']}\nReference answer: {answer_text}",
                    "contents": f"P1 fixture evidence for {rid}\nQuestion: {env_kwargs['question']}\nReference answer: {answer_text}",
                    "fixture_only": True,
                    "ground_truth_derived": True,
                }
            )

    records_path = args.output_dir / "records.jsonl"
    corpus_path = args.output_dir / "fixture_corpus.jsonl"
    records_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records))
    corpus_path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in documents))

    manifest = {
        "schema_version": 1,
        "upstream_dataset": {
            "repo_id": "PeterJinGo/nq_hotpotqa_train",
            "revision": "b7d80abfee334a7a91cb377544f09180d58b34f6",
            "last_modified": "2025-03-13T13:17:36+00:00",
            "license": "not declared in the Hugging Face dataset metadata; manual verification required",
            "raw_files": {
                "train.parquet": {"bytes": 355663891, "lfs_sha256": "c3cc21e862a8469105de666101578cbff23cdc77e91a803cef102622c89cc4f6"},
                "test.parquet": {"bytes": 70370337, "lfs_sha256": "30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc"},
            },
        },
        "code_revisions": {
            "root": "c3b946272c70dea17744988fa3607d834e2bbf1e",
            "verl_agent": "20bd331bdbc9026a5668e11362178e10ab7400c8",
        },
        "selection_algorithm": "per-source quota, SHA256(searchr1-p1-v1\\0source\\0normalized_question), ascending",
        "selection_is_order_independent": True,
        "train_test_normalized_question_overlap": 0,
        "source": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(frames[split]),
            }
            for split, path in source_paths.items()
        },
        "outputs": {
            "train": {"path": str(output_paths["train"]), "sha256": sha256_file(output_paths["train"]), "rows": len(train)},
            "test": {"path": str(output_paths["test"]), "sha256": sha256_file(output_paths["test"]), "rows": len(test)},
            "records": {"path": str(records_path), "sha256": sha256_file(records_path), "rows": len(records)},
            "fixture_corpus": {"path": str(corpus_path), "sha256": sha256_file(corpus_path), "rows": len(documents)},
        },
        "fixture_corpus_policy": {
            "ground_truth_derived": True,
            "allowed_use": "protocol, timeout, formatting, and reward smoke tests only",
            "forbidden_use": "benchmark scores, model-quality claims, or comparison with Search-R1 results",
        },
        "records": records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"created {len(train)} train and {len(test)} test smoke rows at {args.output_dir}")


if __name__ == "__main__":
    main()
