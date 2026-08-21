#!/usr/bin/env python3
"""Stream-audit the immutable Search-VL RL JSONL without loading images."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = ("question", "answer", "images")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def length_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values, default=0),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
        "mean": round(sum(values) / len(values), 2) if values else 0.0,
    }


def classify_language(text: str) -> str:
    return "contains_cjk" if CJK_RE.search(text) else "no_cjk"


def sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def audit(path: Path) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    image_extension_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    image_list_lengths: list[int] = []
    boxed_answers = 0
    invalid_rows = 0
    rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(item, dict):
                invalid_rows += 1
                continue

            rows += 1
            key_counts.update(str(key) for key in item)
            for field in REQUIRED_FIELDS:
                if field not in item or item[field] in (None, "", []):
                    missing_counts[field] += 1

            question = item.get("question", "")
            answer = item.get("answer", "")
            images = item.get("images", [])
            if not isinstance(question, str):
                question = str(question)
                invalid_rows += 1
            if not isinstance(answer, str):
                answer = str(answer)
                invalid_rows += 1
            if not isinstance(images, list):
                images = []
                invalid_rows += 1

            question_lengths.append(len(question))
            answer_lengths.append(len(answer))
            image_list_lengths.append(len(images))
            language_counts[classify_language(question)] += 1
            question_counts[question] += 1
            if "\\boxed{" in answer:
                boxed_answers += 1

            dataset_value = item.get("dataset", "<missing>")
            dataset_counts[str(dataset_value)] += 1
            for image in images:
                image_text = str(image)
                image_counts[image_text] += 1
                suffix = Path(image_text).suffix.lower() or "<none>"
                image_extension_counts[suffix] += 1

    return {
        "schema_version": 1,
        "source": {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "rows": rows,
        "invalid_rows": invalid_rows,
        "required_field_missing": sorted_counter(missing_counts),
        "field_presence": sorted_counter(key_counts),
        "dataset_counts": sorted_counter(dataset_counts),
        "question_language_heuristic": sorted_counter(language_counts),
        "question_length_chars": length_summary(question_lengths),
        "answer_length_chars": length_summary(answer_lengths),
        "image_count_per_row": length_summary(image_list_lengths),
        "image_extension_counts": sorted_counter(image_extension_counts),
        "unique_questions": len(question_counts),
        "duplicate_question_rows": sum(count - 1 for count in question_counts.values()),
        "unique_image_references": len(image_counts),
        "duplicate_image_reference_rows": sum(
            count - 1 for count in image_counts.values()
        ),
        "answers_containing_boxed": boxed_answers,
    }


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = audit(input_path)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
