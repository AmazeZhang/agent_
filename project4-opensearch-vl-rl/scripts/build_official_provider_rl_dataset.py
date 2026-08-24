#!/usr/bin/env python3
"""Publish unchanged v8 QVA tasks behind the official-shaped local provider."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
SOURCE_ROOT = PROJECT_DATA / "datasets/processed/wit-rl-protocol-v8"
OUTPUT_ROOT = PROJECT_DATA / "datasets/processed/wit-rl-official-provider-v10"
EXPECTED_SOURCE_TASKS_SHA256 = (
    "2ccdb0ef507ebbd20dfba54c199a724ab9056a868856d9291dd65949325cce55"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def build(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    allowed = (PROJECT_DATA / "datasets/processed").resolve()
    if not source.is_relative_to(allowed) or not output.is_relative_to(allowed):
        raise ValueError("source/output must stay inside project4 processed datasets")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    if sha256_file(source / "tasks.jsonl") != EXPECTED_SOURCE_TASKS_SHA256:
        raise ValueError("v8 QVA tasks do not match the frozen source hash")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tool_protocol") != "official-local-v1":
        raise ValueError("source is not the frozen official-local-v1 task suite")
    staging = output.with_name(f".{output.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    try:
        staging.mkdir(parents=True)
        shutil.copytree(source / "images", staging / "images")
        for name in ("tasks.jsonl", "dataset_info.json"):
            shutil.copy2(source / name, staging / name)
        derived = dict(manifest)
        derived.pop("sft_sha256", None)
        derived.update(
            {
                "schema_version": 4,
                "status": "rl-official-provider-ready",
                "source_manifest_sha256": sha256_file(source / "manifest.json"),
                "source_tasks_sha256": EXPECTED_SOURCE_TASKS_SHA256,
                "tasks_sha256": sha256_file(staging / "tasks.jsonl"),
                "rows_modified": 0,
                "tool_protocol": "official-local-v1",
                "tool_observation_schema": "official-provider-v1",
                "tool_observation_fields": {
                    "image_search": ["title", "source"],
                    "text_search": ["title", "source", "summary"],
                },
                "provider_note": (
                    "QVA rows are byte-identical to v8; only the hidden execution layer "
                    "now projects frozen retrieval through OfficialLocalSearchProvider."
                ),
            }
        )
        (staging / "manifest.json").write_text(
            json.dumps(derived, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except Exception as error:
        raise RuntimeError(f"v9 build failed; staging preserved at {staging}") from error
    return output


def main() -> int:
    args = parse_args()
    output = build(args.source, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": "rl-official-provider-ready",
                "tasks_sha256": sha256_file(output / "tasks.jsonl"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
