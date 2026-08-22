#!/usr/bin/env python3
"""Publish a non-overwriting v8 dataset with the official-style local tool contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
SOURCE_ROOT = PROJECT_DATA / "datasets/processed/wit-rl-boundary-v7"
OUTPUT_ROOT = PROJECT_DATA / "datasets/processed/wit-rl-protocol-v8"
SYSTEM = (
    "Use one tool call per turn. Pass the literal handle img_1 to image_search. "
    "Image search returns entity candidates only. Use text_search with a natural-language "
    "query to verify evidence. Retry only when error.retryable=true. Do not invent evidence. "
    "Finish with exactly one <response> containing Title and Evidence lines."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def wrap_final(text: str) -> str:
    return f"<response>\n{text.strip()}\n</response>"


def convert_call(value: str, titles: dict[str, str]) -> str:
    call = json.loads(value)
    if call.get("name") != "text_lookup":
        return value
    entity_id = str(call.get("arguments", {}).get("entity_id", ""))
    if entity_id not in titles:
        raise KeyError(f"text_lookup entity missing from task candidates: {entity_id}")
    return json.dumps(
        {"name": "text_search", "arguments": {"q": titles[entity_id], "top_k": 3}},
        separators=(",", ":"),
    )


def convert_record(record: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    titles = {str(item["entity_id"]): str(item["title"]) for item in task["retrieval_results"]}
    converted = dict(record)
    converted["system"] = SYSTEM
    conversations = []
    for message in record["conversations"]:
        copied = dict(message)
        if copied.get("from") == "function":
            copied["value"] = convert_call(str(copied["value"]), titles)
        elif copied.get("from") == "gpt":
            copied["value"] = wrap_final(str(copied["value"]))
        conversations.append(copied)
    converted["conversations"] = conversations
    return converted


def build(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if not source.is_relative_to((PROJECT_DATA / "datasets/processed").resolve()):
        raise ValueError("source must be a project4 processed dataset")
    if not output.is_relative_to((PROJECT_DATA / "datasets/processed").resolve()):
        raise ValueError("output must be a project4 processed dataset")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    with (source / "manifest.json").open(encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if source_manifest.get("status") != "rl-boundary-ready":
        raise ValueError("source must be the immutable v7 boundary suite")
    tasks = [json.loads(line) for line in (source / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line]
    tasks_by_image = {str(task["query_image"]): task for task in tasks}
    staging = output.with_name(f".{output.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    try:
        staging.mkdir(parents=True)
        shutil.copytree(source / "images", staging / "images")
        for split in ("train", "dev", "test"):
            records = json.loads((source / f"sft_{split}.json").read_text(encoding="utf-8"))
            converted = []
            for record in records:
                image = str(record["images"][0])
                converted.append(convert_record(record, tasks_by_image[image]))
            (staging / f"sft_{split}.json").write_text(
                json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        converted_tasks = []
        for task in tasks:
            copied = dict(task)
            copied["oracle_steps"] = ["text_search" if step == "text_lookup" else step for step in task["oracle_steps"]]
            copied["gold_final"] = wrap_final(str(task["gold_final"]))
            copied["final_response_wrapper"] = "response-v1"
            converted_tasks.append(copied)
        with (staging / "tasks.jsonl").open("x", encoding="utf-8") as handle:
            for task in converted_tasks:
                handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
        info = json.loads((source / "dataset_info.json").read_text(encoding="utf-8"))
        renamed = {key.replace("_v6", "_v8"): value for key, value in info.items()}
        (staging / "dataset_info.json").write_text(json.dumps(renamed, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = dict(source_manifest)
        manifest.update({
            "schema_version": 3,
            "status": "rl-protocol-ready",
            "purpose": "local-agentic-official-protocol-sft-rl",
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "tasks_sha256": sha256_file(staging / "tasks.jsonl"),
            "sft_sha256": {split: sha256_file(staging / f"sft_{split}.json") for split in ("train", "dev", "test")},
            "dataset_info_sha256": sha256_file(staging / "dataset_info.json"),
            "tool_protocol": "official-local-v1",
            "tool_observation_schema": "boundary-compact-v1",
            "tool_observation_fields": {"image_search": ["entity_id", "title", "similarity"], "text_search": ["entity_id", "title", "source", "summary"]},
            "final_response_format": "<response>Title: <exact title>\\nEvidence: <first sentence-or-no-match></response>",
            "maximum_agent_turns": 8,
            "protocol_note": "frozen local Wiki replaces web backend; tool loop and final response wrapper align with official agent contract",
        })
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(output)
    except Exception as error:
        raise RuntimeError(f"v8 build failed; staging preserved at {staging}") from error
    return output


def main() -> int:
    args = parse_args()
    output = build(args.source, args.output)
    print(json.dumps({"output": str(output), "status": "rl-protocol-ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
