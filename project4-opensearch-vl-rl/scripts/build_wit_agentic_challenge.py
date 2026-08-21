#!/usr/bin/env python3
"""Build an immutable harder local-agent challenge from the verified WIT pilot."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from local_retrieval import (
    LocalTextIndex,
    entity_tool_observation,
    text_tool_observation,
)
from local_retrieval.resnet50_encoder import sha256_file
from verify_wit_agentic_pilot import first_sentence, tool_schema

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
SOURCE_ROOT = PROJECT_DATA / "datasets/processed/wit-agentic-pilot-v4"
OUTPUT_ROOT = PROJECT_DATA / "datasets/processed/wit-agentic-challenge-v3"
SYSTEM = (
    "Use one tool call per turn. Pass the literal handle img_1 to image_search. "
    "Image search returns entity candidates only; text_lookup supplies answer evidence. "
    "Retry a tool only when its error says retryable=true. Do not invent missing evidence. "
    "The final response must contain exactly two lines named Title and Evidence."
)
NO_MATCH_FINAL = "Title: NO_MATCH\nEvidence: No local evidence found."
TRANSIENT_OBSERVATION = "Tool execution result:\n" + json.dumps(
    {"error": {"code": "TRANSIENT_FAILURE", "retryable": True}},
    sort_keys=True,
)
COUNTS = {
    "train": {
        "clean": 8,
        "candidate-conflict": 32,
        "transient-tool-failure": 24,
        "no-match": 16,
    },
    "dev": {
        "clean": 2,
        "candidate-conflict": 8,
        "transient-tool-failure": 6,
        "no-match": 4,
    },
    "test": {
        "clean": 2,
        "candidate-conflict": 8,
        "transient-tool-failure": 6,
        "no-match": 4,
    },
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]{6,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def load_source(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    required = {
        "status": "retrieval-verified",
        "image_runtime_handle": "img_1",
        "evidence_extraction": "first_terminal_punctuation_or_360_characters",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"source manifest {key} is not {expected!r}")
    with (root / "tasks.jsonl").open(encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    if len(tasks) != 120:
        raise ValueError("source must contain the fixed 120-task pilot")
    return manifest, tasks


def conflict_keyword(task: dict[str, Any]) -> str | None:
    results = task["retrieval_results"]
    if len(results) < 2:
        return None
    target = results[1]
    blocked = set(WORD_RE.findall(str(target["title"]).casefold()))
    other_text = " ".join(
        str(item.get("summary", "")) for item in results if item is not target
    ).casefold()
    candidates = {
        word.casefold()
        for word in WORD_RE.findall(str(target.get("summary", "")))
        if word.casefold() not in blocked and word.casefold() not in other_text
    }
    if not candidates:
        return None
    return min(candidates, key=lambda word: (-len(word), word))


def function_message(name: str, arguments: dict[str, object]) -> dict[str, str]:
    return {
        "from": "function",
        "value": json.dumps(
            {"name": name, "arguments": arguments}, separators=(",", ":")
        ),
    }


def base_record(
    prompt: str, image_name: str, conversations: list[dict[str, str]]
) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": f"<image> {prompt}"},
            *conversations,
        ],
        "images": [image_name],
        "system": SYSTEM,
        "tools": tool_schema(),
    }


def clean_example(
    source: dict[str, Any],
    image_name: str,
    text_results: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    visual_result = source["retrieval_results"][0]
    result = (text_results or {}).get(str(visual_result["entity_id"]), visual_result)
    prompt = (
        "The image has runtime handle img_1. Identify its closest Wikipedia subject, "
        "look up that entity, and report the exact title and first evidence sentence. "
        "If image_search returns no candidates, return Title: NO_MATCH and "
        "Evidence: No local evidence found."
    )
    calls = [
        function_message("image_search", {"image": "img_1", "top_k": 3}),
        {
            "from": "observation",
            "value": entity_tool_observation(source["retrieval_results"]),
        },
        function_message("text_lookup", {"entity_id": result["entity_id"]}),
        {"from": "observation", "value": text_tool_observation([result])},
        {"from": "gpt", "value": source["gold_final"]},
    ]
    task = {
        **source,
        "query_image": image_name,
        "user_prompt": prompt,
        "task_type": "clean",
        "image_search_failures_before_success": 0,
        "oracle_steps": ["image_search", "text_lookup", "final"],
    }
    return base_record(prompt, image_name, calls), task


def transient_example(
    source: dict[str, Any],
    image_name: str,
    text_results: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record, task = clean_example(source, image_name, text_results)
    first_call = function_message("image_search", {"image": "img_1", "top_k": 3})
    record["conversations"][1:1] = [
        first_call,
        {"from": "observation", "value": TRANSIENT_OBSERVATION},
    ]
    task.update(
        {
            "task_type": "transient-tool-failure",
            "image_search_failures_before_success": 1,
            "oracle_steps": ["image_search", "image_search", "text_lookup", "final"],
        }
    )
    return record, task


def conflict_example(
    source: dict[str, Any],
    image_name: str,
    keyword: str,
    text_results: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = source["retrieval_results"]
    first, target = results[0], results[1]
    evidence_by_id = text_results or {}
    first_text = evidence_by_id.get(str(first["entity_id"]), first)
    target_text = evidence_by_id.get(str(target["entity_id"]), target)
    evidence = first_sentence(str(target_text["summary"]))
    final = f"Title: {target_text['title']}\nEvidence: {evidence}"
    prompt = (
        f"The image has runtime handle img_1. Search for its candidate entities. The closest "
        f"visual candidate may be a distractor: select the candidate whose text evidence contains "
        f"the keyword `{keyword}`. Report that candidate's exact title and first evidence sentence."
    )
    calls = [
        function_message("image_search", {"image": "img_1", "top_k": 3}),
        {"from": "observation", "value": entity_tool_observation(results)},
        function_message("text_lookup", {"entity_id": first["entity_id"]}),
        {"from": "observation", "value": text_tool_observation([first_text])},
        function_message("text_lookup", {"entity_id": target["entity_id"]}),
        {"from": "observation", "value": text_tool_observation([target_text])},
        {"from": "gpt", "value": final},
    ]
    task = {
        **source,
        "query_image": image_name,
        "user_prompt": prompt,
        "task_type": "candidate-conflict",
        "conflict_keyword": keyword,
        "gold_title": target_text["title"],
        "gold_evidence_sentence": evidence,
        "gold_final": final,
        "image_search_failures_before_success": 0,
        "oracle_steps": ["image_search", "text_lookup", "text_lookup", "final"],
    }
    return base_record(prompt, image_name, calls), task


def no_match_example(
    split: str, index: int, image_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "The image has runtime handle img_1. Search the local corpus. If no entity candidates "
        "are returned, do not guess and reply exactly with Title: NO_MATCH and "
        "Evidence: No local evidence found."
    )
    calls = [
        function_message("image_search", {"image": "img_1", "top_k": 3}),
        {"from": "observation", "value": entity_tool_observation([])},
        {"from": "gpt", "value": NO_MATCH_FINAL},
    ]
    task = {
        "task_id": f"no-match-{split}-{index:03d}",
        "split": split,
        "query_image": image_name,
        "user_prompt": prompt,
        "task_type": "no-match",
        "retrieval_results": [],
        "gold_title": "NO_MATCH",
        "gold_evidence_sentence": "No local evidence found.",
        "gold_final": NO_MATCH_FINAL,
        "image_search_failures_before_success": 0,
        "oracle_steps": ["image_search", "final"],
        "synthetic_safety_probe": True,
    }
    return base_record(prompt, image_name, calls), task


def dataset_info() -> dict[str, Any]:
    return {
        f"wit_agentic_{split}_v1": {
            "file_name": f"sft_{split}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images",
                "system": "system",
                "tools": "tools",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "observation_tag": "observation",
                "function_tag": "function",
            },
        }
        for split in ("train", "dev", "test")
    }


def build(source_root: Path, output: Path) -> Path:
    source_manifest, tasks = load_source(source_root)
    destination = output.resolve()
    if not destination.is_relative_to(PROJECT_DATA.resolve()):
        raise ValueError("output must be below the project data root")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite challenge: {destination}")
    staging = destination.with_name(f".{destination.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    (staging / "images").mkdir(parents=True)
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in COUNTS}
    published: list[dict[str, Any]] = []
    try:
        by_split = {
            split: sorted(
                (task for task in tasks if task["split"] == split),
                key=lambda item: item["task_id"],
            )
            for split in COUNTS
        }
        with LocalTextIndex(Path(source_manifest["text_index"])) as text_index:
            for split, quotas in COUNTS.items():
                available = by_split[split]
                eligible = [(task, conflict_keyword(task)) for task in available]
                conflicts = [
                    (task, keyword) for task, keyword in eligible if keyword is not None
                ][: quotas["candidate-conflict"]]
                if len(conflicts) != quotas["candidate-conflict"]:
                    raise ValueError(f"not enough conflict candidates for {split}")
                used = {task["task_id"] for task, _ in conflicts}
                remaining = [task for task in available if task["task_id"] not in used]
                clean = remaining[: quotas["clean"]]
                transient = remaining[
                    quotas["clean"] : quotas["clean"] + quotas["transient-tool-failure"]
                ]
                if (
                    len(clean) != quotas["clean"]
                    or len(transient) != quotas["transient-tool-failure"]
                ):
                    raise ValueError(f"not enough source entities for {split}")
                examples: list[
                    tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]
                ] = []
                selected = [task for task, _ in conflicts] + clean + transient
                text_results = {}
                for task in selected:
                    for result in task["retrieval_results"][:2]:
                        entity_id = str(result["entity_id"])
                        looked_up = text_index.lookup(entity_id)
                        if looked_up is None:
                            raise ValueError(f"missing text evidence for {entity_id}")
                        text_results[entity_id] = looked_up
                for task, keyword in conflicts:
                    examples.append(
                        (
                            *conflict_example(
                                task,
                                f"images/{task['task_id']}-conflict.jpg",
                                keyword,
                                text_results,
                            ),
                            task,
                        )
                    )
                for task in clean:
                    examples.append(
                        (
                            *clean_example(
                                task,
                                f"images/{task['task_id']}-clean.jpg",
                                text_results,
                            ),
                            task,
                        )
                    )
                for task in transient:
                    examples.append(
                        (
                            *transient_example(
                                task,
                                f"images/{task['task_id']}-transient.jpg",
                                text_results,
                            ),
                            task,
                        )
                    )
                for record, task, source in examples:
                    source_image = source_root / str(source["query_image"])
                    shutil.copyfile(source_image, staging / task["query_image"])
                    records[split].append(record)
                    published.append(task)
                for index in range(quotas["no-match"]):
                    image_name = f"images/no-match-{split}-{index:03d}.png"
                    image = Image.new("RGB", (224, 224), (127, 127, 127))
                    draw = ImageDraw.Draw(image)
                    for offset in range(0, 224, 32):
                        shade = (offset * 17 + index * 29) % 255
                        draw.rectangle(
                            (offset, 0, min(offset + 15, 223), 223),
                            fill=(shade, 255 - shade, 127),
                        )
                    image.save(staging / image_name, format="PNG")
                    record, task = no_match_example(split, index, image_name)
                    records[split].append(record)
                    published.append(task)
        for split, items in records.items():
            with (staging / f"sft_{split}.json").open("x", encoding="utf-8") as handle:
                json.dump(items, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        with (staging / "tasks.jsonl").open("x", encoding="utf-8") as handle:
            for task in published:
                handle.write(
                    json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n"
                )
        with (staging / "dataset_info.json").open("x", encoding="utf-8") as handle:
            json.dump(
                dataset_info(), handle, ensure_ascii=False, indent=2, sort_keys=True
            )
            handle.write("\n")
        split_counts = Counter(task["split"] for task in published)
        type_counts = Counter(task["task_type"] for task in published)
        manifest = {
            "schema_version": 1,
            "status": "challenge-ready",
            "purpose": "local-agentic-sft-rl-challenge",
            "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
            "source_tasks_sha256": sha256_file(source_root / "tasks.jsonl"),
            "sft_sha256": {
                split: sha256_file(staging / f"sft_{split}.json")
                for split in ("train", "dev", "test")
            },
            "dataset_info_sha256": sha256_file(staging / "dataset_info.json"),
            "text_index": source_manifest["text_index"],
            "text_lookup_summary_max_characters": 360,
            "records": len(published),
            "split_unit": "entity_id-or-synthetic-probe-id",
            "split_counts": dict(sorted(split_counts.items())),
            "task_type_counts": dict(sorted(type_counts.items())),
            "image_runtime_handle": "img_1",
            "image_observation_contains_text_summary": False,
            "final_response_format": "Title: <exact title>\\nEvidence: <first sentence-or-no-match>",
            "evidence_extraction": "first_terminal_punctuation_or_360_characters",
            "maximum_agent_turns": 4,
            "network_required": False,
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        staging.rename(destination)
    except (KeyError, OSError, TypeError, ValueError):
        raise RuntimeError(f"challenge build failed; staging preserved at {staging}")
    return destination


def main() -> int:
    args = parse_args()
    output = build(args.source.resolve(), args.output)
    print(
        json.dumps({"output": str(output), "status": "challenge-ready"}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
