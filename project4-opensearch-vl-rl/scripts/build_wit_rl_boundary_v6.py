#!/usr/bin/env python3
"""Build an immutable decision-boundary RL task suite from verified WIT data."""

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

from build_wit_agentic_challenge import (
    NO_MATCH_FINAL,
    TRANSIENT_OBSERVATION,
    base_record,
    dataset_info,
    function_message,
)
from verify_wit_agentic_pilot import first_sentence

from local_retrieval import (
    LocalTextIndex,
    entity_tool_observation,
    text_tool_observation,
)
from local_retrieval.resnet50_encoder import sha256_file

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
SOURCE_ROOT = PROJECT_DATA / "datasets/processed/wit-agentic-pilot-v4"
OUTPUT_ROOT = PROJECT_DATA / "datasets/processed/wit-rl-boundary-v6"
WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]{5,}")
COUNTS = {
    "train": {"dual-clue-rank2": 24, "dual-clue-rank3": 24, "transient-dual-clue": 16, "no-match-after-retry": 16},
    "dev": {"dual-clue-rank2": 6, "dual-clue-rank3": 6, "transient-dual-clue": 4, "no-match-after-retry": 4},
    "test": {"dual-clue-rank2": 6, "dual-clue-rank3": 6, "transient-dual-clue": 4, "no-match-after-retry": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def word_set(result: dict[str, Any]) -> set[str]:
    title_words = {word.casefold() for word in WORD_RE.findall(str(result["title"]))}
    return {
        word.casefold() for word in WORD_RE.findall(str(result["summary"]))
    } - title_words


def boundary_clues(
    results: list[dict[str, Any]], target_rank: int
) -> tuple[list[str], str] | None:
    if len(results) < 3 or target_rank not in {2, 3}:
        return None
    sets = [word_set(result) for result in results[:3]]
    target_index = target_rank - 1
    other_words = set().union(
        *(words for index, words in enumerate(sets) if index != target_index)
    )
    positives = sorted(
        sets[target_index] - other_words, key=lambda word: (-len(word), word)
    )
    exclusions = sorted(
        sets[0] - sets[target_index], key=lambda word: (-len(word), word)
    )
    if len(positives) < 2 or not exclusions:
        return None
    return positives[:2], exclusions[0]


def load_source(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("status") != "retrieval-verified"
        or manifest.get("image_runtime_handle") != "img_1"
        or manifest.get("evidence_extraction")
        != "first_terminal_punctuation_or_360_characters"
    ):
        raise ValueError("source is not the fixed retrieval-verified pilot")
    with (root / "tasks.jsonl").open(encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    if len(tasks) != 120:
        raise ValueError("source must contain 120 tasks")
    return manifest, tasks


def lookup_candidates(
    task: dict[str, Any], text_index: LocalTextIndex
) -> list[dict[str, Any]]:
    results = []
    for candidate in task["retrieval_results"][:3]:
        result = text_index.lookup(str(candidate["entity_id"]))
        if result is None:
            raise ValueError(f"missing text evidence: {candidate['entity_id']}")
        results.append(result)
    if len(results) != 3:
        raise ValueError("boundary task requires exactly three candidates")
    return results


def boundary_example(
    source: dict[str, Any],
    image_name: str,
    text_results: list[dict[str, Any]],
    *,
    target_rank: int,
    transient: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clues = boundary_clues(text_results, target_rank)
    if clues is None:
        raise ValueError("task does not have auditable boundary clues")
    positives, exclusion = clues
    target = text_results[target_rank - 1]
    evidence = first_sentence(str(target["summary"]))
    final = f"Title: {target['title']}\nEvidence: {evidence}"
    prompt = (
        "The image has runtime handle img_1. Inspect candidate evidence in visual-rank "
        f"order. Select the single candidate whose evidence contains both `{positives[0]}` "
        f"and `{positives[1]}`, and does not contain `{exclusion}`. Do not select a candidate "
        "from its title or visual rank alone. Report its exact title and first evidence sentence."
    )
    calls: list[dict[str, str]] = []
    if transient:
        calls.extend(
            [
                function_message("image_search", {"image": "img_1", "top_k": 3}),
                {"from": "observation", "value": TRANSIENT_OBSERVATION},
            ]
        )
    calls.extend(
        [
            function_message("image_search", {"image": "img_1", "top_k": 3}),
            {
                "from": "observation",
                "value": entity_tool_observation(source["retrieval_results"][:3]),
            },
        ]
    )
    for result in text_results[:target_rank]:
        calls.extend(
            [
                function_message("text_lookup", {"entity_id": result["entity_id"]}),
                {"from": "observation", "value": text_tool_observation([result])},
            ]
        )
    calls.append({"from": "gpt", "value": final})
    task_type = "transient-dual-clue" if transient else f"dual-clue-rank{target_rank}"
    oracle = (["image_search"] if transient else []) + ["image_search"]
    oracle.extend(["text_lookup"] * target_rank)
    oracle.append("final")
    task = {
        **source,
        "query_image": image_name,
        "user_prompt": prompt,
        "task_type": task_type,
        "gold_entity_id": str(target["entity_id"]),
        "gold_title": str(target["title"]),
        "gold_evidence_sentence": evidence,
        "gold_final": final,
        "target_rank": target_rank,
        "positive_clues": positives,
        "excluded_clue": exclusion,
        "image_search_failures_before_success": 1 if transient else 0,
        "oracle_steps": oracle,
        "reward_evidence_contract": {
            "source": "text_lookup_observation",
            "entity_id": str(target["entity_id"]),
            "exact_first_sentence": evidence,
        },
    }
    return base_record(prompt, image_name, calls), task


def no_match_after_retry_example(
    split: str, index: int, image_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        "The image has runtime handle img_1. Search the local corpus. Retry once if the first "
        "error explicitly says retryable=true. If the successful retry returns no candidates, "
        "do not guess and return the exact NO_MATCH response."
    )
    calls = [
        function_message("image_search", {"image": "img_1", "top_k": 3}),
        {"from": "observation", "value": TRANSIENT_OBSERVATION},
        function_message("image_search", {"image": "img_1", "top_k": 3}),
        {"from": "observation", "value": entity_tool_observation([])},
        {"from": "gpt", "value": NO_MATCH_FINAL},
    ]
    task = {
        "task_id": f"boundary-no-match-{split}-{index:03d}",
        "split": split,
        "query_image": image_name,
        "user_prompt": prompt,
        "task_type": "no-match-after-retry",
        "retrieval_results": [],
        "gold_entity_id": None,
        "gold_title": "NO_MATCH",
        "gold_evidence_sentence": "No local evidence found.",
        "gold_final": NO_MATCH_FINAL,
        "image_search_failures_before_success": 1,
        "oracle_steps": ["image_search", "image_search", "final"],
        "synthetic_safety_probe": True,
        "reward_evidence_contract": {"source": "empty-image-search-result"},
    }
    return base_record(prompt, image_name, calls), task


def select_tasks(
    tasks: list[dict[str, Any]], text_index: LocalTextIndex
) -> list[tuple[dict[str, Any], str, int, bool, list[dict[str, Any]]]]:
    selected = []
    candidate_ids_by_split: dict[str, set[str]] = {}
    for split, quotas in COUNTS.items():
        available = sorted(
            (task for task in tasks if task["split"] == split),
            key=lambda task: task["task_id"],
        )
        enriched = [(task, lookup_candidates(task, text_index)) for task in available]
        used_tasks: set[str] = set()
        split_candidates: set[str] = set()
        for task_type, target_rank, transient in (
            ("dual-clue-rank3", 3, False),
            ("dual-clue-rank2", 2, False),
            ("transient-dual-clue", 2, True),
        ):
            needed = quotas[task_type]
            chosen = []
            for task, results in enriched:
                ids = {str(item["entity_id"]) for item in results}
                if task["task_id"] in used_tasks or boundary_clues(results, target_rank) is None:
                    continue
                if any(ids & prior for prior in candidate_ids_by_split.values()):
                    continue
                chosen.append((task, task_type, target_rank, transient, results))
                used_tasks.add(task["task_id"])
                split_candidates.update(ids)
                if len(chosen) == needed:
                    break
            if len(chosen) != needed:
                raise ValueError(f"not enough {task_type} tasks for {split}")
            selected.extend(chosen)
        candidate_ids_by_split[split] = split_candidates
    return selected


def build(source_root: Path, output: Path) -> Path:
    source_manifest, source_tasks = load_source(source_root)
    destination = output.resolve()
    if not destination.is_relative_to(PROJECT_DATA.resolve()):
        raise ValueError("output must be below the project data root")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite boundary suite: {destination}")
    staging = destination.with_name(f".{destination.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    (staging / "images").mkdir(parents=True)
    records: dict[str, list[dict[str, Any]]] = {split: [] for split in COUNTS}
    published: list[dict[str, Any]] = []
    try:
        with LocalTextIndex(Path(source_manifest["text_index"])) as text_index:
            selected = select_tasks(source_tasks, text_index)
        for source, task_type, target_rank, transient, text_results in selected:
            image_name = f"images/{source['task_id']}-{task_type}.jpg"
            record, task = boundary_example(
                source,
                image_name,
                text_results,
                target_rank=target_rank,
                transient=transient,
            )
            shutil.copyfile(source_root / str(source["query_image"]), staging / image_name)
            records[source["split"]].append(record)
            published.append(task)
        for split, quotas in COUNTS.items():
            for index in range(quotas["no-match-after-retry"]):
                image_name = f"images/boundary-no-match-{split}-{index:03d}.png"
                image = Image.new("RGB", (224, 224), (96, 96, 96))
                draw = ImageDraw.Draw(image)
                for offset in range(0, 224, 28):
                    shade = (offset * 11 + index * 31) % 255
                    draw.ellipse(
                        (offset, offset // 2, min(offset + 30, 223), min(offset // 2 + 30, 223)),
                        fill=(shade, 255 - shade, 160),
                    )
                image.save(staging / image_name, format="PNG")
                record, task = no_match_after_retry_example(split, index, image_name)
                records[split].append(record)
                published.append(task)
        for split, items in records.items():
            with (staging / f"sft_{split}.json").open("x", encoding="utf-8") as handle:
                json.dump(items, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        with (staging / "tasks.jsonl").open("x", encoding="utf-8") as handle:
            for task in published:
                handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
        with (staging / "dataset_info.json").open("x", encoding="utf-8") as handle:
            info = dataset_info()
            info = {key.replace("_v1", "_v6"): value for key, value in info.items()}
            json.dump(info, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        split_counts = Counter(task["split"] for task in published)
        type_counts = Counter(task["task_type"] for task in published)
        manifest = {
            "schema_version": 1,
            "status": "rl-boundary-ready",
            "purpose": "local-agentic-decision-boundary-sft-rl",
            "source_manifest_sha256": sha256_file(source_root / "manifest.json"),
            "source_tasks_sha256": sha256_file(source_root / "tasks.jsonl"),
            "tasks_sha256": sha256_file(staging / "tasks.jsonl"),
            "sft_sha256": {
                split: sha256_file(staging / f"sft_{split}.json") for split in COUNTS
            },
            "dataset_info_sha256": sha256_file(staging / "dataset_info.json"),
            "text_index": source_manifest["text_index"],
            "records": len(published),
            "split_counts": dict(sorted(split_counts.items())),
            "task_type_counts": dict(sorted(type_counts.items())),
            "selection_rule": "task-id-order; rank3 then rank2 then transient-rank2; cross-split top3 candidate IDs disjoint",
            "split_unit": "all-top3-candidate-entity-ids-or-synthetic-probe-id",
            "positive_clues": "two target-summary words absent from every other top3 candidate summary and target title",
            "excluded_clue": "one rank1-summary word absent from target summary and rank1 title",
            "image_runtime_handle": "img_1",
            "image_search_top_k_maximum": 3,
            "image_observation_contains_text_summary": False,
            "text_lookup_summary_max_characters": 360,
            "final_response_format": "Title: <exact title>\\nEvidence: <first sentence-or-no-match>",
            "evidence_extraction": "first_terminal_punctuation_or_360_characters",
            "maximum_agent_turns": 5,
            "contains_synthetic_safety_probes": True,
            "fully_synthetic": False,
            "network_required": False,
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        staging.rename(destination)
    except (KeyError, OSError, TypeError, ValueError):
        raise RuntimeError(f"boundary build failed; staging preserved at {staging}")
    return destination


def main() -> int:
    args = parse_args()
    output = build(args.source.resolve(), args.output)
    print(json.dumps({"output": str(output), "status": "rl-boundary-ready"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
