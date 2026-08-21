#!/usr/bin/env python3
"""Verify WIT query retrieval and publish executable multi-turn SFT records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from local_retrieval import (  # noqa: E402
    ExactVisualIndex,
    LocalTextIndex,
    encode_pil_images,
    entity_tool_observation,
    load_resnet50_v1,
    text_tool_observation,
)
from local_retrieval.image_search_backend import resolve_local_image  # noqa: E402
from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402
from reembed_wit_pilot import require_managed_environment  # noqa: E402

PROJECT_DATA_ROOT = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--text-index", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--minimum-self-similarity", type=float, default=0.90)
    return parser.parse_args()


def first_sentence(text: str, *, maximum: int = 360) -> str:
    compact = " ".join(text.split()).strip(' "')
    for index, character in enumerate(compact):
        if character in ".!?" and index >= 39:
            return compact[: index + 1]
    return compact[:maximum].rstrip()


def tool_schema() -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "image_search",
                "description": (
                    "Find local Wikipedia entity candidates. The provided query image is "
                    "registered under the literal runtime handle img_1."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["image"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "text_lookup",
                "description": "Look up local Wikipedia evidence by entity ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"entity_id": {"type": "string"}},
                    "required": ["entity_id"],
                },
            },
        },
    ]
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"))


def make_record(
    task: dict[str, Any],
    visual_results: list[dict[str, object]],
    text_result: dict[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = first_sentence(str(text_result["summary"]))
    final = f"Title: {text_result['title']}\nEvidence: {evidence}"
    image_call = {
        "name": "image_search",
        "arguments": {"image": "img_1", "top_k": 3},
    }
    text_call = {
        "name": "text_lookup",
        "arguments": {"entity_id": str(text_result["entity_id"])},
    }
    sft = {
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image> The provided image has runtime handle img_1. Identify the "
                    "Wikipedia subject most closely matching this image. "
                    "Then use text_lookup on the selected entity and report its exact title "
                    "and the first evidence sentence."
                ),
            },
            {
                "from": "function",
                "value": json.dumps(image_call, separators=(",", ":")),
            },
            {
                "from": "observation",
                "value": entity_tool_observation(visual_results),
            },
            {
                "from": "function",
                "value": json.dumps(text_call, separators=(",", ":")),
            },
            {
                "from": "observation",
                "value": text_tool_observation([text_result]),
            },
            {"from": "gpt", "value": final},
        ],
        "images": [task["query_image"]],
        "system": (
            "Use one tool call per turn. Pass the literal handle img_1 to image_search; "
            "do not replace it with a filename or image description. Image search returns "
            "entity candidates only; text_lookup supplies answer evidence. Do not invent "
            "missing evidence."
        ),
        "tools": tool_schema(),
    }
    published_task = {
        **task,
        "gold_title": str(text_result["title"]),
        "gold_evidence_sentence": evidence,
        "gold_final": final,
        "oracle_steps": ["image_search", "text_lookup", "final"],
        "retrieval_top1_similarity": float(visual_results[0]["similarity"]),
        "retrieval_results": visual_results,
    }
    return sft, published_task


def load_candidates(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "retrieval-unverified":
        raise ValueError("candidate manifest must be retrieval-unverified")
    with (root / "candidates.jsonl").open(encoding="utf-8") as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    if len(tasks) != manifest.get("records") or not tasks:
        raise ValueError("candidate record count does not match manifest")
    entity_ids = [str(task["entity_id"]) for task in tasks]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("candidate entity IDs are not unique")
    return manifest, tasks


def verify_and_publish(
    candidate_root: Path,
    index_root: Path,
    text_index_path: Path,
    weights_path: Path,
    output: Path,
    *,
    batch_size: int,
    top_k: int,
    minimum_self_similarity: float,
) -> Path:
    import torch

    require_managed_environment(os.environ)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("pilot verification requires exactly one managed CUDA device")
    if not 1 <= batch_size <= 128 or not 1 <= top_k <= 5:
        raise ValueError("batch_size/top_k are outside bounded pilot limits")
    if not 0.0 <= minimum_self_similarity <= 1.0:
        raise ValueError("minimum_self_similarity must be between 0 and 1")
    destination = output.resolve()
    if not destination.is_relative_to(PROJECT_DATA_ROOT.resolve()):
        raise ValueError("output must be below the project data root")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite verified pilot: {destination}")

    candidate_manifest, tasks = load_candidates(candidate_root)
    index = ExactVisualIndex(index_root)
    model, preprocess, encoder = load_resnet50_v1(weights_path, device="cuda:0")
    if not str(index.manifest["corpus_revision"]).endswith(
        f"+{encoder['weights_sha256']}"
    ):
        raise ValueError("visual index/query encoder revisions do not match")

    vectors = np.empty((len(tasks), index.vectors.shape[1]), dtype=np.float32)
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        images = []
        for task in batch:
            path = resolve_local_image(Path(task["query_image"]), candidate_root)
            with Image.open(path) as image:
                image.load()
                images.append(image.convert("RGB"))
        encoded = encode_pil_images(model, preprocess, images, device="cuda:0")
        vectors[start : start + len(encoded)] = encoded
        print(f"encoded_candidates={start + len(encoded)}/{len(tasks)}", flush=True)

    results = index.search_batch(vectors, top_k=top_k)
    failures = []
    prepared: list[tuple[dict[str, Any], dict[str, Any]]] = []
    with LocalTextIndex(text_index_path) as text_index:
        for task, visual_results in zip(tasks, results, strict=True):
            top1 = visual_results[0]
            reasons = []
            if top1["entity_id"] != task["entity_id"]:
                reasons.append("top1-entity-mismatch")
            if float(top1["similarity"]) < minimum_self_similarity:
                reasons.append("below-minimum-self-similarity")
            text_result = text_index.lookup(str(task["entity_id"]))
            if text_result is None:
                reasons.append("text-evidence-missing")
            elif str(text_result["title"]) != str(task["title"]):
                reasons.append("text-title-mismatch")
            if reasons:
                failures.append(
                    {
                        "task_id": task["task_id"],
                        "expected_entity_id": task["entity_id"],
                        "top1_entity_id": top1["entity_id"],
                        "top1_similarity": top1["similarity"],
                        "reasons": reasons,
                    }
                )
                continue
            assert text_result is not None
            prepared.append(make_record(task, visual_results, text_result))
    if failures:
        raise RuntimeError(
            f"{len(failures)}/{len(tasks)} candidates failed retrieval verification; "
            f"first failures={failures[:3]}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.building.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging directory: {staging}")
    (staging / "images").mkdir(parents=True)
    try:
        sft_by_split: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ("train", "dev", "test")
        }
        published_tasks = []
        for sft, task in prepared:
            source_image = resolve_local_image(
                Path(task["query_image"]), candidate_root
            )
            target_image = staging / task["query_image"]
            shutil.copyfile(source_image, target_image)
            sft_by_split[task["split"]].append(sft)
            published_tasks.append(task)
        for split, records in sft_by_split.items():
            with (staging / f"sft_{split}.json").open("x", encoding="utf-8") as handle:
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        with (staging / "tasks.jsonl").open("x", encoding="utf-8") as handle:
            for task in published_tasks:
                handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
        dataset_info = {
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
            for split in sft_by_split
        }
        with (staging / "dataset_info.json").open("x", encoding="utf-8") as handle:
            json.dump(dataset_info, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        split_counts = Counter(task["split"] for task in published_tasks)
        manifest = {
            "schema_version": 1,
            "status": "retrieval-verified",
            "purpose": "local-agentic-sft-rl-pilot",
            "candidate_manifest_sha256": sha256_file(candidate_root / "manifest.json"),
            "candidate_records_sha256": sha256_file(candidate_root / "candidates.jsonl"),
            "candidate_manifest": candidate_manifest,
            "visual_index": str(index_root.resolve()),
            "visual_index_revision": index.manifest["corpus_revision"],
            "text_index": str(text_index_path.resolve()),
            "encoder_weights_sha256": encoder["weights_sha256"],
            "verification": {
                "required_top1_entity_match": True,
                "minimum_self_similarity": minimum_self_similarity,
                "failures": 0,
            },
            "split_unit": "entity_id",
            "split_counts": dict(sorted(split_counts.items())),
            "records": len(published_tasks),
            "image_observation_contains_text_summary": False,
            "image_runtime_handle": "img_1",
            "oracle_steps": ["image_search", "text_lookup", "final"],
        }
        with (staging / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        staging.rename(destination)
    except Exception as error:
        raise RuntimeError(f"pilot publication failed; staging preserved at {staging}") from error
    return destination


def main() -> int:
    args = parse_args()
    output = verify_and_publish(
        args.candidates.resolve(),
        args.index.resolve(),
        args.text_index.resolve(),
        args.weights.resolve(),
        args.output,
        batch_size=args.batch_size,
        top_k=args.top_k,
        minimum_self_similarity=args.minimum_self_similarity,
    )
    print(json.dumps({"output": str(output), "status": "retrieval-verified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
