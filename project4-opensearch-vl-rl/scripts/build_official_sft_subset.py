"""Audit official Search-VL SFT rows and publish a deterministic subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

DATA_ROOT = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
REQUIRED_TOOLS = {"image_search", "text_search"}
ROLES = {"human", "gpt", "observation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", default="opensearch-vl-official-sft-v1")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def below_data_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError(f"path must be below project data root: {resolved}")
    return resolved


def canonical_image_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"invalid image path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"non-canonical image path: {value!r}")
    if not path.parts or path.parts[0] != "images":
        raise ValueError(f"image must be below images/: {value!r}")
    return path


def parse_declared_tools(value: object) -> set[str]:
    if not isinstance(value, str):
        raise ValueError("tools must be a JSON string")
    payload = json.loads(value)
    if not isinstance(payload, list) or not payload:
        raise ValueError("tools JSON must be a non-empty list")
    names = []
    for item in payload:
        try:
            name = item["function"]["name"]
        except (KeyError, TypeError) as error:
            raise ValueError("malformed function tool declaration") from error
        if not isinstance(name, str) or not name:
            raise ValueError("tool name must be a non-empty string")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("duplicate declared tool name")
    return set(names)


def called_tools(conversations: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for message in conversations:
        if not isinstance(message, dict) or message.get("from") not in ROLES:
            raise ValueError("invalid conversation role")
        value = message.get("value")
        if not isinstance(value, str):
            raise ValueError("conversation value must be a string")
        for encoded in TOOL_CALL_RE.findall(value):
            call = json.loads(encoded)
            name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("malformed tool call")
            counts[name] += 1
    return counts


def audit_rows(rows: object, payload_root: Path) -> dict[str, Any]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("source JSON must be a non-empty list")
    payload_root = payload_root.resolve()
    declared_reference: str | None = None
    declared_names: set[str] | None = None
    calls: Counter[str] = Counter()
    image_references: list[str] = []
    excluded_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not an object")
        if not {"conversations", "images", "system", "tools"}.issubset(row):
            raise ValueError(f"row {index} lacks required fields")
        if not isinstance(row["system"], str) or not row["system"]:
            raise ValueError(f"row {index} has invalid system prompt")
        if not isinstance(row["conversations"], list) or not row["conversations"]:
            raise ValueError(f"row {index} has invalid conversations")
        if row["conversations"][0].get("from") != "human":
            raise ValueError(f"row {index} does not start with a human message")
        if row["conversations"][-1].get("from") != "gpt":
            excluded_rows.append(
                {
                    "index": index,
                    "reason": "conversation_does_not_end_with_gpt",
                    "terminal_role": row["conversations"][-1].get("from"),
                }
            )

        tools_value = row["tools"]
        names = parse_declared_tools(tools_value)
        if declared_reference is None:
            declared_reference = tools_value
            declared_names = names
        elif tools_value != declared_reference:
            raise ValueError(f"row {index} changes the frozen tool declaration")

        row_calls = called_tools(row["conversations"])
        unknown = set(row_calls) - names
        if unknown:
            raise ValueError(f"row {index} calls undeclared tools: {sorted(unknown)}")
        calls.update(row_calls)

        images = row["images"]
        if not isinstance(images, list) or not images:
            raise ValueError(f"row {index} has no images")
        for value in images:
            relative = canonical_image_path(value)
            source = (payload_root / Path(*relative.parts)).resolve()
            if not source.is_relative_to(payload_root) or not source.is_file():
                raise ValueError(f"row {index} image is missing: {relative}")
            image_references.append(str(relative))

    assert declared_names is not None
    missing = REQUIRED_TOOLS - declared_names
    if missing:
        raise ValueError(f"official declarations lack required tools: {sorted(missing)}")
    if not REQUIRED_TOOLS.issubset(calls):
        raise ValueError("source rows do not exercise both official search tools")
    if len(image_references) != len(set(image_references)):
        raise ValueError("source data reuses an image path across rows")
    return {
        "rows": len(rows),
        "trainable_rows": len(rows) - len(excluded_rows),
        "excluded_rows": excluded_rows,
        "image_references": len(image_references),
        "declared_tools": sorted(declared_names),
        "tool_calls": dict(sorted(calls.items())),
    }


def trainable_indices(rows: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if row["conversations"][-1].get("from") == "gpt"
    ]


def select_indices(
    rows: list[dict[str, Any]],
    sample_size: int,
    seed: str,
    candidates: list[int] | None = None,
) -> list[int]:
    candidates = list(range(len(rows))) if candidates is None else candidates
    if not 1 <= sample_size <= len(candidates):
        raise ValueError(f"sample size must be between 1 and {len(candidates)}")
    ranked = []
    for index in candidates:
        row = rows[index]
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(f"{seed}\0{index}\0{canonical}".encode()).digest()
        ranked.append((digest, index))
    return [index for _, index in sorted(ranked)[:sample_size]]


def dataset_info(file_name: str) -> dict[str, Any]:
    return {
        "wiki_en_official_1000": {
            "file_name": file_name,
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
            },
        }
    }


def publish_subset(
    rows: list[dict[str, Any]],
    indices: list[int],
    payload_root: Path,
    output: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    staging.mkdir()

    selected = [rows[index] for index in indices]
    data_name = "wiki_en_official_1000.json"
    data_path = staging / data_name
    data_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    info_path = staging / "dataset_info.json"
    info_path.write_text(
        json.dumps(dataset_info(data_name), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    copied: set[str] = set()
    for row in selected:
        for value in row["images"]:
            relative = canonical_image_path(value)
            key = str(relative)
            if key in copied:
                continue
            source = payload_root.joinpath(*relative.parts)
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as input_handle, target.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=4 << 20)
            copied.add(key)

    indices_json = json.dumps(indices, separators=(",", ":")).encode()
    manifest = {
        **provenance,
        "dataset_name": "wiki_en_official_1000",
        "sample_size": len(selected),
        "selected_image_files": len(copied),
        "selected_indices": indices,
        "selected_indices_sha256": hashlib.sha256(indices_json).hexdigest(),
        "dataset_sha256": sha256_file(data_path),
        "dataset_info_sha256": sha256_file(info_path),
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    return manifest


def main() -> int:
    args = parse_args()
    source_json = below_data_root(args.source_json)
    payload_root = below_data_root(args.payload_root)
    output = below_data_root(args.output)
    actual_sha256 = sha256_file(source_json)
    if actual_sha256 != args.source_sha256.lower():
        raise ValueError(
            f"source SHA256 mismatch: expected={args.source_sha256} actual={actual_sha256}"
        )
    with source_json.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    audit = audit_rows(rows, payload_root)
    indices = select_indices(rows, args.sample_size, args.seed, trainable_indices(rows))
    manifest = publish_subset(
        rows,
        indices,
        payload_root,
        output,
        {
            "source_json": str(source_json),
            "source_revision": args.source_revision,
            "source_sha256": actual_sha256,
            "selection_seed": args.seed,
            "source_audit": audit,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"official SFT subset build failed: {error}", file=sys.stderr)
        raise
