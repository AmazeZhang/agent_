#!/usr/bin/env python3
"""Publish an immutable official-only subset rich in visual tools and recovery."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_official_sft_subset import canonical_image_path, sha256_file  # noqa: E402

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
PARENT_ROOT = (
    PROJECT_DATA
    / "datasets/processed/search-vl-sft-wiki-en-official-960-safe-v2-r2c1c460-c5120"
)
PARENT_DATA = PARENT_ROOT / "wiki_en_official_960_safe.json"
PARENT_DATA_SHA256 = "571c9c59a02309e8962d10ac0a0fdb14d86aa2c54cd8c9f86f4cfcbfa8e964a5"
DATASET_NAME = "wiki_en_official_toolrich"
DATA_NAME = f"{DATASET_NAME}.json"
VISUAL_TOOLS = {"crop", "layout_parsing", "super_resolution", "sharpen"}
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def called_tools(row: dict[str, Any]) -> list[str]:
    names = []
    for message in row["conversations"]:
        if message["from"] != "gpt":
            continue
        match = TOOL_CALL_RE.search(str(message["value"]))
        if match is not None:
            names.append(str(json.loads(match.group(1))["name"]))
    return names


def has_real_tool_error(row: dict[str, Any]) -> bool:
    return any(
        message["from"] == "observation"
        and "Tool execution error:" in str(message["value"])
        for message in row["conversations"]
    )


def select_rows(rows: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any], list[str], bool]]:
    selected = []
    for index, row in enumerate(rows):
        names = called_tools(row)
        error = has_real_tool_error(row)
        if VISUAL_TOOLS.intersection(names) or error:
            selected.append((index, row, names, error))
    if not selected:
        raise ValueError("official tool-rich selection is empty")
    return selected


def dataset_info() -> dict[str, object]:
    return {
        DATASET_NAME: {
            "file_name": DATA_NAME,
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


def build(output: Path) -> dict[str, Any]:
    destination = output.resolve()
    if not destination.is_relative_to((PROJECT_DATA / "datasets/processed").resolve()):
        raise ValueError("output escaped the processed dataset root")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    if sha256_file(PARENT_DATA) != PARENT_DATA_SHA256:
        raise ValueError("parent official 960-safe data hash changed")
    parent_manifest_path = PARENT_ROOT / "manifest.json"
    with PARENT_DATA.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    with parent_manifest_path.open(encoding="utf-8") as handle:
        parent_manifest = json.load(handle)
    if len(rows) != 960 or parent_manifest.get("rows_modified") != 0:
        raise ValueError("parent is not the frozen official 960-safe dataset")
    selected = select_rows(rows)
    parent_indices = [item[0] for item in selected]
    source_indices = [parent_manifest["selected_indices"][index] for index in parent_indices]
    selected_rows = [item[1] for item in selected]
    staging = destination.with_name(f".{destination.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    staging.mkdir(parents=True)
    try:
        data_path = staging / DATA_NAME
        data_path.write_text(
            json.dumps(selected_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        info_path = staging / "dataset_info.json"
        info_path.write_text(
            json.dumps(dataset_info(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        copied: set[str] = set()
        for row in selected_rows:
            for value in row["images"]:
                relative = canonical_image_path(value)
                key = str(relative)
                if key in copied:
                    continue
                source = PARENT_ROOT.joinpath(*relative.parts)
                target = staging.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as input_handle, target.open("xb") as output_handle:
                    shutil.copyfileobj(input_handle, output_handle, length=4 << 20)
                copied.add(key)
        call_counts = Counter(name for _, _, names, _ in selected for name in names)
        row_counts = Counter(
            name for _, _, names, _ in selected for name in set(names)
        )
        first_call_counts = Counter(
            names[0] for _, _, names, _ in selected if names
        )
        index_bytes = json.dumps(parent_indices, separators=(",", ":")).encode()
        manifest = {
            "schema_version": 1,
            "status": "official-toolrich-sft-ready",
            "dataset_name": DATASET_NAME,
            "sample_size": len(selected_rows),
            "selected_image_files": len(copied),
            "selected_parent_indices": parent_indices,
            "selected_parent_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "selected_source_indices": source_indices,
            "dataset_sha256": sha256_file(data_path),
            "dataset_info_sha256": sha256_file(info_path),
            "parent_dataset_root": str(PARENT_ROOT),
            "parent_dataset_sha256": PARENT_DATA_SHA256,
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "source_revision": parent_manifest["source_revision"],
            "selection_rule": "all rows calling a visual tool, union all rows containing exact Tool execution error observations",
            "visual_tools": sorted(VISUAL_TOOLS),
            "tool_call_counts": dict(sorted(call_counts.items())),
            "tool_row_counts": dict(sorted(row_counts.items())),
            "first_call_counts": dict(sorted(first_call_counts.items())),
            "real_tool_error_rows": sum(error for _, _, _, error in selected),
            "duplicates_added": 0,
            "rows_modified": 0,
            "cutoff_len": 5120,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
        return manifest
    except Exception:
        raise RuntimeError(f"tool-rich build failed; staging preserved at {staging}")


def main() -> int:
    manifest = build(parse_args().output)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
