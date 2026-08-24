"""Publish the official SFT rows that are safe under the frozen engineering cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from build_official_sft_subset import canonical_image_path, sha256_file
from run_official_sft import DATASET_ROOT, PROJECT_DATA

DATA_NAME = "wiki_en_official_960_safe.json"
DATASET_NAME = "wiki_en_official_960_safe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    report_path = args.alignment_report.resolve()
    if not output.is_relative_to((PROJECT_DATA / "datasets/processed").resolve()):
        raise ValueError("output escaped the processed dataset root")
    if not report_path.is_relative_to(DATASET_ROOT.resolve()):
        raise ValueError("alignment report must belong to the frozen parent dataset")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")

    parent_manifest_path = DATASET_ROOT / "manifest.json"
    parent_data_path = DATASET_ROOT / "wiki_en_official_1000.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_rows = json.loads(parent_data_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("checked") != 1000 or report.get("dataset_size") != 1000:
        raise ValueError("alignment report does not cover the full frozen parent dataset")
    if report.get("cutoff_len") != 5120 or report.get("zero_supervision_count") != 0:
        raise ValueError("alignment report violates the frozen cutoff/supervision contract")

    mismatches = report.get("mismatches")
    if not isinstance(mismatches, list) or len(mismatches) != report.get("mismatch_count"):
        raise ValueError("alignment mismatch list is malformed")
    excluded_dataset_indices: list[int] = []
    excluded_source_indices: list[int] = []
    for mismatch in mismatches:
        dataset_index = int(mismatch["dataset_index"])
        source_index = int(mismatch["source_index"])
        if not 0 <= dataset_index < len(parent_rows):
            raise ValueError(f"alignment dataset index out of range: {dataset_index}")
        if parent_manifest["selected_indices"][dataset_index] != source_index:
            raise ValueError("alignment source index no longer matches the parent manifest")
        if int(mismatch["image_tokens"]) == int(mismatch["image_features"]):
            raise ValueError("alignment report marks an aligned row as mismatched")
        excluded_dataset_indices.append(dataset_index)
        excluded_source_indices.append(source_index)
    if excluded_dataset_indices != sorted(set(excluded_dataset_indices)):
        raise ValueError("alignment mismatch indices must be unique and sorted")

    excluded = set(excluded_dataset_indices)
    safe_rows = [row for index, row in enumerate(parent_rows) if index not in excluded]
    safe_source_indices = [
        source_index
        for index, source_index in enumerate(parent_manifest["selected_indices"])
        if index not in excluded
    ]
    if len(safe_rows) != 960 or len(safe_source_indices) != 960:
        raise ValueError("expected exactly 960 cutoff-safe official rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing to overwrite staging: {staging}")
    staging.mkdir()
    data_path = staging / DATA_NAME
    data_path.write_text(
        json.dumps(safe_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    info_path = staging / "dataset_info.json"
    info_path.write_text(
        json.dumps(dataset_info(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    copied: set[str] = set()
    for row in safe_rows:
        for value in row["images"]:
            relative = canonical_image_path(value)
            key = str(relative)
            if key in copied:
                continue
            source = DATASET_ROOT.joinpath(*relative.parts)
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as input_handle, target.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=4 << 20)
            copied.add(key)

    source_indices_bytes = json.dumps(safe_source_indices, separators=(",", ":")).encode()
    manifest = {
        "dataset_name": DATASET_NAME,
        "sample_size": len(safe_rows),
        "selected_image_files": len(copied),
        "selected_indices": safe_source_indices,
        "selected_indices_sha256": hashlib.sha256(source_indices_bytes).hexdigest(),
        "dataset_sha256": sha256_file(data_path),
        "dataset_info_sha256": sha256_file(info_path),
        "parent_dataset_root": str(DATASET_ROOT),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "parent_dataset_sha256": sha256_file(parent_data_path),
        "source_revision": parent_manifest["source_revision"],
        "source_sha256": parent_manifest["source_sha256"],
        "selection_seed": parent_manifest["selection_seed"],
        "alignment_report": str(report_path),
        "alignment_report_sha256": sha256_file(report_path),
        "cutoff_len": 5120,
        "excluded_dataset_indices": excluded_dataset_indices,
        "excluded_source_indices": excluded_source_indices,
        "exclusion_reason": "image_token_feature_mismatch_after_cutoff",
        "rows_modified": 0,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
