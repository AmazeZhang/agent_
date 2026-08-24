"""CPU-only tests for publishing the cutoff-safe official SFT subset."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = PROJECT_ROOT / "scripts/build_alignment_safe_sft_subset.py"
    spec = importlib.util.spec_from_file_location("build_alignment_safe_sft_subset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load alignment-safe SFT builder")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def main() -> None:
    builder = load_module()
    with tempfile.TemporaryDirectory(prefix="p4-alignment-safe-sft.") as temporary:
        project_data = Path(temporary)
        parent = project_data / "datasets/processed/parent"
        (parent / "images").mkdir(parents=True)
        (parent / "images/shared.bin").write_bytes(b"official-image")
        rows = [
            {
                "conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}],
                "images": ["images/shared.bin"],
                "system": "system",
                "tools": "[]",
            }
            for _ in range(1000)
        ]
        parent_data = parent / "wiki_en_official_1000.json"
        parent_data.write_text(json.dumps(rows), encoding="utf-8")
        parent_manifest = {
            "selected_indices": list(range(1000)),
            "source_revision": "fixed-revision",
            "source_sha256": "a" * 64,
            "selection_seed": "fixed-seed",
        }
        (parent / "manifest.json").write_text(json.dumps(parent_manifest), encoding="utf-8")
        mismatches = [
            {
                "dataset_index": index,
                "source_index": index,
                "image_tokens": 1,
                "image_features": 2,
            }
            for index in range(40)
        ]
        report = {
            "checked": 1000,
            "dataset_size": 1000,
            "cutoff_len": 5120,
            "zero_supervision_count": 0,
            "mismatch_count": 40,
            "mismatches": mismatches,
        }
        report_path = parent / "alignment.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        output = project_data / "datasets/processed/safe"
        builder.PROJECT_DATA = project_data
        builder.DATASET_ROOT = parent
        original_argv = sys.argv
        sys.argv = [str(Path(builder.__file__)), "--alignment-report", str(report_path), "--output", str(output)]
        try:
            assert builder.main() == 0
        finally:
            sys.argv = original_argv

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        safe_rows = json.loads((output / builder.DATA_NAME).read_text(encoding="utf-8"))
        assert manifest["sample_size"] == 960
        assert manifest["excluded_dataset_indices"] == list(range(40))
        assert manifest["selected_indices"] == list(range(40, 1000))
        assert manifest["rows_modified"] == 0
        assert len(safe_rows) == 960
        assert (output / "images/shared.bin").read_bytes() == b"official-image"

    print("alignment-safe official SFT subset tests: PASS")


if __name__ == "__main__":
    main()
