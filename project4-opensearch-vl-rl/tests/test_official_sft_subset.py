"""CPU-only tests for deterministic official SFT subset construction."""

import importlib.util
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = PROJECT_ROOT / "scripts/build_official_sft_subset.py"
    spec = importlib.util.spec_from_file_location("build_official_sft_subset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load subset builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_row(image: str, suffix: str) -> dict:
    declarations = [
        {"type": "function", "function": {"name": "image_search"}},
        {"type": "function", "function": {"name": "text_search"}},
    ]
    return {
        "conversations": [
            {"from": "human", "value": f"<image>question {suffix}"},
            {
                "from": "gpt",
                "value": '<tool_call>{"name":"image_search","arguments":{"url":"img_1"}}</tool_call>',
            },
            {"from": "observation", "value": "result"},
            {
                "from": "gpt",
                "value": '<tool_call>{"name":"text_search","arguments":{"q":"x"}}</tool_call>',
            },
            {"from": "observation", "value": "passage"},
            {"from": "gpt", "value": "<response>answer</response>"},
        ],
        "images": [image],
        "system": "system",
        "tools": json.dumps(declarations),
    }


def main() -> None:
    builder = load_module()
    with tempfile.TemporaryDirectory(prefix="p4-official-sft.") as temporary:
        root = Path(temporary)
        payload = root / "payload"
        (payload / "images").mkdir(parents=True)
        rows = []
        for index in range(4):
            relative = f"images/{index}.jpg"
            (payload / relative).write_bytes(f"image-{index}".encode())
            rows.append(make_row(relative, str(index)))

        audit = builder.audit_rows(rows, payload)
        assert audit["rows"] == 4
        assert audit["image_references"] == 4
        assert audit["tool_calls"] == {"image_search": 4, "text_search": 4}

        first = builder.select_indices(rows, 2, "fixed-seed")
        second = builder.select_indices(rows, 2, "fixed-seed")
        assert first == second and len(first) == 2

        (payload / "images/4.jpg").write_bytes(b"image-4")
        incomplete = make_row("images/4.jpg", "incomplete")
        incomplete["conversations"] = incomplete["conversations"][:-1]
        mixed = rows + [incomplete]
        incomplete_audit = builder.audit_rows(mixed, payload)
        assert incomplete_audit["trainable_rows"] == 4
        assert incomplete_audit["excluded_rows"] == [
            {
                "index": 4,
                "reason": "conversation_does_not_end_with_gpt",
                "terminal_role": "observation",
            }
        ]
        assert 4 not in builder.select_indices(
            mixed, 4, "fixed-seed", builder.trainable_indices(mixed)
        )

        output = root / "subset"
        manifest = builder.publish_subset(
            rows,
            first,
            payload,
            output,
            {"source_revision": "fixed", "source_sha256": "0" * 64},
        )
        assert manifest["sample_size"] == 2
        assert len(list((output / "images").glob("*.jpg"))) == 2
        assert len(json.loads((output / "wiki_en_official_1000.json").read_text())) == 2
        assert json.loads((output / "dataset_info.json").read_text())[
            "wiki_en_official_1000"
        ]["formatting"] == "sharegpt"

        bad = make_row("../escape.jpg", "bad")
        try:
            builder.audit_rows([bad], payload)
        except ValueError as error:
            assert "image" in str(error)
        else:
            raise AssertionError("unsafe image path was accepted")

    print("official SFT subset tests: PASS")


if __name__ == "__main__":
    main()
