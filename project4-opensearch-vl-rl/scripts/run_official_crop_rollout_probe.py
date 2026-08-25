#!/usr/bin/env python3
"""Probe SFT-50 crop behaviour with the exact official SFT tool contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import LocalImageSearchBackend, LocalTextIndex  # noqa: E402
from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402
from scripts.evaluate_local_agent import (  # noqa: E402
    MODEL_ROOT,
    RUN_ROOT,
    evaluate_task,
    require_managed_run,
    validate_adapter,
)

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
SFT_ROOT = (
    PROJECT_DATA
    / "datasets/processed/search-vl-sft-wiki-en-official-960-safe-v2-r2c1c460-c5120"
)
SFT_JSON = SFT_ROOT / "wiki_en_official_960_safe.json"
SFT_SHA256 = "571c9c59a02309e8962d10ac0a0fdb14d86aa2c54cd8c9f86f4cfcbfa8e964a5"
PILOT_MANIFEST = PROJECT_DATA / "datasets/processed/wit-agentic-pilot-v4/manifest.json"
ENCODER_WEIGHTS = PROJECT_DATA / "models/torchvision-cache/hub/checkpoints/resnet50-0676ba61.pth"
PROBE_ROWS = (15, 71, 90)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-turns", type=int, default=3)
    return parser.parse_args()


def first_tool_call(row: dict[str, Any]) -> dict[str, Any]:
    first_assistant = next(
        message for message in row["conversations"] if message["from"] == "gpt"
    )
    match = TOOL_CALL_RE.search(str(first_assistant["value"]))
    if match is None:
        raise ValueError("probe row has no first assistant tool call")
    call = json.loads(match.group(1))
    if call.get("name") != "crop" or not isinstance(call.get("arguments"), dict):
        raise ValueError("probe row does not begin with an official crop call")
    return call


def load_probe_rows() -> list[tuple[int, dict[str, Any], dict[str, Any]]]:
    if sha256_file(SFT_JSON) != SFT_SHA256:
        raise ValueError("official SFT probe data no longer matches its frozen hash")
    with SFT_JSON.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    selected = []
    for index in PROBE_ROWS:
        row = rows[index]
        call = first_tool_call(row)
        image_path = (SFT_ROOT / row["images"][0]).resolve(strict=True)
        if not image_path.is_relative_to(SFT_ROOT.resolve()):
            raise ValueError("probe image escaped the frozen SFT root")
        selected.append((index, row, call))
    return selected


def summarize_probe(result: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    turns = result["turns"]
    first = turns[0].get("tool_call") if turns else None
    crop_succeeded = bool(
        turns
        and first
        and first.get("name") == "crop"
        and "Image cropped successfully. New image ID: img_2."
        in str(turns[0].get("observation", ""))
    )
    followup_calls = [turn.get("tool_call") for turn in turns[1:] if turn.get("tool_call")]
    uses_img2 = any(
        call.get("arguments", {}).get("url") == "img_2"
        or call.get("arguments", {}).get("image") == "img_2"
        for call in followup_calls
    )
    live_img2_search = any(
        call.get("name") == "image_search"
        and call.get("arguments", {}).get("url") == "img_2"
        and turn.get("image_search_cache") is False
        for turn in turns[1:]
        if (call := turn.get("tool_call"))
    )
    return {
        "first_tool_crop": bool(first and first.get("name") == "crop"),
        "first_call_exact_official_expert": first == expected,
        "crop_succeeded": crop_succeeded,
        "followup_uses_img2": uses_img2,
        "live_image_search_img2": live_img2_search,
        "fatal": result["fatal"],
    }


def main() -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    if not 64 <= args.max_new_tokens <= 256 or not 2 <= args.maximum_turns <= 4:
        raise ValueError("crop probe is bounded to 64..256 tokens and 2..4 turns")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    adapter = validate_adapter(args.adapter)
    if adapter is None:
        raise ValueError("crop probe requires the frozen SFT adapter")
    output = args.output.resolve()
    if output != (run_dir / "crop_rollout_probe.json").resolve() or output.exists():
        raise ValueError("output must be the absent managed crop_rollout_probe.json")
    selected = load_probe_rows()
    with PILOT_MANIFEST.open(encoding="utf-8") as handle:
        pilot = json.load(handle)
    visual = LocalImageSearchBackend(
        Path(pilot["visual_index"]),
        ENCODER_WEIGHTS,
        device="cpu",
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ROOT, local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ROOT,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    results = []
    with LocalTextIndex(Path(pilot["text_index"])) as text_index:
        for row_index, row, expected_call in selected:
            prompt = str(row["conversations"][0]["value"])
            prompt = re.sub(r"^\s*<image>\s*", "", prompt, count=1)
            task = {
                "task_id": f"official-sft-row-{row_index}",
                "task_type": "official-first-crop-probe",
                "split": "probe",
                "query_image": row["images"][0],
                "user_prompt": prompt,
                "gold_title": "PROBE_ONLY",
                "gold_evidence_sentence": "PROBE_ONLY",
                "final_response_wrapper": "response-v1",
                "allow_official_think_prefix": True,
                "oracle_steps": ["crop", "final"],
            }
            result = evaluate_task(
                model,
                processor,
                task,
                text_index,
                max_new_tokens=args.max_new_tokens,
                dataset_root=SFT_ROOT,
                observation_format="official-provider-v1",
                maximum_turns=args.maximum_turns,
                tool_protocol="official-local-multimodal-v1",
                visual_lookup=visual.search_image,
            )
            summary = summarize_probe(result, expected_call)
            results.append(
                {
                    "official_sft_row": row_index,
                    "expected_first_call": expected_call,
                    "summary": summary,
                    "result": result,
                }
            )
            print(json.dumps({"row": row_index, **summary}, sort_keys=True), flush=True)
    report = {
        "schema_version": 1,
        "purpose": "rollout-only official first-crop behaviour probe",
        "physical_gpu": physical_gpu,
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "official_sft_sha256": SFT_SHA256,
        "probe_rows": list(PROBE_ROWS),
        "tool_protocol": "official-local-multimodal-v1",
        "system_and_tools_source": "contracts/official_sft_tool_contract.json",
        "image_search_backend": "live-local-resnet50-v1-cpu",
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "maximum_turns": args.maximum_turns,
        },
        "results": results,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
