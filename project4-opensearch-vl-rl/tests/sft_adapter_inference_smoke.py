"""Load a local project4 LoRA checkpoint and run deterministic VL inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ROOT = Path(
    "/media/imc/data/yzy/agent/project4-opensearch-vl-rl/models/Qwen3-VL-8B-Instruct"
)
RUN_ROOT = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl/runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = args.adapter.resolve()
    if not adapter.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("adapter must come from a project4 managed Run")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("adapter_model.safetensors is missing")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one managed logical CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "0":
        raise RuntimeError("physical GPU0 must never be selected")

    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 48, 176, 176), fill="red")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": "Name the dominant color in one lowercase word.",
                },
            ],
        }
    ]

    processor = AutoProcessor.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=False,
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
    if not model.active_adapters:
        raise RuntimeError("PEFT adapter is not active")

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    suffix = generated[:, inputs["input_ids"].shape[1] :]
    answer = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
    torch.cuda.synchronize()

    result = {
        "active_adapters": list(model.active_adapters),
        "adapter": str(adapter),
        "answer": answer,
        "input_tokens": inputs["input_ids"].shape[1],
        "output_tokens": suffix.shape[1],
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result, sort_keys=True))
    if not answer:
        raise RuntimeError("SFT adapter generated an empty answer")


if __name__ == "__main__":
    main()
