"""Load the pinned local Qwen3-VL 8B base and run one deterministic image turn."""

import json
import os

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ROOT = (
    "/media/imc/data/yzy/agent/project4-opensearch-vl-rl/models/Qwen3-VL-8B-Instruct"
)


def main() -> None:
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
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ROOT,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
    ).eval()
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
        "answer": answer,
        "input_tokens": inputs["input_ids"].shape[1],
        "output_tokens": suffix.shape[1],
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result, sort_keys=True))
    if not answer:
        raise RuntimeError("model generated an empty answer")


if __name__ == "__main__":
    main()
