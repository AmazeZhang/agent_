"""WP4: GPU smoke — vLLM serves Qwen2.5-Coder-3B-Instruct on one free GPU.

Usage (in tmux): CUDA_VISIBLE_DEVICES=1 python scripts/smoke_gpu.py

Verifies the full serving path the training stack depends on: model loading
from the HF cache, vLLM engine init, and a real generation.
"""

from __future__ import annotations

import os
import sys

from pathlib import Path

# Same attention backend the verl training stack pins; our flash-attn wheel
# was rebuilt for torch 2.11, so no flashinfer JIT (which needs ninja+ nvcc
# at engine init) is ever triggered.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

MODEL = Path("/media/imc/data/yzy/agent/project2/hf-cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5")


def main() -> None:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "0" in gpu.split(","):
        raise SystemExit("REFUSED: set CUDA_VISIBLE_DEVICES to a free GPU (1-7), never 0")
    if not MODEL.exists():
        raise SystemExit(f"model missing: {MODEL}")

    from vllm import LLM, SamplingParams

    print(f"[smoke_gpu] loading model on gpu={gpu} ...")
    llm = LLM(model=str(MODEL), tensor_parallel_size=1, gpu_memory_utilization=0.6, max_model_len=4096)
    out = llm.generate(
        ["def fib(n):\n    "],
        SamplingParams(max_tokens=64, temperature=0.0),
    )
    text = out[0].outputs[0].text
    print("[smoke_gpu] generation OK:")
    print(text[:300])
    print("[smoke_gpu] DONE")


if __name__ == "__main__":
    main()
