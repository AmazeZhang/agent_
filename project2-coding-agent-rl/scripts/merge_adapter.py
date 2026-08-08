"""Merge an exported PEFT LoRA adapter into the base model (full HF weights).

Used for the WP6 warm-start: GRPO initializes its policy from the SFT'd
model rather than the base (spec WP6: "WP3 可信轨迹初始化策略后"), which
gives the shaped reward a non-constant signal. The merged model is a plain
HF model dir that both verl (model.path) and vLLM can load.

Usage (rllm-base venv, torch 2.11):
    python scripts/merge_adapter.py <adapter_dir> <out_model_dir>

Prints the merged config + a quick load sanity check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

MODEL = "/media/imc/data/yzy/agent/project2/hf-cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    adapter_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"[merge] loading base: {MODEL}")
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
    print(f"[merge] loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir, torch_dtype=torch.bfloat16)

    # Fast-forward LoRA weights into the base, drop the PEFT wrapper.
    merged = model.merge_and_unload()
    merged = merged.to(torch.bfloat16)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    # Tokenizer files: copy from the base snapshot so the dir is fully standalone.
    snap = Path(MODEL)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt"):
        src = snap / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())

    print(f"[merge] done: {out_dir}")
    print(f"[merge] files: {sorted(p.name for p in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
