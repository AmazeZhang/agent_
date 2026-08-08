"""Export a verl fsdp2 checkpoint's LoRA weights to PEFT adapter format.

verl 0.8.0 saves LoRA-trained checkpoints as verl shards
(actor/model_world_size_1_rank_0.pt) plus a huggingface/ dir that contains
only tokenizer+config — no standalone adapter files. This script extracts
the lora_A/lora_B tensors and writes adapter_model.safetensors +
adapter_config.json so the adapter can be loaded with PeftModel and served
through vLLM's --enable-lora path (WP6 A4: checkpoint load -> evaluation).

Usage:
    python scripts/export_lora_adapter.py \
        <checkpoint>/global_step_N/actor/model_world_size_1_rank_0.pt \
        <out_adapter_dir> \
        --base-model <hf-cache/base-model-path>

Self-check: after export, loads the adapter onto the base model and prints
how many LoRA params differ from a zero-init (i.e. the delta is real).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

MODEL = "/media/imc/data/yzy/agent/project2/hf-cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard", type=Path, help="path to model_world_size_1_rank_0.pt")
    ap.add_argument("out_dir", type=Path, help="output adapter directory")
    ap.add_argument("--base-model", default=MODEL, help="hf-cache path of the base model")
    ap.add_argument("--check", action="store_true", help="load the exported adapter and report the LoRA delta")
    args = ap.parse_args()

    sd = torch.load(args.shard, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"unexpected shard type: {type(sd).__name__}")

    # GRPO (FSDP2) checkpoints save LoRA weights as DTensor subclasses; the
    # SFT (FSDP1) ones save plain tensors. safetensors cannot write DTensors
    # (invalid python storage), so localize first — with world size 1 the
    # Shard(dim=0) placement yields the full tensor.
    from torch.distributed.tensor import DTensor

    adapter: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k:
            if isinstance(v, DTensor):
                v = v.to_local()
            new = k.removeprefix("base_model.model.").replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
            adapter[new] = v
    if not adapter:
        raise SystemExit("no LoRA tensors found in shard")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(adapter, args.out_dir / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "base_model_name_or_path": args.base_model,
    }
    (args.out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    print(f"exported {len(adapter)} LoRA tensors -> {args.out_dir}")

    if args.check:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(base, args.out_dir, torch_dtype=torch.bfloat16)
        # How many LoRA weights differ from the PEFT zero-init default?
        zeros = sum(v.shape.numel() for v in adapter.values() if v.abs().max() == 0)
        total = sum(v.shape.numel() for v in adapter.values())
        nz = total - zeros
        print(f"[check] loaded adapter OK: params={total} zero={zeros} non-zero={nz}")
        print(f"[check] sample delta norm: lora_A[0]={adapter.get('model.layers.0.self_attn.q_proj.lora_A.weight', torch.zeros(1)).norm().item():.6f}")


if __name__ == "__main__":
    main()
