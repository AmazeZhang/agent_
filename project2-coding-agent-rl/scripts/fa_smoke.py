"""WP4: functional flash-attn forward pass on one GPU (1-7 only).

Run in tmux: CUDA_VISIBLE_DEVICES=1 python scripts/fa_smoke.py
Tests that the rebuilt flash_attn_2_cuda extension actually executes on
this GPU (sm_89) — the wheel built for sm_80/90/100/120 must run via
forward-compatible cubins or PTX JIT.
"""

import os
import sys

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if "0" in gpu.split(",") or not gpu:
    raise SystemExit("REFUSED: set CUDA_VISIBLE_DEVICES to a free GPU (1-7), never 0")

import torch

import flash_attn

print(f"[fa_smoke] gpu={gpu} torch={torch.__version__} flash_attn={flash_attn.__version__}")

q = torch.randn(2, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
out = flash_attn.flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
print(f"[fa_smoke] flash_attn_func OK -> {tuple(out.shape)}")

out2 = flash_attn.flash_attn_func(q, q, q, causal=False)
torch.cuda.synchronize()
print(f"[fa_smoke] non-causal OK -> {tuple(out2.shape)}")
print("[fa_smoke] DONE")
