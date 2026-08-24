#!/usr/bin/env python3
"""Small managed-GPU smoke for the pinned bitsandbytes NF4 kernel."""

from __future__ import annotations

import os
from pathlib import Path

import bitsandbytes as bnb
import torch

RUN_ROOT = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl/runs")


def require_managed_stable_gpu() -> tuple[Path, str]:
    run_id = os.environ.get("PROJECT4_RUN_ID", "")
    run_token = os.environ.get("PROJECT4_RUN_TOKEN", "")
    run_dir = Path(os.environ.get("PROJECT4_RUN_DIR", "/invalid")).resolve()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not run_id or not run_token or run_dir != (RUN_ROOT / run_id).resolve():
        raise RuntimeError("bitsandbytes smoke requires a project4 managed Run")
    if visible in {"", "0", "5"} or "," in visible:
        raise RuntimeError("bitsandbytes smoke requires one stable GPU, excluding GPU0/GPU5")
    return run_dir, visible


def main() -> int:
    _, physical_gpu = require_managed_stable_gpu()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("managed CUDA device is unavailable")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    layer = bnb.nn.Linear4bit(
        4096,
        4096,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
        quant_storage=torch.bfloat16,
    )
    layer.weight.requires_grad_(False)
    layer = layer.to(device)
    value = torch.randn(2, 4096, device=device, dtype=torch.bfloat16, requires_grad=True)
    output = layer(value)
    loss = output.float().square().mean()
    loss.backward()
    if not torch.isfinite(output).all() or value.grad is None or not torch.isfinite(value.grad).all():
        raise RuntimeError("bitsandbytes NF4 forward/backward produced non-finite values")
    print(
        "bitsandbytes NF4 smoke: PASS",
        f"version={bnb.__version__}",
        f"physical_gpu={physical_gpu}",
        f"logical_device={torch.cuda.current_device()}",
        f"name={torch.cuda.get_device_name(0)}",
        f"loss={loss.item():.8f}",
        f"max_allocated={torch.cuda.max_memory_allocated()}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
