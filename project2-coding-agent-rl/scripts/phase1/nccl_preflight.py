#!/usr/bin/env python3
"""Fail-fast NCCL/all-reduce preflight for the exact DeepSpeed GPU mapping.

Launch inside tmux with an explicit physical include list, for example:
  deepspeed --include localhost:2,4,6,7 nccl_preflight.py

Do not set CUDA_VISIBLE_DEVICES yourself and do not use ``--num_gpus``.
"""

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if not visible:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty; launch with DeepSpeed --include")

    # On the host, CUDA_VISIBLE_DEVICES contains physical indices. Inside an
    # NVIDIA-runtime-isolated container it contains logical 0..N-1 instead, so
    # require an explicit, auditable physical mapping. The container launcher
    # separately verifies that its visible UUID list matches this mapping.
    physical = [
        part.strip()
        for part in os.environ.get("PHYSICAL_GPU_IDS", ",".join(visible)).split(",")
        if part.strip()
    ]
    if len(physical) != len(visible):
        raise RuntimeError(
            f"physical/logical GPU mapping length mismatch: physical={physical}, visible={visible}"
        )
    if "0" in physical:
        raise RuntimeError(f"REFUSED: physical GPU 0 appears in PHYSICAL_GPU_IDS={physical}")

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank >= len(visible):
        raise RuntimeError(f"LOCAL_RANK={local_rank} cannot map into CUDA_VISIBLE_DEVICES={visible}")
    physical_gpu = physical[local_rank]
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    value = torch.tensor([float(dist.get_rank() + 1)], device="cuda")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all-reduce mismatch: got {value.item()}, expected {expected}")

    print(
        "NCCL_PREFLIGHT_PASS"
        f" host={socket.gethostname()} rank={dist.get_rank()} local_rank={local_rank}"
        f" physical_gpu={physical_gpu} visible={','.join(visible)} sum={value.item()}"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
