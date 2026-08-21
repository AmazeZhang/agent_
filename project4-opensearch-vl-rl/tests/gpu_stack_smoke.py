"""Minimal CUDA/FlashAttention smoke for the pinned SFT environment."""

import json

import torch
from flash_attn import flash_attn_func


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible logical GPU, got {torch.cuda.device_count()}")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    q = torch.randn(1, 128, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 128, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 128, 8, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    output.float().square().mean().backward()
    torch.cuda.synchronize()

    result = {
        "logical_device_count": torch.cuda.device_count(),
        "logical_device": str(device),
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flash_output_shape": list(output.shape),
        "flash_output_finite": bool(torch.isfinite(output).all().item()),
        "flash_grad_finite": bool(torch.isfinite(q.grad).all().item()),
    }
    print(json.dumps(result, sort_keys=True))

    if not result["flash_output_finite"] or not result["flash_grad_finite"]:
        raise RuntimeError("FlashAttention produced non-finite output or gradients")


if __name__ == "__main__":
    main()
