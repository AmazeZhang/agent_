import os

physical_gpu = os.environ.get("PHYSICAL_GPU_ID")
visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")

if physical_gpu in {None, "", "0"}:
    raise RuntimeError(f"Invalid physical GPU selection: {physical_gpu!r}")
if visible_gpu != physical_gpu:
    raise RuntimeError(
        f"CUDA_VISIBLE_DEVICES={visible_gpu!r} does not match PHYSICAL_GPU_ID={physical_gpu!r}"
    )

import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")

device = torch.device("cuda:0")
x = torch.randn((1024, 1024), device=device)
y = x @ x
checksum = y.float().mean().item()
torch.cuda.synchronize(device)

print(f"physical_gpu={physical_gpu}")
print(f"visible_device={torch.cuda.get_device_name(device)}")
print(f"torch={torch.__version__}")
print(f"checksum={checksum:.6f}")
print("CUDA_SMOKE_OK")

