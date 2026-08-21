"""Fixed torchvision ResNet-50 V1 encoder used by local visual search."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_resnet50_v1(
    weights_path: Path,
    *,
    device: str,
    sha256_prefix: str = "0676ba61",
) -> tuple[Any, Any, dict[str, str]]:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights_sha256 = sha256_file(weights_path)
    if not weights_sha256.startswith(sha256_prefix.lower()):
        raise ValueError(
            f"weight SHA256 prefix mismatch: {weights_sha256}/{sha256_prefix}"
        )
    specification = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=None)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = specification.transforms()
    return model, preprocess, {
        "implementation": "torchvision.models.resnet50",
        "weights": "ResNet50_Weights.IMAGENET1K_V1",
        "weights_url": specification.url,
        "weights_sha256": weights_sha256,
        "preprocess": str(preprocess),
    }


def encode_pil_images(
    model: Any,
    preprocess: Any,
    images: Sequence[Image.Image],
    *,
    device: str,
) -> Any:
    import torch

    if not images:
        raise ValueError("cannot encode an empty image batch")
    batch = torch.stack([preprocess(image.convert("RGB")) for image in images]).to(
        device, non_blocking=True
    )
    with torch.inference_mode():
        return model(batch).detach().cpu().numpy()
