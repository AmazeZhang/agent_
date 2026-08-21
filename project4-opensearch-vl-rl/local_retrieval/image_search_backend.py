"""Local image-search backend with a fixed query/candidate encoder contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .resnet50_encoder import encode_pil_images, load_resnet50_v1
from .visual_index import ExactVisualIndex, tool_observation


def validate_encoder_revision(index_manifest: dict[str, Any], weights_sha256: str) -> None:
    revision = str(index_manifest.get("corpus_revision", ""))
    if not revision.endswith(f"+{weights_sha256}"):
        raise ValueError(
            "visual index/query encoder mismatch: "
            f"index_revision={revision!r} weights_sha256={weights_sha256}"
        )


def resolve_local_image(path: Path, allowed_root: Path) -> Path:
    root = allowed_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    requested = path if path.is_absolute() else allowed_root / path
    candidate = requested.resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"image path escaped allowed root or is not a file: {path}")
    relative = requested.absolute().relative_to(allowed_root.absolute())
    current = allowed_root.absolute()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"image path traverses a symbolic link: {current}")
    return candidate


class LocalImageSearchBackend:
    """Encode local images and query an immutable exact-cosine pilot index."""

    def __init__(
        self,
        index_root: Path,
        weights_path: Path,
        *,
        device: str = "cpu",
        weights_sha256_prefix: str = "0676ba61",
    ):
        self.index = ExactVisualIndex(index_root)
        self.device = device
        self.model, self.preprocess, self.encoder = load_resnet50_v1(
            weights_path,
            device=device,
            sha256_prefix=weights_sha256_prefix,
        )
        validate_encoder_revision(
            self.index.manifest, self.encoder["weights_sha256"]
        )

    def search_image(
        self,
        image: Image.Image,
        *,
        top_k: int = 5,
        minimum_similarity: float = -1.0,
    ) -> list[dict[str, object]]:
        vector = encode_pil_images(
            self.model,
            self.preprocess,
            [image],
            device=self.device,
        )[0]
        return self.index.search(
            vector,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
        )

    def search_path(
        self,
        path: Path,
        *,
        allowed_root: Path,
        top_k: int = 5,
        minimum_similarity: float = -1.0,
    ) -> list[dict[str, object]]:
        resolved = resolve_local_image(path, allowed_root)
        with Image.open(resolved) as image:
            image.load()
            return self.search_image(
                image.convert("RGB"),
                top_k=top_k,
                minimum_similarity=minimum_similarity,
            )

    def tool_call(self, *args: Any, **kwargs: Any) -> str:
        return tool_observation(self.search_path(*args, **kwargs))
