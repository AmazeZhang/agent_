"""Safe local execution for the official image-handle tool contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from PIL import Image

from .official_provider import format_image_results

VisualLookup = Callable[[Image.Image, int], Sequence[Mapping[str, object]]]


@dataclass(frozen=True)
class MultimodalToolResult:
    observation: str
    image_handle: str | None = None
    image: Image.Image | None = None


class LocalImageRegistry:
    """Per-trajectory in-memory registry; never exposes filesystem paths."""

    def __init__(self, initial: Image.Image):
        self._images: dict[str, Image.Image] = {"img_1": initial.convert("RGB").copy()}

    def get(self, handle: str) -> Image.Image:
        if handle not in self._images:
            raise ValueError("image reference is not registered in this trajectory")
        return self._images[handle].copy()

    def add(self, image: Image.Image) -> str:
        handle = f"img_{len(self._images) + 1}"
        self._images[handle] = image.convert("RGB").copy()
        return handle

    @property
    def handles(self) -> tuple[str, ...]:
        return tuple(self._images)


class OfficialLocalMultimodalProvider:
    """Execute crop and live image search behind the official model-facing API."""

    def __init__(self, registry: LocalImageRegistry, visual_lookup: VisualLookup):
        self.registry = registry
        self.visual_lookup = visual_lookup

    @staticmethod
    def _exact_integer(arguments: Mapping[str, object], name: str) -> int:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"crop {name} must be an integer")
        return value

    def crop(self, arguments: Mapping[str, object]) -> MultimodalToolResult:
        expected = {"image", "x", "y", "width", "height"}
        if set(arguments) != expected:
            raise ValueError("crop requires exactly image, x, y, width, and height")
        handle = str(arguments.get("image", ""))
        source = self.registry.get(handle)
        x = self._exact_integer(arguments, "x")
        y = self._exact_integer(arguments, "y")
        width = self._exact_integer(arguments, "width")
        height = self._exact_integer(arguments, "height")
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("crop coordinates must be non-negative with positive size")
        if x + width > source.width or y + height > source.height:
            raise ValueError("crop bounding box exceeds the source image")
        cropped = source.crop((x, y, x + width, y + height)).convert("RGB")
        new_handle = self.registry.add(cropped)
        return MultimodalToolResult(
            observation=f"Image cropped successfully. New image ID: {new_handle}.",
            image_handle=new_handle,
            image=cropped,
        )

    def image_search(self, arguments: Mapping[str, object]) -> MultimodalToolResult:
        if set(arguments) != {"url"}:
            raise ValueError("image_search accepts exactly the official 'url' argument")
        handle = str(arguments.get("url", "")).strip()
        image = self.registry.get(handle)
        results = self.visual_lookup(image, 5)
        return MultimodalToolResult(observation=format_image_results(results))

    def call(self, name: str, arguments: Mapping[str, object]) -> MultimodalToolResult:
        if name == "crop":
            return self.crop(arguments)
        if name == "image_search":
            return self.image_search(arguments)
        raise ValueError(f"multimodal tool is not implemented locally: {name}")

    def safe_call(self, name: str, arguments: Mapping[str, object]) -> MultimodalToolResult:
        try:
            return self.call(name, arguments)
        except Exception:
            return MultimodalToolResult(
                observation=f"Tool execution error:\n{name} failed in the current environment."
            )
