"""Safe local execution for the official image-handle tool contract."""

from __future__ import annotations

import io
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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
    """Execute bounded local tools behind the official model-facing API."""

    _OCR_TIMEOUT_SECONDS = 15
    _OCR_MAX_CHARACTERS = 8_000

    def __init__(self, registry: LocalImageRegistry, visual_lookup: VisualLookup):
        self.registry = registry
        self.visual_lookup = visual_lookup

    @staticmethod
    def _exact_integer(arguments: Mapping[str, object], name: str) -> int:
        value = arguments.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"crop {name} must be an integer")
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
        results = self.visual_lookup(image, top_k=5)
        return MultimodalToolResult(observation=format_image_results(results))

    def layout_parsing(self, arguments: Mapping[str, object]) -> MultimodalToolResult:
        allowed = {
            "image",
            "file_path",
            "use_chart_recognition",
            "use_doc_orientation_classify",
        }
        if not set(arguments).issubset(allowed):
            raise ValueError("layout_parsing received an unsupported argument")
        if "file_path" in arguments:
            raise ValueError("local layout_parsing does not accept filesystem paths")
        if set(arguments).difference(
            {"image", "use_chart_recognition", "use_doc_orientation_classify"}
        ):
            raise ValueError("layout_parsing requires an in-trajectory image reference")
        handle = arguments.get("image")
        if not isinstance(handle, str) or not handle.strip():
            raise ValueError("layout_parsing requires an image reference")
        for flag in ("use_chart_recognition", "use_doc_orientation_classify"):
            value = arguments.get(flag, False)
            if not isinstance(value, bool):
                raise TypeError(f"layout_parsing {flag} must be a boolean")
            if value:
                raise ValueError(f"local layout_parsing does not support {flag}=true")

        image = self.registry.get(handle.strip())
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        completed = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", "6"],
            input=encoded.getvalue(),
            capture_output=True,
            check=False,
            timeout=self._OCR_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError("local OCR process failed")
        text = completed.stdout.decode("utf-8", errors="replace").strip()
        if len(text) > self._OCR_MAX_CHARACTERS:
            text = text[: self._OCR_MAX_CHARACTERS].rstrip() + "\n[truncated]"
        if not text:
            text = "[no text detected]"
        return MultimodalToolResult(observation=f"Layout parsing result:\n{text}")

    def call(self, name: str, arguments: Mapping[str, object]) -> MultimodalToolResult:
        if name == "crop":
            return self.crop(arguments)
        if name == "image_search":
            return self.image_search(arguments)
        if name == "layout_parsing":
            return self.layout_parsing(arguments)
        raise ValueError(f"multimodal tool is not implemented locally: {name}")

    def safe_call(self, name: str, arguments: Mapping[str, object]) -> MultimodalToolResult:
        try:
            return self.call(name, arguments)
        except Exception:  # noqa: BLE001 - model-facing errors must redact backend details.
            return MultimodalToolResult(
                observation=f"Tool execution error:\n{name} failed in the current environment."
            )
