"""Offline retrieval components for the OpenSearch-VL reproduction."""

from .image_search_backend import LocalImageSearchBackend
from .official_provider import (
    OfficialLocalSearchProvider,
    format_image_results,
    format_text_results,
    official_search_tool_schemas,
    official_system_prompt,
    official_tool_schemas,
)
from .official_multimodal_provider import (
    LocalImageRegistry,
    MultimodalToolResult,
    OfficialLocalMultimodalProvider,
)
from .resnet50_encoder import encode_pil_images, load_resnet50_v1
from .text_index import LocalTextIndex, build_text_index, text_tool_observation
from .visual_index import (
    ExactVisualIndex,
    build_exact_index,
    entity_tool_observation,
    tool_observation,
)

__all__ = [
    "ExactVisualIndex",
    "LocalImageSearchBackend",
    "LocalImageRegistry",
    "LocalTextIndex",
    "OfficialLocalSearchProvider",
    "OfficialLocalMultimodalProvider",
    "MultimodalToolResult",
    "build_exact_index",
    "build_text_index",
    "encode_pil_images",
    "entity_tool_observation",
    "format_image_results",
    "format_text_results",
    "load_resnet50_v1",
    "official_search_tool_schemas",
    "official_system_prompt",
    "official_tool_schemas",
    "text_tool_observation",
    "tool_observation",
]
