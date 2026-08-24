"""Official OpenSearch-VL tool contract backed by frozen local retrieval.

The model-facing schema and observations deliberately contain no local index IDs,
filesystem paths, similarity scores, or provider implementation details.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from .text_index import LocalTextIndex

ImageLookup = Callable[[str, int], Sequence[Mapping[str, object]]]

_IMAGE_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "image_search",
        "description": (
            "Visually identify the contents of an image. Returns summarized "
            "title/source pairs filtered by Qwen3-32B."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Image reference (e.g., 'img_1') or a direct image URL.",
                }
            },
            "required": ["url"],
        },
    },
}

_TEXT_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "text_search",
        "description": (
            "Search for text documents using Serper + Jina + Qwen summarization. "
            "Use for entity / fact lookups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "q": {"type": "string"},
                "query": {"type": "string", "description": "Alias for 'q'."},
                "hl": {"type": "string", "default": "en"},
                "lang": {
                    "type": "string",
                    "description": "Alias for 'hl'.",
                    "default": "en",
                },
                "top_k": {"type": "integer", "default": 5},
            },
            "required": [],
        },
    },
}


def official_search_tool_schemas() -> list[dict[str, Any]]:
    """Return an isolated copy of the official model-facing search schemas."""

    return deepcopy([_IMAGE_SEARCH_SCHEMA, _TEXT_SEARCH_SCHEMA])


def _clean_field(value: object, *, maximum: int = 4_000) -> str:
    text = " ".join(str(value).split())
    return text[:maximum]


def format_image_results(results: Sequence[Mapping[str, object]]) -> str:
    """Project local visual matches to official title/source-only observations."""

    if not results:
        return "Tool execution result:\nNo relevant image matches found."
    passages = []
    for index, result in enumerate(results, start=1):
        title = _clean_field(result.get("title", "Untitled"), maximum=500)
        source = _clean_field(result.get("source", ""), maximum=2_000)
        passages.append(
            f"[Visual Match {index}]\nTitle: {title}\nSource: {source}"
        )
    return "Tool execution result:\n" + "\n\n".join(passages)


def format_text_results(results: Sequence[Mapping[str, object]]) -> str:
    """Match the official text-search Passage/Title/URL/Summary layout."""

    if not results:
        return "Tool execution result:\nNo relevant web pages found for the query."
    passages = []
    for index, result in enumerate(results, start=1):
        title = _clean_field(result.get("title", "Untitled"), maximum=500)
        source = _clean_field(result.get("source", ""), maximum=2_000)
        summary = _clean_field(result.get("summary", ""))
        passages.append(
            f"[Passage {index}]\nTitle: {title}\nURL: {source}\nSummary:\n{summary}"
        )
    separator = "\n\n" + "=" * 60 + "\n\n"
    return "Tool execution result:\n" + separator.join(passages)


class OfficialLocalSearchProvider:
    """Execute official calls against read-only local image and text providers."""

    def __init__(self, text_index: LocalTextIndex, image_lookup: ImageLookup):
        self.text_index = text_index
        self.image_lookup = image_lookup

    @staticmethod
    def _image_reference(arguments: Mapping[str, object]) -> str:
        if set(arguments) != {"url"}:
            raise ValueError("image_search accepts exactly the official 'url' argument")
        reference = str(arguments.get("url", "")).strip()
        if not reference:
            raise ValueError("image_search requires a non-empty 'url'")
        if not reference.startswith("img_") or not reference[4:].isdigit():
            raise ValueError("offline image_search accepts only registered img_N references")
        return reference

    @staticmethod
    def _text_arguments(arguments: Mapping[str, object]) -> tuple[str, str, int]:
        allowed = {"q", "query", "hl", "lang", "top_k"}
        if set(arguments) - allowed:
            raise ValueError("text_search received a non-official argument")
        query = str(arguments.get("q") or arguments.get("query") or "").strip()
        if not query:
            raise ValueError("text_search requires 'q' or 'query'")
        if len(query) > 1_000:
            raise ValueError("text_search query exceeds 1,000 characters")
        language = str(arguments.get("hl") or arguments.get("lang") or "en").strip()
        if not language or len(language) > 32:
            raise ValueError("text_search language is invalid")
        top_k = int(arguments.get("top_k", 5))
        if not 1 <= top_k <= 20:
            raise ValueError("text_search top_k must be between 1 and 20")
        return query, language, top_k

    def call(self, name: str, arguments: Mapping[str, object]) -> str:
        if name == "image_search":
            reference = self._image_reference(arguments)
            return format_image_results(self.image_lookup(reference, 5))
        if name == "text_search":
            query, _language, top_k = self._text_arguments(arguments)
            return format_text_results(self.text_index.search(query, top_k=top_k))
        raise ValueError(f"tool is not registered in the local search provider: {name}")

    def safe_call(self, name: str, arguments: Mapping[str, object]) -> str:
        """Model-facing execution that never exposes backend exception details."""

        try:
            return self.call(name, arguments)
        except Exception:
            if name == "image_search":
                return "Tool execution error:\nimage_search failed in the current environment."
            if name == "text_search":
                return "Tool execution error:\ntext_search failed in the current environment."
            return "Tool execution error:\nThe requested tool is not available."
