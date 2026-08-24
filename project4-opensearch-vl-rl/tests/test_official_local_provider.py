"""CPU-only contract and information-hiding tests for the local provider."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_retrieval import (
    LocalTextIndex,
    OfficialLocalSearchProvider,
    build_text_index,
    official_search_tool_schemas,
)


class OfficialLocalProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.index_path = Path(self.temporary.name) / "wiki.sqlite"
        build_text_index(
            self.index_path,
            [
                {
                    "entity_id": "internal:alpha",
                    "title": "Alpha Bridge",
                    "source": "https://en.wikipedia.org/wiki/Alpha_Bridge",
                    "text": "Alpha Bridge opened in 1952 and crosses the Example River.",
                }
            ],
            corpus="private-local-corpus",
            corpus_revision="private-revision",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def provider(self) -> tuple[LocalTextIndex, OfficialLocalSearchProvider]:
        index = LocalTextIndex(self.index_path)
        provider = OfficialLocalSearchProvider(
            index,
            lambda reference, top_k: [
                {
                    "entity_id": "internal:alpha",
                    "title": "Alpha Bridge",
                    "source": "https://en.wikipedia.org/wiki/Alpha_Bridge",
                    "similarity": 0.99,
                    "filesystem_path": f"/private/{reference}.png",
                }
            ][:top_k],
        )
        return index, provider

    def test_schemas_use_official_names_and_arguments(self) -> None:
        schemas = official_search_tool_schemas()
        by_name = {item["function"]["name"]: item for item in schemas}
        self.assertEqual(set(by_name), {"image_search", "text_search"})
        self.assertEqual(
            by_name["image_search"]["function"]["parameters"]["required"], ["url"]
        )
        self.assertEqual(
            set(by_name["text_search"]["function"]["parameters"]["properties"]),
            {"q", "query", "hl", "lang", "top_k"},
        )

    def test_image_provider_hides_local_implementation(self) -> None:
        index, provider = self.provider()
        try:
            result = provider.call("image_search", {"url": "img_1"})
        finally:
            index.close()
        self.assertIn("Title: Alpha Bridge", result)
        self.assertIn("Source: https://en.wikipedia.org/wiki/Alpha_Bridge", result)
        for secret in ("entity_id", "similarity", "filesystem_path", "/private/"):
            self.assertNotIn(secret, result)

    def test_text_provider_matches_official_passage_layout(self) -> None:
        index, provider = self.provider()
        try:
            result = provider.call(
                "text_search", {"query": "Alpha Bridge", "lang": "en", "top_k": 1}
            )
        finally:
            index.close()
        self.assertIn("[Passage 1]", result)
        self.assertIn("Title: Alpha Bridge", result)
        self.assertIn("URL: https://en.wikipedia.org/wiki/Alpha_Bridge", result)
        self.assertIn("Summary:", result)
        for secret in ("internal:alpha", "private-local-corpus", "private-revision"):
            self.assertNotIn(secret, result)

    def test_offline_provider_rejects_external_urls_and_internal_arguments(self) -> None:
        index, provider = self.provider()
        try:
            with self.assertRaises(ValueError):
                provider.call("image_search", {"url": "https://example.com/image.png"})
            with self.assertRaises(ValueError):
                provider.call("image_search", {"image": "img_1"})
            with self.assertRaises(ValueError):
                provider.call("text_search", {"entity_id": "internal:alpha"})
        finally:
            index.close()

    def test_schemas_are_returned_by_copy(self) -> None:
        first = official_search_tool_schemas()
        first[0]["function"]["name"] = "mutated"
        self.assertEqual(
            official_search_tool_schemas()[0]["function"]["name"], "image_search"
        )

    def test_safe_call_hides_backend_exceptions(self) -> None:
        index = LocalTextIndex(self.index_path)
        provider = OfficialLocalSearchProvider(
            index,
            lambda _reference, _top_k: (_ for _ in ()).throw(
                RuntimeError("/private/index/path secret-token")
            ),
        )
        try:
            result = provider.safe_call("image_search", {"url": "img_1"})
        finally:
            index.close()
        self.assertEqual(
            result,
            "Tool execution error:\nimage_search failed in the current environment.",
        )
        self.assertNotIn("private", result)
        self.assertNotIn("secret-token", result)


if __name__ == "__main__":
    unittest.main()
