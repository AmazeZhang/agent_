"""CPU-only tests for official crop/image-handle execution."""

from __future__ import annotations

import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from PIL import Image

from local_retrieval import (
    LocalImageRegistry,
    OfficialLocalMultimodalProvider,
    official_system_prompt,
    official_tool_schemas,
)


class OfficialMultimodalProviderTests(unittest.TestCase):
    def provider(self):
        registry = LocalImageRegistry(Image.new("RGB", (100, 80), "white"))
        seen = []

        def lookup(image, top_k):
            seen.append((image.size, top_k))
            return [
                {
                    "entity_id": "hidden:1",
                    "title": "Cropped subject",
                    "source": "https://en.wikipedia.org/wiki/Cropped_subject",
                    "similarity": 0.9,
                }
            ]

        return registry, seen, OfficialLocalMultimodalProvider(registry, lookup)

    def test_complete_schema_matches_official_order_and_arguments(self) -> None:
        schemas = official_tool_schemas()
        names = [item["function"]["name"] for item in schemas]
        self.assertEqual(
            names,
            [
                "crop",
                "layout_parsing",
                "web_search",
                "image_search",
                "text_search",
                "perspective_correct",
                "super_resolution",
                "sharpen",
            ],
        )
        crop = schemas[0]["function"]["parameters"]
        self.assertEqual(crop["required"], ["image", "x", "y", "width", "height"])
        self.assertIn('CORE PHILOSOPHY: "Verify, Don\'t Guess"', official_system_prompt())

    def test_crop_registers_real_image_and_live_search_uses_it(self) -> None:
        registry, seen, provider = self.provider()
        result = provider.call(
            "crop",
            {"image": "img_1", "x": 10, "y": 15, "width": 30, "height": 20},
        )
        self.assertEqual(result.image_handle, "img_2")
        self.assertEqual(result.image.size, (30, 20))
        self.assertEqual(registry.handles, ("img_1", "img_2"))
        search = provider.call("image_search", {"url": "img_2"})
        self.assertEqual(seen, [((30, 20), 5)])
        self.assertIn("Title: Cropped subject", search.observation)
        self.assertNotIn("hidden:1", search.observation)
        self.assertNotIn("similarity", search.observation)

    def test_crop_rejects_padding_unknown_handles_and_extra_arguments(self) -> None:
        _registry, _seen, provider = self.provider()
        invalid = [
            {"image": "img_9", "x": 0, "y": 0, "width": 10, "height": 10},
            {"image": "img_1", "x": 90, "y": 0, "width": 20, "height": 10},
            {"image": "img_1", "x": 0, "y": 0, "width": 10, "height": 10, "path": "/tmp/x"},
            {"image": "img_1", "x": True, "y": 0, "width": 10, "height": 10},
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(
                (TypeError, ValueError)
            ):
                provider.call("crop", arguments)

    def test_safe_call_hides_backend_details(self) -> None:
        registry = LocalImageRegistry(Image.new("RGB", (10, 10), "white"))
        provider = OfficialLocalMultimodalProvider(
            registry, lambda _image, _top_k: (_ for _ in ()).throw(RuntimeError("/secret"))
        )
        result = provider.safe_call("image_search", {"url": "img_1"})
        self.assertEqual(
            result.observation,
            "Tool execution error:\nimage_search failed in the current environment.",
        )
        self.assertNotIn("secret", result.observation)

    def test_layout_parsing_uses_registered_image_without_path_exposure(self) -> None:
        _registry, _seen, provider = self.provider()
        completed = CompletedProcess(
            args=["tesseract"],
            returncode=0,
            stdout=b"THE STATION, WORCESTER, MASS.\n",
            stderr=b"",
        )
        with patch(
            "local_retrieval.official_multimodal_provider.subprocess.run",
            return_value=completed,
        ) as run:
            result = provider.call(
                "layout_parsing",
                {
                    "image": "img_1",
                    "use_chart_recognition": False,
                    "use_doc_orientation_classify": False,
                },
            )
        self.assertEqual(
            result.observation,
            "Layout parsing result:\nTHE STATION, WORCESTER, MASS.",
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertEqual(run.call_args.args[0][:3], ["tesseract", "stdin", "stdout"])
        self.assertTrue(run.call_args.kwargs["input"].startswith(b"\x89PNG"))

    def test_layout_parsing_rejects_paths_and_unsupported_modes(self) -> None:
        _registry, _seen, provider = self.provider()
        invalid = [
            {"file_path": "/secret/image.png"},
            {"image": "img_1", "use_chart_recognition": True},
            {"image": "img_1", "use_doc_orientation_classify": True},
            {"image": "img_1", "unknown": False},
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                provider.call("layout_parsing", arguments)


if __name__ == "__main__":
    unittest.main()
