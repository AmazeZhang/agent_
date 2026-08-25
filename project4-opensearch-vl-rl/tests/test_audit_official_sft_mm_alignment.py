"""Manifest compatibility tests for official SFT alignment audits."""

from __future__ import annotations

import unittest

from scripts.audit_official_sft_mm_alignment import source_indices_for_manifest


class OfficialSftAlignmentManifestTests(unittest.TestCase):
    def test_accepts_parent_and_derived_manifest_keys(self) -> None:
        self.assertEqual(source_indices_for_manifest({"selected_indices": [1, 2]}, 2), [1, 2])
        self.assertEqual(
            source_indices_for_manifest({"selected_source_indices": [3, 4]}, 2),
            [3, 4],
        )

    def test_rejects_wrong_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "no longer align"):
            source_indices_for_manifest({"selected_source_indices": [3]}, 2)


if __name__ == "__main__":
    unittest.main()
