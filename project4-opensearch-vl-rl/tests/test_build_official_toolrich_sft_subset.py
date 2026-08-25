"""Selection tests for the official-only tool-rich SFT subset."""

from __future__ import annotations

import unittest

from scripts.build_official_toolrich_sft_subset import select_rows


def row(tool: str, *, error: bool = False):
    conversations = [
        {"from": "human", "value": "<image>question"},
        {
            "from": "gpt",
            "value": f'<tool_call>{{"name":"{tool}","arguments":{{}}}}</tool_call>',
        },
        {
            "from": "observation",
            "value": "Tool execution error:\nfailed" if error else "<observation>ok</observation>",
        },
    ]
    return {"conversations": conversations, "images": ["images/x.png"]}


class OfficialToolrichSelectionTests(unittest.TestCase):
    def test_selects_visual_tools_and_real_errors_without_duplicates(self) -> None:
        rows = [
            row("text_search"),
            row("crop"),
            row("layout_parsing"),
            row("image_search", error=True),
            row("sharpen", error=True),
        ]
        selected = select_rows(rows)
        self.assertEqual([item[0] for item in selected], [1, 2, 3, 4])
        self.assertEqual(len({item[0] for item in selected}), len(selected))
        self.assertEqual(sum(item[3] for item in selected), 2)


if __name__ == "__main__":
    unittest.main()
