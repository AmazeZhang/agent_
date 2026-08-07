import unittest

from text_utils import slugify


class SlugifyTests(unittest.TestCase):
    def test_spaces_and_punctuation(self) -> None:
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_repeated_separators(self) -> None:
        self.assertEqual(slugify("  API---Design  "), "api-design")

    def test_underscore(self) -> None:
        self.assertEqual(slugify("Clean_value"), "clean-value")


if __name__ == "__main__":
    unittest.main()
