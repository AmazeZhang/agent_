import unittest

from pagination import paginate


class PaginationTests(unittest.TestCase):
    def test_first_page(self) -> None:
        self.assertEqual(paginate([1, 2, 3, 4, 5], 1, 2), [1, 2])

    def test_later_and_partial_pages(self) -> None:
        self.assertEqual(paginate([1, 2, 3, 4, 5], 2, 2), [3, 4])
        self.assertEqual(paginate([1, 2, 3, 4, 5], 3, 2), [5])

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            paginate([1], 0, 1)
        with self.assertRaises(ValueError):
            paginate([1], 1, 0)


if __name__ == "__main__":
    unittest.main()
