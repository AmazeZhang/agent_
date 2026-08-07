import unittest

from collections_utils import unique_in_order


class UniqueInOrderTests(unittest.TestCase):
    def test_integer_order(self) -> None:
        self.assertEqual(unique_in_order([3, 1, 3, 2, 1]), [3, 1, 2])

    def test_string_order(self) -> None:
        self.assertEqual(unique_in_order(["b", "a", "b", "c"]), ["b", "a", "c"])

    def test_generator(self) -> None:
        self.assertEqual(unique_in_order(x for x in [2, 2, 1]), [2, 1])


if __name__ == "__main__":
    unittest.main()
