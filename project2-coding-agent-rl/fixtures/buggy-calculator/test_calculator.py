import unittest

from calculator import divide


class DivideTests(unittest.TestCase):
    def test_positive_numbers(self) -> None:
        self.assertEqual(divide(10, 2), 5)

    def test_negative_number(self) -> None:
        self.assertEqual(divide(-9, 3), -3)

    def test_zero_divisor(self) -> None:
        with self.assertRaises(ZeroDivisionError):
            divide(1, 0)


if __name__ == "__main__":
    unittest.main()

