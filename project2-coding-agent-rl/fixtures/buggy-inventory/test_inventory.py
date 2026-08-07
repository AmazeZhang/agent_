import unittest

from inventory import total_value


class InventoryTests(unittest.TestCase):
    def test_multiple_items(self) -> None:
        items = [
            {"quantity": 3, "unit_price": 4.5},
            {"quantity": 2, "unit_price": 10.0},
        ]
        self.assertEqual(total_value(items), 33.5)

    def test_fractional_quantity(self) -> None:
        self.assertEqual(total_value([{"quantity": 1.5, "unit_price": 8.0}]), 12.0)

    def test_empty_inventory(self) -> None:
        self.assertEqual(total_value([]), 0)


if __name__ == "__main__":
    unittest.main()
