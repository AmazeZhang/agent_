from typing import Iterable, Mapping


def total_value(items: Iterable[Mapping[str, float]]) -> float:
    """Return the combined value of all inventory entries."""
    return sum(item["quantity"] + item["unit_price"] for item in items)
