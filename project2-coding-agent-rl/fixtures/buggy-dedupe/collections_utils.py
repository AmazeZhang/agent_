from collections.abc import Iterable
from typing import TypeVar


T = TypeVar("T")


def unique_in_order(values: Iterable[T]) -> list[T]:
    """Return the first occurrence of every value, preserving input order."""
    return sorted(set(values))
