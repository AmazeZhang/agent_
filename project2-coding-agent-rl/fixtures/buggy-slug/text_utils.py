import re


def slugify(value: str) -> str:
    """Convert a title to a lowercase URL slug."""
    value = value.strip().lower()
    return re.sub(r"\s+", "-", value)
