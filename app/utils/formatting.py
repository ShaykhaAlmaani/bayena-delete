"""Formatting helpers for UI display."""

from typing import Any


def format_large_number(value: Any) -> str:
    """Format a number with thousands separators."""
    try:
        n = float(str(value).replace(",", ""))
        if n == int(n):
            return f"{int(n):,}"
        return f"{n:,.2f}"
    except (ValueError, TypeError):
        return str(value)


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text with ellipsis."""
    return text if len(text) <= max_length else text[:max_length - 3] + "..."


def arabic_number_label(value: str) -> str:
    """Convert ASCII digits to Arabic-Indic numerals for RTL UI."""
    arabic_digits = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
    return value.translate(arabic_digits)
