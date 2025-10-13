"""
Utility functions module.
=========================
Common utility functions used across the application.
"""

import datetime
from typing import List, Union


def format_message(message: str, timestamp: bool = True) -> str:
    """Format a message with optional timestamp."""
    if timestamp:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{now}] {message}"
    return message


def calculate_total(values: List[Union[int, float]]) -> float:
    """Calculate the total of a list of numeric values."""
    return sum(values)


def calculate_average(values: List[Union[int, float]]) -> float:
    """Calculate the average of a list of numeric values."""
    return 0.0 if not values else calculate_total(values) / len(values)


def filter_by_type(items: List[dict], item_type: str) -> List[dict]:
    """Filter items by their type field."""
    return [item for item in items if item.get("type") == item_type]


def validate_item(item: dict) -> bool:
    """Validate that an item has required fields."""
    required_fields = ["id", "value", "type"]
    return all(field in item for field in required_fields)


