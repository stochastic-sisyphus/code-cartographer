"""
Data processing module for the mini example package.
"""

from typing import List, Dict, Any


class DataProcessor:
    """Process data in various ways."""
    
    def __init__(self, name: str):
        """Initialize the data processor."""
        self.name = name
        self.data = []
    
    def add_data(self, item: Any) -> None:
        """Add an item to the data collection."""
        self.data.append(item)
    
    def get_count(self) -> int:
        """Get the count of items."""
        return len(self.data)
    
    def clear(self) -> None:
        """Clear all data."""
        self.data = []


def process_list(items: List[int]) -> Dict[str, Any]:
    """Process a list of numbers and return statistics."""
    if not items:
        return {"count": 0, "sum": 0, "average": 0}
    
    total = sum(items)
    count = len(items)
    average = total / count
    
    return {
        "count": count,
        "sum": total,
        "average": average,
        "min": min(items),
        "max": max(items)
    }
