"""
Data processing module.
=======================
Handles data processing and validation operations.
"""

from typing import List, Dict, Any
from utils import validate_item, filter_by_type


class DataProcessor:
    """Processes and validates data items."""

    def __init__(self):
        self.processed_count = 0
        self.errors = []

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data item."""
        if not validate_item(item):
            error_msg = f"Invalid item: {item}"
            self.errors.append(error_msg)
            raise ValueError(error_msg)

        # Process the item (add some computed fields)
        processed_item = item.copy()
        processed_item["processed"] = True
        processed_item["double_value"] = item["value"] * 2

        self.processed_count += 1
        return processed_item

    def process_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of items."""
        processed_items = []

        for item in items:
            try:
                processed_item = self.process_item(item)
                processed_items.append(processed_item)
            except ValueError as e:
                print(f"Error processing item: {e}")
                continue

        return processed_items

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_count": self.processed_count,
            "error_count": len(self.errors),
            "errors": self.errors.copy(),
        }

    def reset(self) -> None:
        """Reset processor state."""
        self.processed_count = 0
        self.errors.clear()


class AdvancedProcessor(DataProcessor):
    """Advanced data processor with additional features."""

    def __init__(self):
        super().__init__()
        self.filters_applied = 0

    def filter_and_process(
        self, items: List[Dict[str, Any]], item_type: str
    ) -> List[Dict[str, Any]]:
        """Filter items by type and then process them."""
        filtered_items = filter_by_type(items, item_type)
        self.filters_applied += 1
        return self.process_batch(filtered_items)

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced processing statistics."""
        base_stats = super().get_stats()
        base_stats["filters_applied"] = self.filters_applied
        return base_stats


# Orphaned class - not used anywhere
class UnusedProcessor:
    """This processor class is not used anywhere."""

    def __init__(self):
        self.name = "unused"

    def do_nothing(self):
        """This method does nothing useful."""
        pass
