"""
Main application module.
========================
Entry point for the mini application with basic functionality.
"""

from utils import format_message, calculate_total
from data_processor import DataProcessor


class Application:
    """Main application class."""

    def __init__(self, name: str):
        self.name = name
        self.processor = DataProcessor()
        self.data = []

    def add_data(self, item: dict) -> None:
        """Add data item to the application."""
        self.data.append(item)
        print(format_message(f"Added item: {item}"))

    def process_data(self) -> list:
        """Process all data items."""
        return self.processor.process_batch(self.data)

    def get_summary(self) -> dict:
        """Get application summary."""
        processed = self.process_data()
        total = calculate_total([item.get("value", 0) for item in processed])

        return {
            "name": self.name,
            "total_items": len(self.data),
            "processed_items": len(processed),
            "total_value": total,
        }


def main():
    """Main entry point."""
    app = Application("Mini App")

    # Add some sample data
    app.add_data({"id": 1, "value": 10, "type": "A"})
    app.add_data({"id": 2, "value": 20, "type": "B"})
    app.add_data({"id": 3, "value": 15, "type": "A"})

    # Get and display summary
    summary = app.get_summary()
    print(f"Application Summary: {summary}")


if __name__ == "__main__":
    main()
