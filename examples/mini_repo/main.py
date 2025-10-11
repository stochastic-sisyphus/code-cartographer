"""
Main application module.
"""

from mini_repo.utils import calculate_sum, format_output
from mini_repo.processor import DataProcessor, process_list


def run_example():
    """Run a simple example."""
    # Basic calculations
    result = calculate_sum(10, 20)
    print(format_output(result))
    
    # Data processing
    processor = DataProcessor("example")
    processor.add_data(1)
    processor.add_data(2)
    processor.add_data(3)
    
    print(f"Processor has {processor.get_count()} items")
    
    # List processing
    numbers = [1, 2, 3, 4, 5]
    stats = process_list(numbers)
    print(f"Statistics: {stats}")


if __name__ == "__main__":
    run_example()
