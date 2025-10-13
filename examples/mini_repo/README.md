# Mini Repository Example

This is a minimal Python repository designed for testing and demonstrating the Code Cartographer tool.

## Structure

- `main.py` - Main application entry point with an Application class
- `utils.py` - Utility functions for formatting, calculations, and validation
- `data_processor.py` - Data processing classes with inheritance
- `config.py` - Configuration constants and settings

## Features Demonstrated

### Code Patterns
- Class inheritance (`AdvancedProcessor` extends `DataProcessor`)
- Function calls across modules
- Import dependencies
- Type hints and docstrings

### Analysis Targets
- **Orphaned Code**: Contains intentionally unused functions and classes
  - `unused_helper_function()` in `utils.py`
  - `UnusedProcessor` class in `data_processor.py`
  - `UNUSED_SETTING` variable in `config.py`

- **Dependencies**: Clear import relationships
  - `main.py` imports from `utils.py` and `data_processor.py`
  - `data_processor.py` imports from `utils.py`

- **Call Graph**: Function calls that can be traced
  - `main()` → `Application` methods
  - `Application.process_data()` → `DataProcessor.process_batch()`
  - Various utility function calls

## Usage

Run the mini application:
```bash
cd examples/mini_repo
python main.py
```

Analyze with Code Cartographer:
```bash
python -m code_cartographer.cli analyze --dir examples/mini_repo --output analysis.json
```

## Baseline Analysis

The file `baseline_analysis.json` contains the expected analysis output for this repository. This serves as a regression test baseline - any changes to the analysis engine should produce identical results when run on this unchanged codebase.

To verify the analysis is working correctly:
1. Run the analysis command above
2. Compare the output with `baseline_analysis.json`
3. The results should be identical (allowing for timestamp differences)