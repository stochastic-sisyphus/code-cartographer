# Mini Repository Example

This is a small example Python package used to demonstrate the code-cartographer tool.

## Structure

- `__init__.py` - Package initialization
- `utils.py` - Utility functions for basic calculations
- `processor.py` - Data processing classes and functions
- `main.py` - Main application entry point

## Running the Analysis

To analyze this mini repository:

```bash
codecart analyze -d examples/mini_repo -o examples/output/analysis.json
```

This will generate:
- JSON analysis data at `examples/output/analysis.json`
- Markdown report at `examples/output/code_analysis_report.md`
- HTML report at `examples/output/code_analysis_report.html`
