# Example Analysis Output

This directory contains example output from running code-cartographer on the mini_repo.

## Files

- `analysis.json` - Raw JSON analysis data
- `code_analysis_report.md` - Markdown report
- `code_analysis_report.html` - HTML report (can be opened directly in a browser)

## How to Generate

To regenerate these files, run:

```bash
codecart analyze -d examples/mini_repo -o examples/output/analysis.json
```

or from the repository root:

```bash
python -m code_cartographer.cli analyze -d examples/mini_repo -o examples/output/analysis.json
```
