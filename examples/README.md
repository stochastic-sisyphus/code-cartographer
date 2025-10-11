# Code Cartographer Examples

This directory contains example repositories and their analysis outputs to demonstrate the capabilities of code-cartographer.

## Directory Structure

```
examples/
├── mini_repo/          # Small example Python package
│   ├── __init__.py
│   ├── main.py
│   ├── utils.py
│   ├── processor.py
│   └── README.md
└── output/             # Analysis outputs for mini_repo
    ├── analysis.json              # Raw JSON analysis data
    ├── code_analysis_report.md    # Markdown report
    ├── code_analysis_report.html  # HTML report
    └── README.md
```

## Mini Repository

The `mini_repo/` directory contains a simple Python package designed to demonstrate various code analysis features:

- **Multiple modules**: Separate files for utilities, data processing, and main application logic
- **Classes and functions**: Examples of both to test definition detection
- **Dependencies**: Internal imports and standard library usage
- **Various patterns**: Different coding patterns for comprehensive analysis

## Running Analysis

### Using the CLI

From the repository root:

```bash
# Analyze mini_repo and save to artifacts/
python -m code_cartographer.cli analyze -d examples/mini_repo

# Analyze and save to a specific location
python -m code_cartographer.cli analyze \
  -d examples/mini_repo \
  -o examples/output/analysis.json
```

Or if you have installed the package:

```bash
codecart analyze -d examples/mini_repo -o examples/output/analysis.json
```

### Generated Outputs

The analysis generates three files:

1. **analysis.json** - Complete raw data in JSON format
   - File-level metrics
   - Function/class definitions
   - Call graphs
   - Orphaned code detection
   - Variable usage tracking

2. **code_analysis_report.md** - Human-readable Markdown report
   - Overview statistics
   - Call graph analysis
   - Orphaned code listing
   - File-by-file breakdown

3. **code_analysis_report.html** - Self-contained HTML report
   - All styles inline (no CDN dependencies)
   - Can be opened directly in any browser
   - Includes table of contents for easy navigation
   - Same content as Markdown but with better formatting

## Viewing the Results

### HTML Report

The HTML report is the recommended way to view results. Simply open it in your browser:

```bash
open examples/output/code_analysis_report.html
# or
xdg-open examples/output/code_analysis_report.html
# or just double-click the file
```

The HTML report is completely self-contained with:
- Inline CSS styling
- No external dependencies
- No need for a web server
- Works offline

### JSON Data

For programmatic access or further processing:

```python
import json

with open('examples/output/analysis.json') as f:
    analysis = json.load(f)

# Access different parts
print(f"Analyzed {len(analysis['files'])} files")
print(f"Found {len(analysis['orphans'])} orphaned elements")
print(f"Call graph has {len(analysis['call_graph'])} functions")
```

## Creating Your Own Examples

To add a new example repository:

1. Create a new directory under `examples/`
2. Add your Python code
3. Run analysis:
   ```bash
   codecart analyze -d examples/your_repo -o examples/your_repo_output/analysis.json
   ```
4. Document any interesting patterns or findings

## What Gets Analyzed

Code Cartographer analyzes:

- **Structure**: Files, functions, classes, methods
- **Dependencies**: Imports, internal references, call graphs
- **Complexity**: Cyclomatic complexity, maintainability index
- **Quality**: Orphaned code, unused variables, missing documentation
- **Patterns**: OOP patterns, comprehensions, async usage
- **Relationships**: Bidirectional call graphs, prerequisites

## Advanced Usage

### Exclude Patterns

Exclude specific files or directories:

```bash
codecart analyze -d examples/mini_repo \
  -e "tests/.*" -e ".*_test\.py" \
  -o output/analysis.json
```

### Generating Only Reports

If you already have analysis JSON:

```bash
# Generate Markdown and HTML from existing JSON
codecart report examples/output/analysis.json
```

### Interactive Dashboard

Generate an interactive dashboard (requires additional dependencies):

```bash
codecart visualize examples/output/analysis.json \
  -o examples/output/dashboard.html
```

## Troubleshooting

### Missing Dependencies

If you see errors about missing modules:

```bash
pip install -r requirements.txt
```

### Permission Errors

Make sure the output directory is writable:

```bash
mkdir -p artifacts
chmod 755 artifacts
```

## Learn More

- See the main [README](../README.md) for full documentation
- Check [mini_repo/README.md](mini_repo/README.md) for details about the example package
- Review [output/README.md](output/README.md) for information about generated outputs
