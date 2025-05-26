# User Guide: Code Cartographer

## Introduction

Code Cartographer is a comprehensive static analysis tool designed to help you understand complex Python codebases by mapping the relationships between functions, variables, classes, and other code elements. This guide will walk you through how to use the enhanced version of Code Cartographer to analyze your Python projects.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/stochastic-sisyphus/code-cartographer.git
   cd code-cartographer
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Install system dependencies:

   ```bash
   # For Ubuntu/Debian
   sudo apt-get install graphviz graphviz-dev

   # For macOS
   brew install graphviz

   # For Windows
   # Download and install from https://graphviz.org/download/
   ```

## Running Code Cartographer

### Basic Usage

The simplest way to analyze your codebase is to use the provided shell script:

```bash
./analyze_codebase.sh /path/to/your/project
```

This will:

1. Analyze your codebase
2. Generate visualizations
3. Create comprehensive reports
4. Save all outputs to the `analysis_output` directory

### Advanced Usage

For more control over the analysis process, you can use the Python module directly:

```python
from code_cartographer.core.analyzer import CodeAnalyzer
from pathlib import Path

# Initialize the analyzer
analyzer = CodeAnalyzer(
    project_root=Path("/path/to/your/project"),
    output_dir=Path("/path/to/output")
)

# Run the analysis
results = analyzer.analyze(
    exclude_patterns=["tests/", "venv/", "build/"]
)

# Generate reports
report_path = analyzer.generate_report(results)
graph_path = analyzer.generate_call_graph(results)

print(f"Report generated at: {report_path}")
print(f"Call graph generated at: {graph_path}")
```

## Understanding the Results

### Code Analysis Report

The main output is a comprehensive Markdown report (`code_analysis_report.md`) that includes:

1. **Summary Statistics**
   - Total files analyzed
   - Function, class, and variable counts
   - Orphaned code elements
   - Code complexity metrics

2. **Orphaned Code**
   - Functions defined but never called
   - Classes defined but never instantiated
   - Variables defined but never used

3. **Code Variants**
   - Similar implementations across the codebase
   - Potential refactoring opportunities

4. **Dependency Analysis**
   - Initialization order requirements
   - Circular dependencies
   - Prerequisite relationships

5. **Variable Usage**
   - Variable definitions and usages
   - Scope analysis
   - Redefinition detection

### Visualizations

The tool generates several visualizations to help you understand your codebase:

1. **Call Graph**
   - Shows which functions call which other functions
   - Bidirectional relationships
   - Entry points and leaf nodes

2. **Dependency Graph**
   - Shows prerequisites between code elements
   - Helps identify initialization order requirements

3. **Variable Usage Chart**
   - Shows where variables are defined and used
   - Highlights orphaned variables

4. **Class Hierarchy**
   - Visualizes inheritance relationships
   - Shows method overrides

5. **Initialization Sequence**
   - Suggests a safe order for initializing components
   - Handles circular dependencies

## Interpreting the Results

### Identifying Code Issues

Look for:

- **Orphaned Functions/Classes**: These might be dead code that can be removed
- **Circular Dependencies**: These can cause initialization problems
- **Variables with Multiple Definitions**: Potential naming conflicts
- **High Complexity Metrics**: Functions that might need refactoring

### Improving Code Structure

Use the dependency analysis to:

- Reorganize code to reduce circular dependencies
- Identify modules that should be split
- Find opportunities for better encapsulation

### Cleaning Up Code

The orphan analysis helps you:

- Remove unused code
- Identify forgotten implementations
- Clean up unused variables

## Advanced Features

### Custom Exclusion Patterns

You can specify patterns to exclude from analysis:

```python
analyzer.analyze(exclude_patterns=[
    r"\.git/",
    r"\.venv/",
    r"__pycache__/",
    r"tests/",
    r"examples/"
])
```

### Focus on Specific Areas

To analyze only certain parts of your codebase:

```python
analyzer.analyze(focus_paths=[
    "src/core/",
    "src/utils/important_module.py"
])
```

### Custom Reporting

Generate specialized reports:

```python
# Generate a report focusing on orphaned code
analyzer.generate_orphan_report(results)

# Generate a report on variable usage
analyzer.generate_variable_report(results)
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all Python dependencies are installed
   - Verify Graphviz is installed correctly

2. **Memory Issues with Large Codebases**
   - Analyze smaller portions of the codebase
   - Increase available memory

3. **Visualization Errors**
   - Check Graphviz installation
   - Try different output formats (PNG, SVG, PDF)

4. **Incorrect Analysis Results**
   - Verify exclusion patterns aren't too broad
   - Check for dynamic code generation or eval usage

## Getting Help

If you encounter issues or have questions:

- Check the GitHub repository for updates
- Open an issue with detailed information about your problem
- Consult the API documentation for advanced usage
