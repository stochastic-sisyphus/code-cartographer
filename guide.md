# Code Analysis & Variant Detection Guide

This toolset helps you analyze Python codebases to find code duplicates, compare variants, and generate insights. It's particularly useful when you have multiple versions or forks of a project.

## Setup

1. **Install Dependencies**
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install radon jinja2 networkx
```

2. **Required Files**
Copy these files to your project root:
- `deep_code_analyzer.py` - Core analysis engine
- `compare_code_variants.py` - Variant comparison and merging
- `run_code_cartography.sh` - Automation script
- `templates/dashboard.html.j2` - Dashboard template

## Usage Options

### Option 1: Full Automated Analysis

Use this when you want to analyze multiple codebases/forks at once:

```bash
# Make script executable
chmod +x run_code_cartography.sh

# Run full analysis
./run_code_cartography.sh
```

This will:
1. Analyze all code in specified directories
2. Compare and find variants
3. Generate diffs and merged code
4. Create an HTML dashboard

### Option 2: Step-by-Step Analysis

#### 1. Deep Code Analysis
Analyze a single codebase:
```bash
python deep_code_analyzer.py \
  -d /path/to/your/project \
  --output analysis_output.json \
  --markdown analysis_report.md \
  --graphviz dependency_graph.dot \
  --exclude "tests/.*" "build/.*" ".venv/.*" ".git/.*"
```

This generates:
- `analysis_output.json` - Detailed code analysis
- `analysis_report.md` - Human-readable summary
- `dependency_graph.dot` - Import dependency graph

#### 2. Compare Code Variants
If you have multiple analysis outputs:
```bash
python compare_code_variants.py compare \
  --summary-dir output \
  --diffs-dir output/diffs \
  --summary-csv output/summary.csv \
  --profile balanced \
  --match-threshold 0.7
```

Available profiles:
- `balanced` - Equal weight to all metrics
- `simplicity` - Favors simpler code
- `robustness` - Favors well-tested code
- `maintainability` - Favors readable code

#### 3. Merge Similar Code
```bash
python compare_code_variants.py merge \
  --summary-dir output \
  --output-dir merged_code \
  --format both \
  --preserve-structure
```

Format options:
- `script` - Python files only
- `markdown` - Markdown documentation
- `both` - Both formats
- `json` - JSON summary

#### 4. Generate Dashboard
```bash
python compare_code_variants.py dashboard \
  --summary-csv output/summary.csv \
  --diffs-dir output/diffs \
  --template-dir templates \
  --output dashboard.html \
  --show-all-variants
```

## Output Structure

```
project_root/
├── output/                     # Analysis outputs
│   ├── *.json                 # Raw analysis data
│   ├── *.md                   # Markdown reports
│   ├── *.dot                  # Dependency graphs
│   ├── diffs/                 # Code variant diffs
│   └── dashboard.html         # Interactive dashboard
├── merged_code/               # Merged variants
│   ├── *.py                   # Merged Python files
│   └── *.md                   # Documentation
```

## Advanced Usage

### Custom Exclusion Patterns
```bash
python deep_code_analyzer.py -d /path/to/project \
  --exclude \
    "tests/.*" \
    "build/.*" \
    ".venv/.*" \
    ".git/.*" \
    ".*_test\.py$" \
    "setup\.py"
```

### Variant Matching Sensitivity
```bash
python compare_code_variants.py compare \
  --summary-dir output \
  --match-threshold 0.8  # Higher = stricter matching
  --max-variants 5       # Limit variants per group
```

### Custom Scoring Weights
```bash
python compare_code_variants.py compare \
  --summary-dir output \
  --weights 1 2 3 1 1  # lines calls branches cc mi
```

## Analysis Features

The tools analyze:
1. **Code Structure**
   - Functions and classes
   - Import dependencies
   - Code complexity metrics

2. **Code Quality**
   - Cyclomatic complexity
   - Maintainability index
   - Type hint usage
   - Documentation coverage

3. **Patterns**
   - Code duplication
   - Similar implementations
   - Common patterns
   - Refactoring opportunities

4. **Dependencies**
   - Internal imports
   - External dependencies
   - Module relationships

## Best Practices

1. **Before Analysis**
   - Clean up your codebase
   - Remove generated files
   - Update `.gitignore` patterns

2. **During Analysis**
   - Start with default settings
   - Adjust thresholds if needed
   - Check intermediate outputs

3. **After Analysis**
   - Review the dashboard
   - Examine variant diffs
   - Consider refactoring suggestions

## Troubleshooting

1. **Syntax Errors**
   - The analyzer will warn about syntax errors
   - Fix the errors in the source files
   - Re-run the analysis

2. **Memory Issues**
   - Analyze smaller chunks
   - Increase exclusion patterns
   - Run step-by-step instead of automated

3. **Missing Variants**
   - Lower the match threshold
   - Check file exclusions
   - Verify file encoding

4. **Dashboard Issues**
   - Check template directory
   - Verify JSON/CSV files exist
   - Check file permissions

Remember to always backup your code before applying any automated changes or refactoring suggestions!
