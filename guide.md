# Complete Guide to Using the Code Cartographer Toolkit

## Setup

1. **Create Project Directory**
```bash
mkdir code-analysis-toolkit
cd code-analysis-toolkit
```

2. **Install Required Files**
```
code-analysis-toolkit/
├── code_analyzer_engine.py     # Core static analysis
├── code_variant_analyzer.py    # Variant detection/merging
├── analyze_codebase.sh        # Main automation script
├── templates/
│   └── dashboard.html.j2      # Dashboard template
└── requirements.txt           # Dependencies
```

3. **Set Up Environment**
```bash
# The script will automatically create this if it doesn't exist
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Basic Usage

### Single Command Analysis
Analyze any Python project with one command:
```bash
./analyze_codebase.sh --project-dir /path/to/your/project
```

This generates:
```
/path/to/your/project/
├── code_analysis/
│   ├── analysis.json         # Complete analysis data
│   ├── analysis.md          # Human-readable summary
│   ├── dependencies.dot     # Dependency graph
│   ├── diffs/              # Code variant differences
│   └── dashboard.html      # Interactive visualization
└── merged_code/            # Merged variant implementations
```

### Customizing Analysis

1. **Change Output Location**:
```bash
./analyze_codebase.sh \
    --project-dir /path/to/your/project \
    --output-dir ./my-analysis
```

2. **Exclude Patterns**:
```bash
./analyze_codebase.sh \
    --project-dir /path/to/your/project \
    --exclude "tests/.*,build/.*,.venv/.*,docs/.*"
```

## Component-by-Component Usage

### 1. Static Code Analysis
```bash
python code_analyzer_engine.py \
    -d /path/to/project \
    --output analysis.json \
    --markdown report.md \
    --graphviz deps.dot \
    --exclude "tests/.*" "build/.*" ".venv/.*"
```

Options:
- `-d, --dir`: Project to analyze
- `--output`: JSON output file
- `--markdown`: Generate Markdown report
- `--graphviz`: Generate dependency graph
- `--exclude`: Patterns to exclude
- `--no-git`: Skip git SHA tagging
- `--indent`: JSON indentation level

### 2. Code Variant Analysis

a) **Compare Variants**:
```bash
python code_variant_analyzer.py compare \
    --summary-dir ./analysis \
    --diffs-dir ./diffs \
    --summary-csv ./summary.csv \
    --profile balanced \
    --match-threshold 0.7
```

b) **Merge Similar Code**:
```bash
python code_variant_analyzer.py merge \
    --summary-dir ./analysis \
    --output-dir ./merged \
    --format both \
    --preserve-structure
```

c) **Generate Dashboard**:
```bash
python code_variant_analyzer.py dashboard \
    --summary-csv ./summary.csv \
    --diffs-dir ./diffs \
    --template-dir ./templates \
    --output ./dashboard.html
```

### 3. Matching Profiles
```bash
--profile strict    # 90% similarity required
--profile balanced  # 70% similarity required
--profile lenient   # 50% similarity required
```

## Understanding the Output

### 1. Analysis JSON Structure
```json
{
    "files": [
        {
            "path": "file.py",
            "imports": [...],
            "definitions": [...],
            "metrics": {
                "cyclomatic": 5,
                "maintainability_index": 75.0,
                "risk_flag": false
            }
        }
    ],
    "variants": {...},
    "dependencies": [...]
}
```

### 2. Dashboard Sections
- Overview metrics
- Code variant analysis
- Complexity distribution
- Dependency visualization
- Documentation coverage

## Advanced Usage

### 1. Large Projects
```bash
# Analyze specific directories
./analyze_codebase.sh \
    --project-dir ./src \
    --exclude "tests/.*,docs/.*,*.pyc"
```

### 2. CI/CD Integration

```yaml
# .github/workflows/code-analysis.yml
name: Code Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - name: Run Analysis
        run: |
          ./analyze_codebase.sh --project-dir .
      - uses: actions/upload-artifact@v2
        with:
          name: code-analysis
          path: code_analysis/
```

### 3. Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

./analyze_codebase.sh --project-dir .
if [ $? -ne 0 ]; then
    echo "Code analysis failed"
    exit 1
fi
```

## Troubleshooting

### Common Issues:

1. **Memory Problems**:
```bash
# Reduce analysis scope
./analyze_codebase.sh \
    --project-dir . \
    --exclude "tests/.*,docs/.*,*.pyc,__pycache__/.*"
```

2. **Slow Analysis**:
```bash
# Focus on core directories
./analyze_codebase.sh --project-dir ./src/core
```

3. **Dashboard Issues**:
- Check browser console
- Verify template paths
- Ensure all JSON files exist

### Best Practices:

1. **Regular Analysis**:
- Run on each commit/PR
- Track metrics over time
- Review variant reports regularly

2. **Output Management**:
- Version control analysis results
- Archive dashboards by date
- Track trends in metrics

3. **Customization**:
- Adjust similarity thresholds based on needs
- Customize exclusion patterns
- Modify dashboard template for specific metrics

Remember to always backup your code before applying any automated changes or refactoring suggestions!

__xoxo stochastic-sisyphus__
