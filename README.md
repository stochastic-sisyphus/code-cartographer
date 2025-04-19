# Code Cartographer  
## Deep Multilayer Static Analyzer for Python Projects  

---

As a masochist, I am never satisfied with a project until I reach perfection.  
To me, perfection is so far beyond running correctly or achieving 0 problems across all files in a directory.  
Although I've never actually achieved the elusive "perfect" finish line in any project ever, so I can't be sure about the definition.
It's also just what happens when you're experimenting. When you're deep in research, you're iterating in a vacuum; by the time something works, you've rewritten it five times. Therefore, I unsurprisingly constantly have at least a few iterations of the same project on my local machine that I'm actively making more refinements to at all times.  
This can cause confusion (shocker!), especially because I am reluctant to push or publish incomplete or "inadequate" code. I also have a fear of being perceived—and what's more vulnerable than my code?
> *"For when your code is too chaotic for flake8 and too personal for git push."*

Just like that, a vicious cycle is born.  
An unproductive, deeply confusing, memory-consuming vicious cycle.

Unfortunately the cycle is **much** harder to follow when there are dozens of moving parts in dozens of subfolders, each with (dozens) of lengthy scripts. You could feed your directory setup as context in an attempt to gain some clarity, but that's a gamble that backfires 9 times out of 10. Has any LLM ever in all of human history actually internalized any tree structure to assist you in reorganizing a repo? Yeah, I didn't think so. Not for me either. So if a vicious cycle is now my daily routine, at a certain point I decided to give myself the illusion of respite, however brief. This has given me solace at least once. Enjoy!

---

# Code Cartographer 
> *"If Git is for branches, this is for forks of forks."*

---

## Features

- **Full file and definition level metadata**  
  Class/function blocks, line counts, docstrings, decorators, async flags, calls, type hints

- **Function/class SHA-256 hashes**  
  Detects variants, clones, and partial rewrites across versions

- **Cyclomatic complexity & maintainability index analysis (via `radon`)**  
  Flags "at-risk" code with CC > 10 or MI < 65

- **Auto-generated LLM refactor prompts**  
  Variant grouping, inline diffs, rewrite guidance

- **Internal dependency graph**  
  Outputs a Graphviz `.dot` of all intra-project imports

- **Markdown summary**  
  Skimmable digest with risk flags and structure

- **Interactive Dashboard**  
  Visual analysis of code complexity, variants, and dependencies

- **CLI flexibility**  
  Exclusion patterns, Git SHA tagging, output formatting, Markdown/Graphviz toggles

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- (Optional) Graphviz for dependency visualization
- (Optional) jq for JSON manipulation

### Installation

1. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Required Files Structure**
```
your-project/
├── code_analyzer_engine.py     # Core analysis engine
├── code_variant_analyzer.py    # Variant detection and merging
├── analyze_codebase.sh        # Main automation script
├── templates/
│   └── dashboard.html.j2      # Dashboard template
└── requirements.txt           # Dependencies
```

---

## Usage Guide

### Quick Start
The simplest way to analyze any project:
```bash
./analyze_codebase.sh --project-dir /path/to/your/project
```

This will:
1. Run deep code analysis
2. Detect code variants
3. Generate comparison reports
4. Create an interactive dashboard

### Advanced Usage

#### 1. Deep Code Analysis
```bash
python code_analyzer_engine.py \
  -d /path/to/project \
  --markdown summary.md \
  --graphviz deps.dot \
  --exclude "tests/.*" "build/.*"
```

#### 2. Code Variant Analysis
```bash
# Compare variants
python code_variant_analyzer.py compare \
    --summary-dir ./analysis \
    --diffs-dir ./diffs \
    --summary-csv ./summary.csv \
    --profile balanced

# Merge similar code
python code_variant_analyzer.py merge \
    --summary-dir ./analysis \
    --output-dir ./merged \
    --format both

# Generate dashboard
python code_variant_analyzer.py dashboard \
    --summary-csv ./summary.csv \
    --diffs-dir ./diffs \
    --template-dir ./templates \
    --output ./dashboard.html
```

#### 3. Comparing Multiple Versions
```bash
python deep_code_analyzer.py -d ./version-A -o version-A.json
python deep_code_analyzer.py -d ./version-B -o version-B.json

# Merge summaries (requires jq)
jq -n \
  --argfile A version-A.json \
  --argfile B version-B.json \
  '{ "version-A": $A, "version-B": $B }' > combined_summary.json
```

### Customization Options

#### Matching Profiles
```bash
--profile strict      # 90% similarity required
--profile balanced   # 70% similarity required
--profile lenient    # 50% similarity required
```

#### Output Formats
```bash
--format python     # Python files only
--format markdown   # Markdown documentation
--format both      # Both formats
```

---

## Output Structure

After analysis, you'll find:

```
analyzed-project/
├── code_analysis/           
│   ├── analysis.json       # Complete analysis data
│   ├── analysis.md         # Human-readable summary
│   ├── dependencies.dot    # Dependency graph
│   ├── diffs/             # Code variant differences
│   └── dashboard.html      # Interactive dashboard
└── merged_code/           # Merged variant implementations
```

### Dashboard Features
1. Overview metrics and trends
2. Code variant analysis
3. Complexity distribution
4. Dependency visualization
5. Documentation coverage

### Key Metrics
- Total files and trends
- Code variant count
- Average complexity
- Documentation coverage
- High-complexity files
- Most referenced modules
- External dependencies

---

## Best Practices

### 1. Regular Analysis
```bash
# Add to your CI/CD pipeline
./analyze_codebase.sh --project-dir . --output-dir ./analysis-$(date +%Y%m%d)
```

### 2. Large Projects
```bash
# Analyze specific directories
./analyze_codebase.sh \
    --project-dir ./src \
    --exclude "tests/.*,docs/.*,*.pyc,__pycache__/.*"
```

### 3. Memory Management
For very large projects:
```bash
# Analyze in chunks
for dir in src/*/; do
    ./analyze_codebase.sh --project-dir "$dir" --output-dir "analysis-$(basename $dir)"
done
```

---

## Integration Examples

### GitHub Actions
```yaml
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
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
          ./analyze_codebase.sh --project-dir .
      - uses: actions/upload-artifact@v2
        with:
          name: code-analysis
          path: code_analysis/
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

./analyze_codebase.sh --project-dir . --output-dir ./latest-analysis
if [ $? -ne 0 ]; then
    echo "Code analysis failed - please review"
    exit 1
fi
```

---

## Troubleshooting

### Memory Issues
```bash
# Reduce analysis scope
./analyze_codebase.sh \
    --project-dir . \
    --exclude "tests/.*,docs/.*,*.pyc,__pycache__/.*,migrations/.*"
```

### Slow Analysis
```bash
# Focus on specific directories
./analyze_codebase.sh --project-dir ./src/core
```

### Dashboard Issues
- Ensure all JavaScript dependencies are accessible
- Check browser console for errors
- Verify JSON data structure matches template expectations

---

## Author Notes

This tool exists to reconcile broken, duplicated, or ghost-forked Python projects.
It helps you detect what's salvageable, refactor what's duplicated, and visualize the mess you made.

Whether you're dealing with:
- Fragmented directories
- Local edits lost to time
- Abandoned branches and reanimated scripts

This is for you.
Or at least, for the version of you that still wants to fix it.

> *"Structured remorse for unstructured code."*

---

## License 
MIT License. See LICENSE file for details. 
