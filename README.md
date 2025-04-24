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

### Installation

You can install code-cartographer directly from PyPI:

```bash
pip install code-cartographer
```

For development installation:

1. **Clone the Repository**
```bash
git clone https://github.com/stochastic-sisyphus/code-cartographer.git
cd code-cartographer
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install in Development Mode**
```bash
pip install -e ".[dev]"
```

---

## Usage Guide

### Quick Start

The code-cartographer package provides tools for analyzing Python codebases:

```python
from code_cartographer import ProjectAnalyzer, VariantAnalyzer

# Initialize analyzers
project_analyzer = ProjectAnalyzer("/path/to/your/project")
variant_analyzer = VariantAnalyzer()

# Run analysis
analysis_results = project_analyzer.analyze()
variant_results = variant_analyzer.analyze(analysis_results)

# Generate reports
project_analyzer.generate_markdown("analysis.md")
project_analyzer.generate_dependency_graph("dependencies.dot")
```

### Advanced Usage

#### 1. Deep Code Analysis
```python
from code_cartographer import ProjectAnalyzer

analyzer = ProjectAnalyzer(
    project_dir="/path/to/project",
    exclude_patterns=["tests/.*", "build/.*"]
)

# Run analysis
results = analyzer.analyze()

# Generate reports
analyzer.generate_markdown("summary.md")
analyzer.generate_dependency_graph("deps.dot")
```

#### 2. Code Variant Analysis
```python
from code_cartographer import VariantAnalyzer

analyzer = VariantAnalyzer(
    similarity_threshold=0.7,  # 70% similarity required
    normalize_code=True
)

# Analyze for variants
variants = analyzer.analyze(source_files)

# Get variant groups
groups = analyzer.get_variant_groups()

# Generate comparison report
analyzer.generate_report("variants.md")
```

---

## Output Structure

After analysis, you'll find:

```
analyzed-project/
├── analysis.md         # Human-readable summary
├── dependencies.dot    # Dependency graph (if Graphviz is installed)
└── variants.md        # Code variant analysis report
```

### Key Metrics
- Code complexity metrics
- Import dependencies
- Function/class definitions
- Documentation coverage
- Code variants and duplicates
- Semantic similarity scores

---

## Best Practices

### 1. Regular Analysis
```python
from code_cartographer import ProjectAnalyzer
import datetime

# Add to your analysis pipeline
analyzer = ProjectAnalyzer(".")
results = analyzer.analyze()
date_str = datetime.datetime.now().strftime("%Y%m%d")
analyzer.generate_markdown(f"analysis-{date_str}.md")
```

### 2. Large Projects
```python
from code_cartographer import ProjectAnalyzer

# Analyze specific directories with exclusions
analyzer = ProjectAnalyzer(
    "src",
    exclude_patterns=[
        "tests/.*",
        "docs/.*",
        "*.pyc",
        "__pycache__/.*"
    ]
)
results = analyzer.analyze()
```

---

## Troubleshooting

### Memory Issues
- Reduce analysis scope using exclude patterns
- Process directories sequentially for large projects
- Use the similarity threshold in VariantAnalyzer to limit comparisons

### Performance Tips
- Focus analysis on specific directories
- Use appropriate similarity thresholds
- Leverage code normalization options

### Common Issues
- Ensure Python 3.8+ is being used
- Check file permissions for output directories
- Verify Graphviz installation for dependency graphs

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
