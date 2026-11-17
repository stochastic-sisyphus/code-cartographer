# Code Cartographer  

## Automated Deep Multilayer Code Analysis and Optimization for Large Scale Codebases 

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
>
> *"If Git is for branches, this is for forks of forks."*

---

## Features

- **Full file and definition level metadata**  
  Class/function blocks, line counts, docstrings, decorators, async flags, calls, type hints

- **Intelligent Code Normalization**  
  Standardizes variable names, function signatures, and code structure for better comparison

- **Advanced Variant Detection & Merging**  
  Automatically identifies and merges similar code blocks with semantic analysis

- **Auto-patching System**  
  Safely applies merged variants with automatic backup creation

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
  Exclusion patterns, Git SHA tagging, output formatting, variant merging controls

## Temporal Topography 

**Temporal Traversal**

code cartography absolutely did not require a new feat but alas i present: temporal traversal. 
ive never been creative ever 
my brain has one single mode: alliteration. 
normalize making fun of this level of infuriating cringe bc no one talks about the youth's clumsy alliteration and its rampant! 
i simply cannot help myself we need to invest in the epidemic that is obnoxious literary device usage. # alliteration theme on 100 # codebase's evolution through time # i <3 time 

- navigate git history and track code evolution
- complexity trends, dependency graphs, file hotspots
- automatically identify renames, splits, merges, and extractions
- fast, responsive interface built with FastAPI and vanilla JavaScript (ha ew)
- WebSocket-based progress tracking during analysis
- code complexity changes over time
- et al
- barney and friends 

### Quick Start with Temporal Topography

```bash
# Start the web server
python -m code_cartographer serve

# Open browser to http://localhost:8000
# Click "Analyze Project" and enter your project path
# Explore your codebase through interactive visualizations
```

See [Temporal Topography Documentation](docs/TEMPORAL_TOPOGRAPHY.md) for complete guide and API reference.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
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

2. **Install Dependencies**

Recommended: Use mise for automatic environment setup

```bash
mise trust
mise install
mise run install
```

Or manually:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
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

# Initialize analyzer with custom settings
analyzer = VariantAnalyzer(
    root="/path/to/project",
    semantic_threshold=0.8,  # 80% similarity required
    min_lines=5  # Minimum lines for variant consideration
)

# Run analysis
results = analyzer.analyze()

# Apply merged variants (with automatic backups)
analyzer.apply_merged_variants(backup=True)
```

#### 3. CLI Variant Management

```bash
# Analyze and merge variants with backups
code-cartographer variants -d /path/to/project --apply-merges

# Analyze and merge variants without backups
code-cartographer variants -d /path/to/project --apply-merges --no-backup

# Analyze with custom similarity threshold
code-cartographer variants -d /path/to/project --semantic-threshold 0.9
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

_honestly, ur so real for that_

- Reduce analysis scope using exclude patterns
- Process directories sequentially for large projects
- Use the similarity threshold in VariantAnalyzer to limit comparisons
- Get a better machine, idk 

### Performance Tips

_no clue lol_

### Common Issues

[DSM-5](https://www.psychiatry.org/psychiatrists/practice/dsm/)

---

## Author Notes

This tool exists to reconcile broken, duplicated, or ghost-forked Python projects.
It helps you detect what's salvageable, refactor what's duplicated, and look the mess you made.

Whether you're dealing with:

- Fragmented directories
- Local edits lost to time
- Abandoned branches and reanimated scripts

You are not alone. You can call our hotline 24/7 for support 

This is for you.
Or at least, for the version of you that still wants to fix it.

> *"Structured remorse for unstructured code."*

---

## License

MIT License. See LICENSE file for details.
patent pending! _borat_: NOT!
stochastic-sisyphus TM 
no trademark just @stochastic-sisyphus
ttyl
xoxo 
gossip sisyphus 
