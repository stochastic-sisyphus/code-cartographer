# code-cartographer  
## Deep Multilayer Static Analyzer for Python Projects  

---

As a masochist, I am never satisfied with a project until I reach perfection.  
To me, perfection is so far beyond running correctly or achieving 0 problems across all files in a directory.  
Although I’ve never actually achieved the elusive “perfect” finish line in any project ever, so I can’t be sure about the definition.
Therefore, I unsurprisingly constantly have at least a few iterations of the same project on my local machine that I’m actively making more refinements to at all times.  
This can cause confusion (shocker!), especially because I am reluctant to push or publish incomplete or “inadequate” code. I also have a fear of being perceived—and what’s more vulnerable than my code?
> *“For when your code is too chaotic for flake8 and too personal for git push.”*

Just like that, a vicious cycle is born.  
An unproductive, deeply confusing, memory-consuming vicious cycle.

Has any LLM ever in all of human history actually internalized even a tree structure to assist you in reorganizing a repo? Yeah, I didn’t think so. Not for me either. So if a vicious cycle is now my daily routine, at a certain point I decided to give myself the illusion of respite, however brief.  
This has given me solace at least once. Enjoy!

---

# Code Cartographer 
> *“If Git is for branches, this is for forks of forks.”*

---

## Features

- **Full file- and definition-level metadata**  
  Class/function blocks, line counts, docstrings, decorators, async flags, calls, type hints

- **Function/class SHA-256 hashes**  
  Detects variants, clones, and partial rewrites across versions

- **Cyclomatic complexity & maintainability index analysis (via `radon`)**  
  Flags “at-risk” code with CC > 10 or MI < 65

- **Auto-generated LLM refactor prompts**  
  Variant grouping, inline diffs, rewrite guidance

- **Internal dependency graph**  
  Outputs a Graphviz `.dot` of all intra-project imports

- **Markdown summary**  
  Skimmable digest with risk flags and structure

- **CLI flexibility**  
  Exclusion patterns, Git SHA tagging, output formatting, Markdown/Graphviz toggles

---

## Usage

### Single Project Analysis

```bash
python deep_code_analyzer.py \
  -d /path/to/project \
  --markdown summary.md \
  --graphviz deps.dot \
  --exclude "tests/.*" "build/.*"
```

### Comparing Multiple Versions
```bash
python deep_code_analyzer.py -d ./version-A -o version-A.json
python deep_code_analyzer.py -d ./version-B -o version-B.json
```

### Merging Summaries for Comparison
```bash
jq -n \
  --argfile A version-A.json \
  --argfile B version-B.json \
  '{ "version-A": $A, "version-B": $B }' > combined_summary.json
```

---

## Output Files

| File | Description |
| --- | --- |
| deep_code_summary.json | Machine-readable full analysis output |
| deep_code_summary.md | Human-friendly digest of structure/complexity |
| dependencies.dot | Graphviz internal import graph |

---

## Project Layout Example
.
├── deep_code_analyzer.py           # The analyzer
├── repo-local-V1/                  # One version of your project directory
├── repo-local-V1_summary.json      # JSON analysis output
├── repo-local-V1_summary.md        # Markdown digest
├── repo-local-V1_deps.dot          # Dependency graph

---

## Installation

Python 3.8+ required.

Install dependencies: 
```bash 
pip install radon
```

Optional tools:
- jq — for merging summaries
- Graphviz CLI — to render .dot files: 
 
```bash 
dot -Tpng dependencies.dot -o dependencies.png
```

---

## Author Notes

This tool exists to reconcile broken, duplicated, or ghost-forked Python projects.
It helps you detect what’s salvageable, refactor what’s duplicated, and visualize the mess you made.

Whether you’re dealing with:
- Fragmented directories
- Local edits lost to time
- Abandoned branches and reanimated scripts

This is for you.
Or at least, for the version of you that still wants to fix it.

> *“Structured remorse for unstructured code.”*

---

## License 
MIT License. See LICENSE file for details.

