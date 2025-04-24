#!/usr/bin/env python3
"""
Advanced Code Variant Analyzer
=============================
A sophisticated tool for analyzing, comparing, and merging similar code implementations
across Python codebases. Identifies code variants, generates detailed diffs, and 
provides intelligent merging suggestions.

Core Features:
-------------
1. Variant Detection:
   - AST-based similarity analysis
   - Configurable matching thresholds
   - Support for functions, classes, methods
   - Fuzzy name matching for renamed variants

2. Diff Generation:
   - Unified diff format
   - Side-by-side comparisons
   - Syntax-highlighted output
   - Detailed change annotations

3. Intelligent Merging:
   - Automatic variant consolidation
   - Preservation of unique features
   - Conflict resolution hints
   - Documentation merging

4. Analysis Dashboard:
   - Interactive HTML visualization
   - Variant relationship graphs
   - Diff browsing interface
   - Metrics and statistics

Usage:
------
```bash
# Compare variants
python code_variant_analyzer.py compare \\
    --summary-dir ./analysis \\
    --diffs-dir ./diffs \\
    --summary-csv ./summary.csv

# Merge variants
python code_variant_analyzer.py merge \\
    --summary-dir ./analysis \\
    --output-dir ./merged \\
    --format both

# Generate dashboard
python code_variant_analyzer.py dashboard \\
    --summary-csv ./summary.csv \\
    --diffs-dir ./diffs \\
    --template-dir ./templates \\
    --output ./dashboard.html
```
"""

import argparse
import ast
import csv
import difflib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import jinja2
from sklearn.base import defaultdict

# Matching profiles with pre-tuned thresholds
MATCHING_PROFILES = {
    "strict": {
        "name_similarity": 0.9,
        "structure_similarity": 0.9,
        "content_similarity": 0.9,
        "call_similarity": 0.9
    },
    "balanced": {
        "name_similarity": 0.7,
        "structure_similarity": 0.7,
        "content_similarity": 0.7,
        "call_similarity": 0.7
    },
    "lenient": {
        "name_similarity": 0.5,
        "structure_similarity": 0.5,
        "content_similarity": 0.5,
        "call_similarity": 0.5
    }
}

@dataclass
class CodeVariant:
    name: str
    path: str
    source: str
    ast_node: ast.AST
    complexity: Dict[str, Any]
    calls: Set[str]
    decorators: List[str]
    docstring: Optional[str]

@dataclass
class VariantGroup:
    base_variant: CodeVariant
    similar_variants: List[CodeVariant]
    similarity_scores: Dict[str, float]
    diffs: Dict[str, List[str]]

class VariantAnalyzer:
    """Analyzes code variants using AST-based similarity metrics."""
    
    def __init__(self, profile: str = "balanced"):
        self.thresholds = MATCHING_PROFILES[profile]
        self.variants: Dict[str, List[CodeVariant]] = {}
        
    def load_analysis_files(self, summary_dir: Path) -> None:
        """Load and parse analysis files from directory."""
        for json_file in summary_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                self._process_analysis_data(data, json_file)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                
    def _process_analysis_data(self, data: Dict, source_file: Path) -> None:
        """Extract variants from analysis data."""
        for file_summary in data.get("summary", []):
            for definition in file_summary.get("definitions", []):
                variant = CodeVariant(
                    name=definition["name"],
                    path=file_summary["path"],
                    source=definition["source_text"],
                    ast_node=ast.parse(definition["source_text"]),
                    complexity=definition["metrics"],
                    calls=set(definition["outbound_calls"]),
                    decorators=definition["decorators"],
                    docstring=definition["docstring"]
                )
                
                self.variants.setdefault(variant.name, []).append(variant)
                
    def find_similar_variants(self, include_similar: bool = False) -> List[VariantGroup]:
        """Identify groups of similar code variants."""
        groups = []
        processed = set()
        
        for name, variants in self.variants.items():
            if name in processed or len(variants) < 2:
                continue
                
            base = variants[0]
            similar = []
            scores = {}
            diffs = {}
            
            for variant in variants[1:]:
                similarity = self._compute_similarity(base, variant)
                if similarity >= self.thresholds["content_similarity"]:
                    similar.append(variant)
                    scores[variant.path] = similarity
                    diffs[variant.path] = self._generate_diff(base, variant)
                elif include_similar and similarity >= self.thresholds["content_similarity"] * 0.8:
                    similar.append(variant)
                    scores[variant.path] = similarity
                    diffs[variant.path] = self._generate_diff(base, variant)
                    
            if similar:
                groups.append(VariantGroup(base, similar, scores, diffs))
                processed.add(name)
                
        return groups
        
    def _compute_similarity(self, a: CodeVariant, b: CodeVariant) -> float:
        """Compute overall similarity score between two variants."""
        # Name similarity (allowing for minor variations)
        name_sim = self._fuzzy_name_match(a.name, b.name)
        
        # Structure similarity (AST-based)
        struct_sim = self._ast_similarity(a.ast_node, b.ast_node)
        
        # Content similarity (normalized Levenshtein)
        content_sim = self._normalized_levenshtein(a.source, b.source)
        
        # Call graph similarity
        call_sim = len(a.calls & b.calls) / max(len(a.calls | b.calls), 1)
        
        # Weighted combination
        weights = {
            "name": 0.2,
            "structure": 0.3,
            "content": 0.3,
            "calls": 0.2
        }
        
        return (
            weights["name"] * name_sim +
            weights["structure"] * struct_sim +
            weights["content"] * content_sim +
            weights["calls"] * call_sim
        )
        
    def _fuzzy_name_match(self, a: str, b: str) -> float:
        """Compute fuzzy similarity between names."""
        a_parts = set(re.split(r"[_.]", a.lower()))
        b_parts = set(re.split(r"[_.]", b.lower()))
        return len(a_parts & b_parts) / max(len(a_parts | b_parts), 1)
        
    def _ast_similarity(self, a: ast.AST, b: ast.AST) -> float:
        """Compute structural similarity between AST nodes."""
        a_nodes = {type(node).__name__ for node in ast.walk(a)}
        b_nodes = {type(node).__name__ for node in ast.walk(b)}
        return len(a_nodes & b_nodes) / max(len(a_nodes | b_nodes), 1)
        
    def _normalized_levenshtein(self, a: str, b: str) -> float:
        """Compute normalized Levenshtein similarity."""
        return difflib.SequenceMatcher(None, a, b).ratio()
        
    def _generate_diff(self, a: CodeVariant, b: CodeVariant) -> List[str]:
        """Generate unified diff between variants."""
        return list(difflib.unified_diff(
            a.source.splitlines(),
            b.source.splitlines(),
            fromfile=f"{a.path}:{a.name}",
            tofile=f"{b.path}:{b.name}",
            lineterm=""
        ))

class VariantMerger:
    """Handles intelligent merging of code variants."""
    
    def __init__(self, groups: List[VariantGroup], format: str = "both"):
        self.groups = groups
        self.format = format
        
    def merge_to_directory(self, output_dir: Path, preserve_structure: bool = True) -> None:
        """Merge variants while preserving directory structure."""
        for group in self.groups:
            merged_source = self._merge_variant_group(group)
            
            if preserve_structure:
                # Maintain original paths
                rel_path = Path(group.base_variant.path)
                target_dir = output_dir / rel_path.parent
                target_dir.mkdir(parents=True, exist_ok=True)
                
                if self.format in ("python", "both"):
                    py_file = target_dir / rel_path.name
                    py_file.write_text(merged_source)
                    
                if self.format in ("markdown", "both"):
                    md_file = target_dir / f"{rel_path.stem}.md"
                    md_file.write_text(self._generate_markdown(group, merged_source))
            else:
                # Flat output structure
                if self.format in ("python", "both"):
                    (output_dir / f"{group.base_variant.name}.py").write_text(merged_source)
                if self.format in ("markdown", "both"):
                    (output_dir / f"{group.base_variant.name}.md").write_text(
                        self._generate_markdown(group, merged_source)
                    )
                    
    def _merge_variant_group(self, group: VariantGroup) -> str:
        """Intelligently merge a group of variants."""
        base = group.base_variant

        # Merge docstrings
        docstrings = [v.docstring for v in [base] + group.similar_variants if v.docstring]
        merged_docstring = self._merge_docstrings(docstrings) if docstrings else None

        # Merge decorators
        all_decorators = {d for v in [base] + group.similar_variants for d in v.decorators}

        # Build merged source
        lines = []

        # Add decorators
        lines.extend(f"@{decorator}" for decorator in sorted(all_decorators))
        # Extract definition line (def/class line)
        def_line = base.source.splitlines()[0]
        lines.append(def_line)

        # Add merged docstring
        if merged_docstring:
            lines.extend(['    """', *merged_docstring.splitlines(), '    """'])

        # Add implementation (using base variant as primary)
        impl_lines = base.source.splitlines()[1:]
        if merged_docstring:
            # Skip original docstring if present
            in_doc = False
            filtered_lines = []
            for line in impl_lines:
                if '"""' in line or "'''" in line:
                    in_doc = not in_doc
                    continue
                if not in_doc:
                    filtered_lines.append(line)
            impl_lines = filtered_lines

        lines.extend(impl_lines)

        return "\n".join(lines)
        
    def _merge_docstrings(self, docstrings: List[str]) -> str:
        """Merge multiple docstrings intelligently."""
        sections = defaultdict(set)
        
        for doc in docstrings:
            current_section = "Description"
            current_content = []
            
            for line in doc.splitlines():
                line = line.strip()
                if line and line[0].isalpha() and line[-1] == ":":
                    # New section detected
                    if current_content:
                        sections[current_section].update(current_content)
                        current_content = []
                    current_section = line[:-1]
                else:
                    current_content.append(line)
                    
            if current_content:
                sections[current_section].update(current_content)
                
        # Combine sections
        merged = []
        for section, content in sections.items():
            if section != "Description":
                merged.append(f"\n{section}:")
            merged.extend(sorted(content))
            
        return "\n".join(merged)
        
    def _generate_markdown(self, group: VariantGroup, merged_source: str) -> str:
        """Generate Markdown documentation for merged variant."""
        lines = [
            f"# {group.base_variant.name}",
            "\n## Original Variants",
            f"- Base: `{group.base_variant.path}`"
        ]
        
        for variant in group.similar_variants:
            score = group.similarity_scores[variant.path]
            lines.append(f"- Similar ({score:.2f}): `{variant.path}`")
            
        lines.extend([
            "\n## Merged Implementation",
            "```python",
            merged_source,
            "```",
            "\n## Variant Diffs",
        ])
        
        for variant in group.similar_variants:
            lines.extend([
                f"\n### vs {variant.path}",
                "```diff",
                *group.diffs[variant.path],
                "```"
            ])
            
        return "\n".join(lines)

class DashboardGenerator:
    """Generates interactive HTML dashboard for variant analysis."""
    
    def __init__(self, template_dir: Path):
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    def generate(
        self,
        summary_csv: Path,
        diffs_dir: Path,
        output_file: Path,
        show_all: bool = False
    ) -> None:
        """Generate HTML dashboard from analysis results."""
        # Load variant data
        entries = []
        with open(summary_csv) as f:
            entries.extend(iter(csv.DictReader(f)))
        # Load diffs if showing all variants
        diffs = {}
        if show_all and diffs_dir.exists():
            for diff_file in diffs_dir.glob("*.diff"):
                name = diff_file.stem
                diffs[name] = diff_file.read_text()

        # Render template
        template = self.env.get_template("dashboard.html.j2")
        html = template.render(
            entries=entries,
            diffs=diffs,
            show_all_variants=show_all
        )

        output_file.write_text(html)

def compare_command(args):
    """Execute variant comparison analysis."""
    analyzer = VariantAnalyzer(args.profile)
    analyzer.load_analysis_files(args.summary_dir)
    
    groups = analyzer.find_similar_variants(args.include_similar)
    
    # Ensure output directories exist
    args.diffs_dir.mkdir(parents=True, exist_ok=True)
    
    # Write diffs
    for group in groups:
        for variant in group.similar_variants:
            diff_file = args.diffs_dir / f"{group.base_variant.name}.diff"
            diff_file.write_text("\n".join(group.diffs[variant.path]))
            
    # Generate summary CSV
    with open(args.summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, ["name", "winner", "score"])
        writer.writeheader()
        for group in groups:
            writer.writerow({
                "name": group.base_variant.name,
                "winner": group.base_variant.path,
                "score": max(group.similarity_scores.values()) if group.similarity_scores else 1.0
            })

def merge_command(args):
    """Execute variant merging."""
    analyzer = VariantAnalyzer(args.profile)
    analyzer.load_analysis_files(args.summary_dir)
    
    groups = analyzer.find_similar_variants(args.include_similar)
    merger = VariantMerger(groups, args.format)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merger.merge_to_directory(args.output_dir, args.preserve_structure)

def dashboard_command(args):
    """Generate analysis dashboard."""
    generator = DashboardGenerator(args.template_dir)
    generator.generate(
        args.summary_csv,
        args.diffs_dir,
        args.output,
        args.show_all_variants
    )

def main():
    parser = argparse.ArgumentParser(
        description="Advanced code variant analyzer and merger",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare code variants")
    compare_parser.add_argument(
        "--summary-dir",
        type=Path,
        required=True,
        help="Directory containing analysis JSON files"
    )
    compare_parser.add_argument(
        "--diffs-dir",
        type=Path,
        required=True,
        help="Output directory for diff files"
    )
    compare_parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Output path for summary CSV"
    )
    compare_parser.add_argument(
        "--profile",
        choices=MATCHING_PROFILES.keys(),
        default="balanced",
        help="Matching sensitivity profile"
    )
    compare_parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for matching"
    )
    compare_parser.add_argument(
        "--include-similar",
        action="store_true",
        help="Include slightly less similar matches"
    )
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge similar variants")
    merge_parser.add_argument(
        "--summary-dir",
        type=Path,
        required=True,
        help="Directory containing analysis JSON files"
    )
    merge_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for merged files"
    )
    merge_parser.add_argument(
        "--format",
        choices=["python", "markdown", "both"],
        default="both",
        help="Output format for merged files"
    )
    merge_parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve original directory structure"
    )
    merge_parser.add_argument(
        "--profile",
        choices=MATCHING_PROFILES.keys(),
        default="balanced",
        help="Matching sensitivity profile"
    )
    merge_parser.add_argument(
        "--include-similar",
        action="store_true",
        help="Include slightly less similar matches"
    )
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Generate analysis dashboard")
    dashboard_parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Path to summary CSV file"
    )
    dashboard_parser.add_argument(
        "--diffs-dir",
        type=Path,
        required=True,
        help="Directory containing diff files"
    )
    dashboard_parser.add_argument(
        "--template-dir",
        type=Path,
        required=True,
        help="Directory containing dashboard templates"
    )
    dashboard_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for dashboard HTML"
    )
    dashboard_parser.add_argument(
        "--show-all-variants",
        action="store_true",
        help="Show all variant diffs in dashboard"
    )
    
    args = parser.parse_args()
    
    if args.command == "compare":
        compare_command(args)
    elif args.command == "merge":
        merge_command(args)
    elif args.command == "dashboard":
        dashboard_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
