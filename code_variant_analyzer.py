#!/usr/bin/env python3
"""
Code Variant Analyzer
====================
Command-line interface for the code variant analyzer functionality.
This script wraps the core variant analyzer functionality for command-line use.
"""

import argparse
import json
import sys
from pathlib import Path

# Import the actual implementation from the package
from code_cartographer.core.variant_analyzer import VariantAnalyzer

def compare_command(args):
    """Run variant comparison analysis."""
    analyzer = VariantAnalyzer(
        root=Path(args.summary_dir).parent,
        semantic_threshold=0.7 if args.profile == "balanced" else 0.9 if args.profile == "strict" else 0.5,
        min_lines=5,
    )
    
    # Load analysis data
    with open(Path(args.summary_dir) / "analysis.json", "r") as f:
        analysis_data = json.load(f)
    
    # Run variant analysis
    results = analyzer.analyze(analysis_data)
    
    # Save results
    Path(args.diffs_dir).mkdir(parents=True, exist_ok=True)
    
    # Write summary CSV
    with open(args.summary_csv, "w") as f:
        f.write("name,variants,risk_level,similarity\n")
        for name, data in results.items():
            f.write(f"{name},{len(data['implementations'])},{data.get('risk_level', 'low')},{data.get('similarity', 0.0)}\n")
    
    # Write detailed diffs
    for name, data in results.items():
        diff_file = Path(args.diffs_dir) / f"{name.replace('.', '_')}.json"
        with open(diff_file, "w") as f:
            json.dump(data, f, indent=2)
    
    print(f"[INFO] Variant analysis complete: {args.summary_csv}")
    print(f"[INFO] Detailed diffs saved to: {args.diffs_dir}")

def merge_command(args):
    """Run variant merging."""
    analyzer = VariantAnalyzer(
        root=Path(args.summary_dir).parent,
        semantic_threshold=0.7 if args.profile == "balanced" else 0.9 if args.profile == "strict" else 0.5,
        min_lines=5,
    )
    
    # Load analysis data
    with open(Path(args.summary_dir) / "analysis.json", "r") as f:
        analysis_data = json.load(f)
    
    # Run variant analysis if not already done
    results = analyzer.analyze(analysis_data)
    
    # Merge variants
    merged = analyzer.merge_variants(results, preserve_structure=args.preserve_structure)
    
    # Save merged implementations
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, implementation in merged.items():
        output_file = Path(args.output_dir) / f"{name.replace('.', '_')}.py"
        with open(output_file, "w") as f:
            f.write(implementation)
    
    print(f"[INFO] Merged variants saved to: {args.output_dir}")

def dashboard_command(args):
    """Generate dashboard visualization."""
    from jinja2 import Environment, FileSystemLoader
    import pandas as pd
    
    # Load summary data
    summary_df = pd.read_csv(args.summary_csv)
    
    # Load diff details
    diffs_data = {}
    for diff_file in Path(args.diffs_dir).glob("*.json"):
        with open(diff_file, "r") as f:
            diffs_data[diff_file.stem] = json.load(f)
    
    # Set up Jinja environment
    env = Environment(loader=FileSystemLoader(args.template_dir))
    template = env.get_template("dashboard.html.j2")
    
    # Render dashboard
    dashboard_html = template.render(
        summary=summary_df.to_dict(orient="records"),
        diffs=diffs_data,
        show_all=args.show_all_variants,
    )
    
    # Write output
    with open(args.output, "w") as f:
        f.write(dashboard_html)
    
    print(f"[INFO] Dashboard generated: {args.output}")

def main():
    """Main entry point for the code variant analyzer."""
    parser = argparse.ArgumentParser(
        description="Code Variant Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare code variants")
    compare_parser.add_argument("--summary-dir", required=True, help="Directory containing analysis.json")
    compare_parser.add_argument("--diffs-dir", required=True, help="Directory to save diff files")
    compare_parser.add_argument("--summary-csv", required=True, help="Path to save summary CSV")
    compare_parser.add_argument("--profile", choices=["strict", "balanced", "lenient"], default="balanced", 
                               help="Matching profile (strict=90%, balanced=70%, lenient=50%)")
    compare_parser.add_argument("--include-similar", action="store_true", help="Include semantically similar variants")
    compare_parser.set_defaults(func=compare_command)
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge code variants")
    merge_parser.add_argument("--summary-dir", required=True, help="Directory containing analysis.json")
    merge_parser.add_argument("--output-dir", required=True, help="Directory to save merged implementations")
    merge_parser.add_argument("--format", choices=["python", "markdown", "both"], default="both",
                             help="Output format for merged implementations")
    merge_parser.add_argument("--preserve-structure", action="store_true", help="Preserve original code structure")
    merge_parser.add_argument("--profile", choices=["strict", "balanced", "lenient"], default="balanced",
                             help="Matching profile (strict=90%, balanced=70%, lenient=50%)")
    merge_parser.add_argument("--include-similar", action="store_true", help="Include semantically similar variants")
    merge_parser.set_defaults(func=merge_command)
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Generate analysis dashboard")
    dashboard_parser.add_argument("--summary-csv", required=True, help="Path to summary CSV")
    dashboard_parser.add_argument("--diffs-dir", required=True, help="Directory containing diff files")
    dashboard_parser.add_argument("--template-dir", required=True, help="Directory containing dashboard template")
    dashboard_parser.add_argument("--output", required=True, help="Path to save dashboard HTML")
    dashboard_parser.add_argument("--show-all-variants", action="store_true", help="Show all variants in dashboard")
    dashboard_parser.set_defaults(func=dashboard_command)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
