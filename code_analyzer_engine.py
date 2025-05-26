#!/usr/bin/env python3
"""
Code Analyzer Engine
===================
Main entry point for the code analysis functionality.
This script wraps the core analyzer functionality for command-line use.
"""

import argparse
import json
import sys
from pathlib import Path

# Import the actual implementation from the package
from code_cartographer.core.analyzer import (
    ProjectAnalyzer,
    generate_dependency_graph,
    generate_markdown,
)

def main():
    """Main entry point for the code analyzer engine."""
    parser = argparse.ArgumentParser(
        description="Advanced Python codebase analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Root directory of the project to analyze",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("code_analysis.json"),
        help="Path for JSON output",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        nargs="*",
        default=[],
        help="Regex patterns for files/dirs to exclude",
    )

    parser.add_argument(
        "--markdown", type=Path, help="Generate Markdown report at specified path"
    )

    parser.add_argument(
        "--graphviz",
        type=Path,
        help="Generate Graphviz dependency graph at specified path",
    )

    parser.add_argument("--no-git", action="store_true", help="Skip git SHA tagging")

    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level")

    args = parser.parse_args()

    # Run analysis
    analyzer = ProjectAnalyzer(args.dir, args.exclude)
    analysis = analyzer.execute()

    # Write JSON output
    args.output.write_text(json.dumps(analysis, indent=args.indent))
    print(f"[INFO] Analysis complete: {args.output.resolve()}")

    # Generate additional outputs if requested
    if args.markdown:
        generate_markdown(analysis, args.markdown)
        print(f"[INFO] Markdown report: {args.markdown.resolve()}")

    if args.graphviz:
        generate_dependency_graph(analysis["dependencies"], args.graphviz)
        print(f"[INFO] Dependency graph: {args.graphviz.resolve()}")

if __name__ == "__main__":
    main()
