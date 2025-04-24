"""
Code Cartographer CLI
====================
Command line interface for the code cartographer tool.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from code_cartographer.core.analyzer import ProjectAnalyzer, generate_markdown, generate_dependency_graph
from code_cartographer.core.variant_analyzer import VariantAnalyzer

def analyze_command(args: argparse.Namespace) -> None:
    """Run the main code analysis."""
    analyzer = ProjectAnalyzer(
        root=args.dir,
        exclude_patterns=args.exclude or []
    )
    
    analysis = analyzer.execute()
    
    # Write JSON output
    args.output.write_text(
        json.dumps(analysis, indent=2)
    )
    print(f"[INFO] Analysis complete: {args.output.resolve()}")
    
    # Generate additional outputs if requested
    if args.markdown:
        generate_markdown(analysis, args.markdown)
        print(f"[INFO] Markdown report: {args.markdown.resolve()}")
        
    if args.graphviz:
        generate_dependency_graph(analysis["dependencies"], args.graphviz)
        print(f"[INFO] Dependency graph: {args.graphviz.resolve()}")

def variants_command(args: argparse.Namespace) -> None:
    """Run variant analysis."""
    analyzer = VariantAnalyzer(
        root=args.dir,
        semantic_threshold=args.semantic_threshold,
        min_lines=args.min_lines,
        exclude_patterns=args.exclude
    )
    
    analysis = analyzer.analyze()
    
    args.output.write_text(
        json.dumps(analysis, indent=2)
    )
    print(f"[INFO] Variant analysis complete: {args.output.resolve()}")

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Code Cartographer - Advanced Python Codebase Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True
    )

    analyze_parser = _extracted_from_main_15(
        subparsers, "analyze", "Run deep code analysis", "code_analysis.json"
    )
    analyze_parser.add_argument(
        "-e", "--exclude",
        nargs="*",
        help="Regex patterns for paths to exclude"
    )

    analyze_parser.add_argument(
        "--markdown",
        type=Path,
        help="Generate Markdown report at specified path"
    )

    analyze_parser.add_argument(
        "--graphviz",
        type=Path,
        help="Generate Graphviz dependency graph at specified path"
    )

    variants_parser = _extracted_from_main_15(
        subparsers,
        "variants",
        "Analyze code variants and duplicates",
        "variant_analysis.json",
    )
    variants_parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for semantic variants (0.0-1.0)"
    )

    variants_parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines for variant consideration"
    )

    variants_parser.add_argument(
        "-e", "--exclude",
        nargs="*",
        help="Regex patterns for paths to exclude"
    )

    # Parse and dispatch
    args = parser.parse_args()

    try:
        if args.command == "analyze":
            analyze_command(args)
        elif args.command == "variants":
            variants_command(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# TODO Rename this here and in `main`
def _extracted_from_main_15(subparsers, arg1, help, arg3):
    # Analyze command
    result = subparsers.add_parser(arg1, help=help)

    result.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Root directory to analyze",
    )

    result.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(arg3),
        help="Output JSON file path",
    )

    return result

if __name__ == "__main__":
    main() 