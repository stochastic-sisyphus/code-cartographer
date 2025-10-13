"""
Code Cartographer CLI
====================
Command line interface for the code cartographer tool.
"""

import argparse
import json
import sys
from pathlib import Path

from code_cartographer.core.analyzer import ProjectAnalyzer
from code_cartographer.core.variant_analyzer import VariantAnalyzer


class SetEncoder(json.JSONEncoder):
    """JSON encoder that handles sets by converting them to lists."""

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def analyze_command(args: argparse.Namespace) -> None:
    """Run the main code analysis."""
    analyzer = ProjectAnalyzer(root=args.dir, exclude_patterns=args.exclude or [])

    analysis = analyzer.execute()

    # Write JSON output
    args.output.write_text(json.dumps(analysis, indent=2, cls=SetEncoder))
    print(f"[INFO] Analysis complete: {args.output.resolve()}")


def variants_command(args: argparse.Namespace) -> None:
    """Run variant analysis."""
    analyzer = VariantAnalyzer(
        root=args.dir,
        semantic_threshold=args.semantic_threshold,
        min_lines=args.min_lines,
        exclude_patterns=args.exclude,
    )

    analysis = analyzer.analyze()

    # Write analysis output
    args.output.write_text(json.dumps(analysis, indent=2, cls=SetEncoder))
    print(f"[INFO] Variant analysis complete: {args.output.resolve()}")

    # Apply merged variants if requested
    if args.apply_merges:
        print("[INFO] Applying merged variants...")
        analyzer.apply_merged_variants(backup=not args.no_backup)
        print("[INFO] Variants merged successfully")


def _setup_parser(subparsers, command, help_text, default_output):
    """Set up a subparser with common arguments."""
    parser = subparsers.add_parser(
        command, help=help_text, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Root directory to analyze",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(default_output),
        help="Output JSON file path",
    )

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Code Cartographer - Advanced Python Codebase Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # Analyze command
    analyze_parser = _setup_parser(
        subparsers, "analyze", "Run deep code analysis", "code_analysis.json"
    )
    analyze_parser.add_argument(
        "-e", "--exclude", nargs="*", help="Regex patterns for paths to exclude"
    )

    # Variants command
    variants_parser = _setup_parser(
        subparsers,
        "variants",
        "Analyze code variants and duplicates",
        "variant_analysis.json",
    )
    variants_parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for semantic variants (0.0-1.0)",
    )

    variants_parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines for variant consideration",
    )

    variants_parser.add_argument(
        "-e", "--exclude", nargs="*", help="Regex patterns for paths to exclude"
    )

    variants_parser.add_argument(
        "--apply-merges",
        action="store_true",
        help="Apply merged variants to codebase",
    )

    variants_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files when applying merges",
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


if __name__ == "__main__":
    main()
