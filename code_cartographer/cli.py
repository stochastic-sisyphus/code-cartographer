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
from code_cartographer.core.reporter import ReportGenerator


def analyze_command(args: argparse.Namespace) -> None:
    """Run the main code analysis."""
    # Create output directory
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Analyzing directory: {args.dir}")
    print(f"[INFO] Output directory: {output_dir}")

    # Run the analysis
    analyzer = ProjectAnalyzer(root=args.dir, exclude_patterns=args.exclude or [])
    analysis = analyzer.execute()

    # Write JSON output
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"[INFO] JSON analysis saved: {args.output.resolve()}")

    # Generate reports
    reporter = ReportGenerator(output_dir)

    # Generate Markdown report
    markdown_path = reporter.generate_markdown_report(analysis)
    print(f"[INFO] Markdown report: {markdown_path.resolve()}")

    # Generate HTML report
    html_path = reporter.generate_html_report(analysis, markdown_path)
    print(f"[INFO] HTML report: {html_path.resolve()}")

    print(f"[INFO] Analysis complete!")
    print(f"[INFO] Open the HTML report in your browser: {html_path.resolve()}")


def variants_command(args: argparse.Namespace) -> None:
    """Run variant analysis."""
    # Lazy import to avoid requiring all dependencies
    try:
        from code_cartographer.core.variant_analyzer import VariantAnalyzer
    except ImportError as e:
        print(
            f"Error: Variant analysis requires additional dependencies: {e}",
            file=sys.stderr,
        )
        print("Install them with: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Analyzing code variants in: {args.dir}")

    analyzer = VariantAnalyzer(
        root=args.dir,
        semantic_threshold=args.semantic_threshold,
        min_lines=args.min_lines,
        exclude_patterns=args.exclude or [],
    )

    analysis = analyzer.analyze()

    # Write analysis output
    with open(args.output, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"[INFO] Variant analysis saved: {args.output.resolve()}")

    # Apply merged variants if requested
    if args.apply_merges:
        print("[INFO] Applying merged variants...")
        analyzer.apply_merged_variants(backup=not args.no_backup)
        print("[INFO] Variants merged successfully")


def report_command(args: argparse.Namespace) -> None:
    """Generate a report from existing analysis JSON."""
    # Load the analysis data
    with open(args.input, "r") as f:
        analysis = json.load(f)

    # Create output directory
    output_dir = args.output.parent if args.output else args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    reporter = ReportGenerator(output_dir)

    # Generate Markdown report
    markdown_path = reporter.generate_markdown_report(analysis, args.output)
    print(f"[INFO] Markdown report: {markdown_path.resolve()}")

    # Generate HTML report
    html_path = reporter.generate_html_report(analysis, markdown_path)
    print(f"[INFO] HTML report: {html_path.resolve()}")


def visualize_command(args: argparse.Namespace) -> None:
    """Generate interactive visualizations from analysis JSON."""
    # Load the analysis data
    with open(args.input, "r") as f:
        analysis = json.load(f)

    # Create output directory
    output_dir = args.output.parent if args.output else args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    reporter = ReportGenerator(output_dir)

    # Generate interactive dashboard
    dashboard_path = reporter.generate_interactive_dashboard(
        analysis, output_path=args.output
    )
    print(f"[INFO] Interactive dashboard: {dashboard_path.resolve()}")
    print(f"[INFO] Open the dashboard in your browser: {dashboard_path.resolve()}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Code Cartographer - Advanced Python Codebase Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run deep code analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyze_parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path("."),
        help="Root directory to analyze (default: current directory)",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/code_analysis.json"),
        help="Output JSON file path (default: artifacts/code_analysis.json)",
    )
    analyze_parser.add_argument(
        "-e", "--exclude", nargs="*", help="Regex patterns for paths to exclude"
    )

    # Variants command
    variants_parser = subparsers.add_parser(
        "variants",
        help="Analyze code variants and duplicates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    variants_parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        default=Path("."),
        help="Root directory to analyze (default: current directory)",
    )
    variants_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("artifacts/variant_analysis.json"),
        help="Output JSON file path (default: artifacts/variant_analysis.json)",
    )
    variants_parser.add_argument(
        "-e", "--exclude", nargs="*", help="Regex patterns for paths to exclude"
    )
    variants_parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for semantic variants (0.0-1.0, default: 0.8)",
    )
    variants_parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines for variant consideration (default: 5)",
    )
    variants_parser.add_argument(
        "--apply-merges", action="store_true", help="Apply merged variants to codebase"
    )
    variants_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup files when applying merges",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate report from analysis JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    report_parser.add_argument("input", type=Path, help="Input JSON file from analysis")
    report_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output report path (default: same directory as input)",
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize",
        help="Generate interactive visualizations from analysis JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    visualize_parser.add_argument(
        "input", type=Path, help="Input JSON file from analysis"
    )
    visualize_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output dashboard HTML path (default: same directory as input)",
    )

    # Parse and dispatch
    args = parser.parse_args()

    try:
        if args.command == "analyze":
            analyze_command(args)
        elif args.command == "variants":
            variants_command(args)
        elif args.command == "report":
            report_command(args)
        elif args.command == "visualize":
            visualize_command(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
