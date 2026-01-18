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
from code_cartographer.core.immersive_dashboard import ImmersiveDashboardGenerator
from code_cartographer.core.temporal_analyzer import TemporalAnalyzer


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


def visualize_command(args: argparse.Namespace) -> None:
    """Generate immersive visualization dashboard."""
    print("[INFO] Running code analysis...")
    analyzer = ProjectAnalyzer(root=args.dir, exclude_patterns=args.exclude or [])
    analysis = analyzer.execute()
    
    temporal_data = None
    if args.temporal:
        print("[INFO] Analyzing temporal evolution...")
        temporal_analyzer = TemporalAnalyzer(args.dir)
        snapshots = temporal_analyzer.analyze_git_history(
            max_commits=args.max_commits,
            file_patterns=['*.py']
        )
        
        temporal_data = {
            'timeline': temporal_analyzer.get_evolution_timeline(),
            'velocity': temporal_analyzer.calculate_code_velocity(),
            'interactions': temporal_analyzer.get_interaction_patterns()
        }
    
    print("[INFO] Generating immersive dashboard...")
    dashboard_gen = ImmersiveDashboardGenerator()
    output_path = dashboard_gen.generate(analysis, args.output, temporal_data)
    
    print(f"[INFO] Dashboard generated: {output_path.resolve()}")
    print(f"[INFO] Open {output_path.resolve()} in your browser to explore!")
    
    # Also export JSON data
    if args.export_json:
        json_path = args.output.with_suffix('.json')
        dashboard_gen.export_json_data(analysis, json_path, temporal_data)
        print(f"[INFO] Visualization data exported: {json_path.resolve()}")


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

    # Visualize command
    visualize_parser = _setup_parser(
        subparsers,
        "visualize",
        "Generate immersive interactive visualization",
        "codebase_visualization.html",
    )
    visualize_parser.add_argument(
        "-e", "--exclude", nargs="*", help="Regex patterns for paths to exclude"
    )
    visualize_parser.add_argument(
        "--temporal",
        action="store_true",
        help="Include temporal evolution analysis from git history",
    )
    visualize_parser.add_argument(
        "--max-commits",
        type=int,
        default=100,
        help="Maximum number of commits to analyze for temporal view (default: 100)",
    )
    visualize_parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export visualization data as JSON",
    )

    # Parse and dispatch
    args = parser.parse_args()

    try:
        if args.command == "analyze":
            analyze_command(args)
        elif args.command == "variants":
            variants_command(args)
        elif args.command == "visualize":
            visualize_command(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
