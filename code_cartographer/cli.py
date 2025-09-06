"""
Code Cartographer CLI
====================
Command line interface for the code cartographer tool.
"""

import argparse
import json
import sys
from pathlib import Path

from code_cartographer.core.analyzer import CodeAnalyzer, ProjectAnalyzer
from code_cartographer.core.visualizer import CodeVisualizer
from code_cartographer.core.reporter import ReportGenerator
from code_cartographer.core.variant_analyzer import VariantAnalyzer


def analyze_command(args: argparse.Namespace) -> None:
    """Run the main code analysis."""
    # Create output directory if it doesn't exist
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the analyzer
    analyzer = CodeAnalyzer(
        project_root=args.dir,
        output_dir=output_dir
    )
    
    # Run the analysis
    analysis = analyzer.analyze(exclude_patterns=args.exclude or [])
    
    # Generate report
    if args.markdown:
        report_path = analyzer.generate_report(analysis)
        print(f"[INFO] Analysis report generated: {report_path}")
    
    # Generate visualizations
    if args.graphviz:
        visualizer = CodeVisualizer(output_dir)
        
        # Generate function call graph using the correct method name
        if "call_graph" in analysis and isinstance(analysis["call_graph"], dict):
            graph_path = visualizer.generate_function_call_graph(analysis["call_graph"])
            print(f"[INFO] Function call graph generated: {graph_path}")
        
        # Generate class hierarchy if available
        if "classes" in analysis and isinstance(analysis["classes"], dict):
            class_path = visualizer.generate_class_hierarchy(analysis["classes"])
            print(f"[INFO] Class hierarchy generated: {class_path}")
        
        # Generate variable usage chart if available
        if "variables" in analysis and isinstance(analysis["variables"], dict):
            # Convert the nested structure to a flat dictionary for visualization
            variable_data = {}
            for var_name, var_instances in analysis["variables"].items():
                if var_instances and isinstance(var_instances, list):
                    # Take the first instance for simplicity
                    var_data = var_instances[0]
                    if isinstance(var_data, dict):
                        variable_data[var_name] = {
                            "definition_count": len(var_instances),
                            "usage_count": len(var_data.get("used_in", [])),
                            "is_orphan": var_data.get("is_orphan", True),
                            "is_redefined": len(var_instances) > 1
                        }
            
            if variable_data:
                var_path = visualizer.generate_variable_usage_chart(variable_data)
                print(f"[INFO] Variable usage chart generated: {var_path}")
        
        # Generate orphan analysis if available
        if "orphans" in analysis:
            # Convert to expected format
            orphan_data = {}
            
            # Extract functions, classes, and variables from the orphans data
            if isinstance(analysis["orphans"], dict):
                orphan_data = analysis["orphans"]
            else:
                # Try to build orphan data from other parts of the analysis
                orphan_data = {
                    "functions": [],
                    "classes": [],
                    "variables": []
                }
                
                # Extract orphaned functions and classes from definitions
                for file_data in analysis.get("files", []):
                    if isinstance(file_data, dict) and "definitions" in file_data:
                        for defn in file_data.get("definitions", []):
                            if isinstance(defn, dict) and defn.get("is_orphan", False):
                                if defn.get("category") == "function":
                                    orphan_data["functions"].append(defn)
                                elif defn.get("category") == "class":
                                    orphan_data["classes"].append(defn)
                
                # Extract orphaned variables
                for var_name, var_instances in analysis.get("variables", {}).items():
                    for var_data in var_instances:
                        if isinstance(var_data, dict) and var_data.get("is_orphan", False):
                            orphan_data["variables"].append(var_data)
            
            if orphan_data:
                orphan_path = visualizer.generate_orphan_analysis(orphan_data)
                print(f"[INFO] Orphan analysis chart generated: {orphan_path}")
        
        # Generate prerequisite graph if available
        if "prerequisites" in analysis and isinstance(analysis["prerequisites"], dict):
            prereq_path = visualizer.generate_prerequisite_graph(analysis["prerequisites"])
            print(f"[INFO] Prerequisite graph generated: {prereq_path}")
        elif "dependencies" in analysis:
            # Try to convert dependencies to prerequisites format
            prerequisites = {}
            for dep in analysis.get("dependencies", []):
                if isinstance(dep, (list, tuple)) and len(dep) == 2:
                    source, target = dep
                    if source not in prerequisites:
                        prerequisites[source] = set()
                    prerequisites[source].add(target)
            
            if prerequisites:
                prereq_path = visualizer.generate_prerequisite_graph(prerequisites)
                print(f"[INFO] Prerequisite graph generated: {prereq_path}")
        
        # Generate initialization sequence if available
        if "initialization_sequence" in analysis and isinstance(analysis["initialization_sequence"], list):
            init_path = visualizer.generate_initialization_sequence(analysis["initialization_sequence"])
            print(f"[INFO] Initialization sequence generated: {init_path}")
    
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
    args.output.write_text(json.dumps(analysis, indent=2))
    print(f"[INFO] Variant analysis complete: {args.output.resolve()}")

    # Apply merged variants if requested
    if args.apply_merges:
        print("[INFO] Applying merged variants...")
        analyzer.apply_merged_variants(backup=not args.no_backup)
        print("[INFO] Variants merged successfully")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Code Cartographer - Advanced Python Codebase Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    analyze_parser = _setup_parser(
        subparsers, "analyze", "Run deep code analysis", "analysis_output/code_analysis.json"
    )
    analyze_parser.add_argument(
        "-e", "--exclude", nargs="*", help="Regex patterns for paths to exclude"
    )

    analyze_parser.add_argument(
        "--markdown", 
        action="store_true",
        default=True,
        help="Generate Markdown report"
    )

    analyze_parser.add_argument(
        "--graphviz",
        action="store_true",
        default=True,
        help="Generate visualization graphs",
    )

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


if __name__ == "__main__":
    main()
