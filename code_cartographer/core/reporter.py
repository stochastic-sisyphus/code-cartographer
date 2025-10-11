"""
Enhanced Reporting Module
=======================
Generates comprehensive reports with advanced analysis and visualization results.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import markdown
from jinja2 import Environment, FileSystemLoader


class ReportGenerator:
    """Generates comprehensive reports from code analysis results."""

    def __init__(self, output_dir: Path):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save report outputs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create templates directory if it doesn't exist
        self.templates_dir = self.output_dir / "templates"
        self.templates_dir.mkdir(exist_ok=True)

    def generate_markdown_report(
        self, analysis_data: Dict, output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a comprehensive Markdown report from analysis data.

        Args:
            analysis_data: Complete analysis data dictionary
            output_path: Path to save the Markdown report (optional)

        Returns:
            Path to the saved report
        """
        if output_path is None:
            output_path = self.output_dir / "code_analysis_report.md"

        # Start building the report
        report_lines = ["# Code Analysis Report\n", "## Overview\n"]

        # Add summary statistics
        files_count = len(analysis_data.get("files", []))
        definitions_count = sum(
            len(f.get("definitions", [])) for f in analysis_data.get("files", [])
        )

        # Handle orphans - can be dict or list
        orphans = analysis_data.get("orphans", [])
        if isinstance(orphans, dict):
            orphan_functions = len(orphans.get("functions", []))
            orphan_classes = len(orphans.get("classes", []))
            orphan_variables = len(orphans.get("variables", []))
        else:
            # It's a list of orphan names
            orphan_functions = len(
                [o for o in orphans if not o.startswith("__") and "." not in o]
            )
            orphan_classes = 0
            orphan_variables = 0

        report_lines.extend(
            [
                f"- **Total Files**: {files_count}",
                f"- **Total Definitions**: {definitions_count}",
                f"- **Orphaned Code Elements**: {len(orphans) if isinstance(orphans, list) else orphan_functions + orphan_classes + orphan_variables}",
                "\n",
            ]
        )

        # Add call graph information
        report_lines.extend(
            [
                "## Call Graph Analysis\n",
                "The following analysis shows the relationships between functions and methods in the codebase.\n",
            ]
        )

        # Add bidirectional call information
        if "call_graph" in analysis_data and "reverse_call_graph" in analysis_data:
            report_lines.append("### Function Call Relationships\n")

            # Find the most called functions
            most_called = []
            for func, callers in analysis_data.get("reverse_call_graph", {}).items():
                most_called.append((func, len(callers)))

            most_called.sort(key=lambda x: x[1], reverse=True)

            report_lines.append("#### Most Called Functions\n")
            for func, count in most_called[:10]:  # Top 10
                report_lines.append(f"- **{func}**: Called by {count} functions")

            report_lines.append("\n#### Functions with Most Dependencies\n")

            # Find functions with most dependencies
            most_deps = []
            for func, callees in analysis_data.get("call_graph", {}).items():
                most_deps.append((func, len(callees)))

            most_deps.sort(key=lambda x: x[1], reverse=True)

            for func, count in most_deps[:10]:  # Top 10
                report_lines.append(f"- **{func}**: Calls {count} other functions")

            report_lines.append("\n")

        # Add orphan analysis
        report_lines.extend(
            [
                "## Orphaned Code Analysis\n",
                "The following code elements are defined but never used in the codebase.\n",
            ]
        )

        if "orphans" in analysis_data:
            orphans = analysis_data["orphans"]

            if isinstance(orphans, dict):
                # Dictionary format with functions, classes, variables
                report_lines.append("### Orphaned Functions\n")
                for func in orphans.get("functions", []):
                    report_lines.append(
                        f"- **{func['name']}** in {func['file']}:{func['line']}"
                    )

                report_lines.append("\n### Orphaned Classes\n")
                for cls in orphans.get("classes", []):
                    report_lines.append(
                        f"- **{cls['name']}** in {cls['file']}:{cls['line']}"
                    )

                report_lines.append("\n### Orphaned Variables\n")
                for var in orphans.get("variables", [])[
                    :20
                ]:  # Limit to 20 to avoid excessive length
                    report_lines.append(
                        f"- **{var['name']}** in {var['file']}:{var['line']}"
                    )

                if len(orphans.get("variables", [])) > 20:
                    report_lines.append(
                        f"- ... and {len(orphans['variables']) - 20} more"
                    )
            else:
                # List format with orphan names
                report_lines.append("### Orphaned Code Elements\n")
                for orphan in orphans[:50]:  # Limit to 50
                    report_lines.append(f"- **{orphan}**")
                if len(orphans) > 50:
                    report_lines.append(f"- ... and {len(orphans) - 50} more")

            report_lines.append("\n")

        # Add variable usage analysis
        if "variables" in analysis_data:
            report_lines.extend(
                [
                    "## Variable Usage Analysis\n",
                    "This section shows how variables are defined and used across the codebase.\n",
                ]
            )

            # Handle list-based variable data
            variables = analysis_data["variables"]
            if variables and isinstance(list(variables.values())[0], list):
                # Find variables with multiple definitions
                multi_defined = []
                for var_name, var_list in variables.items():
                    if len(var_list) > 1:
                        multi_defined.append((var_name, len(var_list)))

                if multi_defined:
                    report_lines.append("### Variables with Multiple Definitions\n")
                    for var_name, count in sorted(
                        multi_defined, key=lambda x: x[1], reverse=True
                    )[:15]:
                        report_lines.append(f"- **{var_name}**: Defined {count} times")
            else:
                # Handle dict-based variable data (older format)
                multi_defined = []
                for var_name, var_data in variables.items():
                    if (
                        isinstance(var_data, dict)
                        and var_data.get("definition_count", 0) > 1
                    ):
                        multi_defined.append((var_name, var_data))

                if multi_defined:
                    report_lines.append("### Variables with Multiple Definitions\n")
                    for var_name, var_data in sorted(
                        multi_defined,
                        key=lambda x: x[1].get("definition_count", 0),
                        reverse=True,
                    )[:15]:
                        def_count = var_data.get("definition_count", 0)
                        use_count = var_data.get("usage_count", 0)
                        report_lines.append(
                            f"- **{var_name}**: Defined {def_count} times, used {use_count} times"
                        )

            report_lines.append("\n")

        # Add prerequisite and dependency analysis
        if "dependency_levels" in analysis_data:
            report_lines.extend(
                [
                    "## Prerequisite and Dependency Analysis\n",
                    "This section shows the initialization order and dependencies between code elements.\n",
                ]
            )

            # Show initialization order
            if "initialization_order" in analysis_data:
                report_lines.append("### Recommended Initialization Order\n")
                for i, element in enumerate(analysis_data["initialization_order"][:20]):
                    report_lines.append(f"{i+1}. **{element}**")

                if len(analysis_data["initialization_order"]) > 20:
                    report_lines.append(
                        f"... and {len(analysis_data['initialization_order']) - 20} more"
                    )

                report_lines.append("\n")

            # Show dependency cycles if any
            if "cycles" in analysis_data and analysis_data["cycles"]:
                report_lines.append("### Dependency Cycles\n")
                report_lines.append(
                    "The following cycles in dependencies were detected:\n"
                )

                for i, cycle in enumerate(analysis_data["cycles"][:10]):
                    report_lines.append(
                        f"Cycle {i+1}: " + " → ".join(cycle) + " → " + cycle[0]
                    )

                if len(analysis_data["cycles"]) > 10:
                    report_lines.append(
                        f"... and {len(analysis_data['cycles']) - 10} more cycles"
                    )

                report_lines.append("\n")

        # Add file-by-file analysis
        report_lines.extend(
            [
                "## File Analysis\n",
                "This section provides detailed analysis for each file in the codebase.\n",
            ]
        )

        for file_data in analysis_data.get("files", []):
            file_path = file_data.get("path", "unknown")
            report_lines.append(f"### {file_path}\n")

            # Add file metrics
            metrics = file_data.get("metrics", {})
            mi = metrics.get("maintainability_index")
            cc = metrics.get("cyclomatic")

            if mi is not None:
                report_lines.append(f"- **Maintainability Index**: {mi:.2f}")
            if cc is not None:
                report_lines.append(f"- **Cyclomatic Complexity**: {cc}")

            # Add definitions
            report_lines.append("\n#### Definitions\n")

            for definition in file_data.get("definitions", []):
                name = definition.get("name", "unknown")
                category = definition.get("category", "unknown")
                line_count = definition.get("line_count", 0)

                # Add indicators for risk and orphan status
                indicators = []
                if definition.get("metrics", {}).get("risk_flag", False):
                    indicators.append("⚠️ High Risk")
                if definition.get("is_orphan", False):
                    indicators.append("🔕 Orphan")

                indicator_str = f" ({', '.join(indicators)})" if indicators else ""

                report_lines.append(
                    f"- **{name}** ({category}, {line_count} lines){indicator_str}"
                )

                # Add inbound calls
                if definition.get("inbound_calls"):
                    callers = list(definition.get("inbound_calls", []))
                    if len(callers) <= 5:
                        report_lines.append(f"  - Called by: {', '.join(callers)}")
                    else:
                        report_lines.append(
                            f"  - Called by: {', '.join(callers[:5])} and {len(callers) - 5} more"
                        )

                # Add prerequisites
                if definition.get("prerequisites"):
                    prereqs = list(definition.get("prerequisites", []))
                    if len(prereqs) <= 5:
                        report_lines.append(f"  - Prerequisites: {', '.join(prereqs)}")
                    else:
                        report_lines.append(
                            f"  - Prerequisites: {', '.join(prereqs[:5])} and {len(prereqs) - 5} more"
                        )

            # Add orphaned code section if any
            if file_data.get("orphaned_code"):
                report_lines.append("\n#### Orphaned Code\n")
                report_lines.append("The following definitions are never called:\n")

                for orphan in file_data.get("orphaned_code", []):
                    report_lines.append(f"- {orphan}")

            report_lines.append("\n")

        # Add variant analysis
        if "variants" in analysis_data and analysis_data["variants"]:
            report_lines.extend(
                [
                    "## Code Variant Analysis\n",
                    "This section shows similar implementations that could be consolidated.\n",
                ]
            )

            for name, variant_data in analysis_data["variants"].items():
                implementations = variant_data.get("implementations", [])
                if len(implementations) <= 1:
                    continue

                report_lines.append(f"### `{name}`\n")
                report_lines.append(
                    f"Found {len(implementations)} similar implementations:\n"
                )

                for impl in implementations:
                    report_lines.append(f"- In file: {impl.get('path', 'unknown')}")

                report_lines.append(
                    "\n**Recommendation**: "
                    + variant_data.get(
                        "refactor_prompt", "Consider refactoring these implementations."
                    ).split("\n")[0]
                )
                report_lines.append("\n")

        # Write the report to file
        with open(output_path, "w") as f:
            f.write("\n".join(report_lines))

        return output_path

    def generate_html_report(
        self,
        analysis_data: Dict,
        markdown_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate an HTML report from the Markdown report.

        Args:
            analysis_data: Complete analysis data dictionary
            markdown_path: Path to the Markdown report
            output_path: Path to save the HTML report (optional)

        Returns:
            Path to the saved report
        """
        if output_path is None:
            output_path = self.output_dir / "code_analysis_report.html"

        # Read the Markdown content
        with open(markdown_path, "r") as f:
            markdown_content = f.read()

        # Convert Markdown to HTML
        html_content = markdown.markdown(
            markdown_content, extensions=["tables", "fenced_code"]
        )

        # Create a simple HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Code Analysis Report</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                    margin-top: 1.5em;
                }}
                h1 {{
                    border-bottom: 2px solid #eaecef;
                    padding-bottom: 0.3em;
                }}
                h2 {{
                    border-bottom: 1px solid #eaecef;
                    padding-bottom: 0.3em;
                }}
                pre {{
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    padding: 16px;
                    overflow: auto;
                }}
                code {{
                    background-color: #f6f8fa;
                    border-radius: 3px;
                    padding: 0.2em 0.4em;
                    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 1em;
                }}
                th, td {{
                    border: 1px solid #dfe2e5;
                    padding: 8px 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f6f8fa;
                }}
                .risk {{
                    color: #e74c3c;
                }}
                .orphan {{
                    color: #f39c12;
                }}
                .toc {{
                    background-color: #f8f9fa;
                    border: 1px solid #eaecef;
                    border-radius: 3px;
                    padding: 1em;
                    margin-bottom: 1em;
                }}
                .toc ul {{
                    list-style-type: none;
                    padding-left: 1em;
                }}
                .toc li {{
                    margin: 0.5em 0;
                }}
                .toc a {{
                    text-decoration: none;
                    color: #0366d6;
                }}
                .toc a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="toc">
                <h2>Table of Contents</h2>
                <ul>
                    <li><a href="#overview">Overview</a></li>
                    <li><a href="#call-graph-analysis">Call Graph Analysis</a></li>
                    <li><a href="#orphaned-code-analysis">Orphaned Code Analysis</a></li>
                    <li><a href="#variable-usage-analysis">Variable Usage Analysis</a></li>
                    <li><a href="#prerequisite-and-dependency-analysis">Prerequisite and Dependency Analysis</a></li>
                    <li><a href="#file-analysis">File Analysis</a></li>
                    <li><a href="#code-variant-analysis">Code Variant Analysis</a></li>
                </ul>
            </div>
            {html_content}
        </body>
        </html>
        """

        # Write the HTML to file
        with open(output_path, "w") as f:
            f.write(html_template)

        return output_path

    def generate_interactive_dashboard(
        self,
        analysis_data: Dict,
        template_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate an interactive HTML dashboard for code analysis results.

        Args:
            analysis_data: Complete analysis data dictionary
            template_path: Path to the Jinja2 template for the dashboard (optional)
            output_path: Path to save the dashboard HTML (optional)

        Returns:
            Path to the saved dashboard
        """
        if output_path is None:
            output_path = self.output_dir / "dashboard.html"

        # Use default template if not provided
        if template_path is None:
            template_path = self._create_default_template()

        import datetime

        # Prepare data for the dashboard
        dashboard_data = {
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": self._prepare_metrics(analysis_data),
            "variant_groups": self._prepare_variant_groups(analysis_data),
            "complexity_data": self._prepare_complexity_data(analysis_data),
            "dependency_data": self._prepare_dependency_data(analysis_data),
            "documentation_data": self._prepare_documentation_data(analysis_data),
            "orphan_data": self._prepare_orphan_data(analysis_data),
            "variable_data": self._prepare_variable_data(analysis_data),
            "prerequisite_data": self._prepare_prerequisite_data(analysis_data),
        }

        # Set up Jinja environment
        env = Environment(loader=FileSystemLoader(template_path.parent))
        template = env.get_template(template_path.name)

        # Render the template
        html_content = template.render(**dashboard_data)

        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _create_default_template(self) -> Path:
        """Create a default dashboard template if none is provided."""
        template_path = self.templates_dir / "dashboard.html.j2"

        # Check if template already exists
        if template_path.exists():
            return template_path

        # Create a basic template
        template_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Code Analysis Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    padding-top: 2rem;
                }
                .card {
                    margin-bottom: 1.5rem;
                }
                .metric-card {
                    text-align: center;
                    padding: 1.5rem;
                }
                .metric-value {
                    font-size: 2.5rem;
                    font-weight: bold;
                    color: #0d6efd;
                }
                .metric-label {
                    font-size: 1rem;
                    color: #6c757d;
                }
                .nav-tabs {
                    margin-bottom: 1rem;
                }
                .tab-pane {
                    padding: 1rem;
                }
                .code-block {
                    font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
                    background-color: #f8f9fa;
                    padding: 1rem;
                    border-radius: 0.25rem;
                    white-space: pre-wrap;
                }
                .risk-high {
                    color: #dc3545;
                }
                .risk-medium {
                    color: #fd7e14;
                }
                .risk-low {
                    color: #198754;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mb-4">Code Analysis Dashboard</h1>
                <p class="text-muted">Generated on {{ generation_time }}</p>
                
                <!-- Metrics Overview -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="metric-value">{{ metrics.total_files }}</div>
                            <div class="metric-label">Total Files</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="metric-value">{{ metrics.total_definitions }}</div>
                            <div class="metric-label">Total Definitions</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="metric-value">{{ metrics.avg_complexity|round(1) }}</div>
                            <div class="metric-label">Avg. Complexity</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card metric-card">
                            <div class="metric-value">{{ (metrics.doc_coverage * 100)|round }}%</div>
                            <div class="metric-label">Documentation</div>
                        </div>
                    </div>
                </div>
                
                <!-- Main Tabs -->
                <ul class="nav nav-tabs" id="mainTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab">Overview</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="complexity-tab" data-bs-toggle="tab" data-bs-target="#complexity" type="button" role="tab">Complexity</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="dependencies-tab" data-bs-toggle="tab" data-bs-target="#dependencies" type="button" role="tab">Dependencies</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="orphans-tab" data-bs-toggle="tab" data-bs-target="#orphans" type="button" role="tab">Orphans</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="variables-tab" data-bs-toggle="tab" data-bs-target="#variables" type="button" role="tab">Variables</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="variants-tab" data-bs-toggle="tab" data-bs-target="#variants" type="button" role="tab">Variants</button>
                    </li>
                </ul>
                
                <!-- Tab Content -->
                <div class="tab-content" id="mainTabsContent">
                    <!-- Overview Tab -->
                    <div class="tab-pane fade show active" id="overview" role="tabpanel">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Project Summary</h5>
                                    </div>
                                    <div class="card-body">
                                        <canvas id="summaryChart"></canvas>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Key Metrics</h5>
                                    </div>
                                    <div class="card-body">
                                        <table class="table">
                                            <tbody>
                                                <tr>
                                                    <th>Total Files</th>
                                                    <td>{{ metrics.total_files }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Total Definitions</th>
                                                    <td>{{ metrics.total_definitions }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Functions</th>
                                                    <td>{{ metrics.function_count }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Classes</th>
                                                    <td>{{ metrics.class_count }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Methods</th>
                                                    <td>{{ metrics.method_count }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Orphaned Elements</th>
                                                    <td>{{ metrics.orphan_count }}</td>
                                                </tr>
                                                <tr>
                                                    <th>Code Variants</th>
                                                    <td>{{ metrics.variant_count }}</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Complexity Tab -->
                    <div class="tab-pane fade" id="complexity" role="tabpanel">
                        <div class="row">
                            <div class="col-md-8">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Complexity Distribution</h5>
                                    </div>
                                    <div class="card-body">
                                        <canvas id="complexityChart"></canvas>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">High Complexity Functions</h5>
                                    </div>
                                    <div class="card-body">
                                        <ul class="list-group">
                                            {% for func in complexity_data.high_complexity %}
                                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                                {{ func.name }}
                                                <span class="badge bg-danger rounded-pill">{{ func.complexity }}</span>
                                            </li>
                                            {% endfor %}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Dependencies Tab -->
                    <div class="tab-pane fade" id="dependencies" role="tabpanel">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Most Referenced Modules</h5>
                                    </div>
                                    <div class="card-body">
                                        <table class="table">
                                            <thead>
                                                <tr>
                                                    <th>Module</th>
                                                    <th>References</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {% for module in dependency_data.most_referenced %}
                                                <tr>
                                                    <td>{{ module.name }}</td>
                                                    <td>{{ module.references }}</td>
                                                </tr>
                                                {% endfor %}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Initialization Order</h5>
                                    </div>
                                    <div class="card-body">
                                        <ol class="list-group list-group-numbered">
                                            {% for item in prerequisite_data.initialization_order[:10] %}
                                            <li class="list-group-item">{{ item }}</li>
                                            {% endfor %}
                                            {% if prerequisite_data.initialization_order|length > 10 %}
                                            <li class="list-group-item text-muted">... and {{ prerequisite_data.initialization_order|length - 10 }} more</li>
                                            {% endif %}
                                        </ol>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-md-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Dependency Cycles</h5>
                                    </div>
                                    <div class="card-body">
                                        {% if prerequisite_data.cycles %}
                                        <div class="alert alert-warning">
                                            <strong>Warning:</strong> Dependency cycles detected!
                                        </div>
                                        <ul class="list-group">
                                            {% for cycle in prerequisite_data.cycles %}
                                            <li class="list-group-item">{{ " → ".join(cycle) }} → {{ cycle[0] }}</li>
                                            {% endfor %}
                                        </ul>
                                        {% else %}
                                        <div class="alert alert-success">
                                            <strong>Good news!</strong> No dependency cycles detected.
                                        </div>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Orphans Tab -->
                    <div class="tab-pane fade" id="orphans" role="tabpanel">
                        <div class="row">
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Orphan Distribution</h5>
                                    </div>
                                    <div class="card-body">
                                        <canvas id="orphanChart"></canvas>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-8">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Orphaned Functions</h5>
                                    </div>
                                    <div class="card-body">
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    <th>Function</th>
                                                    <th>File</th>
                                                    <th>Line</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {% for func in orphan_data.functions %}
                                                <tr>
                                                    <td>{{ func.name }}</td>
                                                    <td>{{ func.file }}</td>
                                                    <td>{{ func.line }}</td>
                                                </tr>
                                                {% endfor %}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Variables Tab -->
                    <div class="tab-pane fade" id="variables" role="tabpanel">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Variable Usage</h5>
                                    </div>
                                    <div class="card-body">
                                        <canvas id="variableChart"></canvas>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Multi-defined Variables</h5>
                                    </div>
                                    <div class="card-body">
                                        <table class="table">
                                            <thead>
                                                <tr>
                                                    <th>Variable</th>
                                                    <th>Definitions</th>
                                                    <th>Usages</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {% for var in variable_data.multi_defined %}
                                                <tr>
                                                    <td>{{ var.name }}</td>
                                                    <td>{{ var.definition_count }}</td>
                                                    <td>{{ var.usage_count }}</td>
                                                </tr>
                                                {% endfor %}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Variants Tab -->
                    <div class="tab-pane fade" id="variants" role="tabpanel">
                        <div class="row">
                            <div class="col-md-12">
                                <div class="card">
                                    <div class="card-header">
                                        <h5 class="card-title">Code Variants</h5>
                                    </div>
                                    <div class="card-body">
                                        <div class="accordion" id="variantsAccordion">
                                            {% for group in variant_groups %}
                                            <div class="accordion-item">
                                                <h2 class="accordion-header" id="heading{{ loop.index }}">
                                                    <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse{{ loop.index }}">
                                                        {{ group.name }} ({{ group.variants|length + 1 }} implementations)
                                                    </button>
                                                </h2>
                                                <div id="collapse{{ loop.index }}" class="accordion-collapse collapse" data-bs-parent="#variantsAccordion">
                                                    <div class="accordion-body">
                                                        <h6>Base Implementation:</h6>
                                                        <div class="code-block mb-3">{{ group.base_code }}</div>
                                                        
                                                        <h6>Variants:</h6>
                                                        {% for variant in group.variants %}
                                                        <div class="card mb-2">
                                                            <div class="card-header">
                                                                {{ variant.path }} ({{ (variant.similarity * 100)|round }}% similar)
                                                            </div>
                                                            <div class="card-body">
                                                                <pre class="code-block">{% for line in variant.diff %}{{ line }}
{% endfor %}</pre>
                                                            </div>
                                                        </div>
                                                        {% endfor %}
                                                    </div>
                                                </div>
                                            </div>
                                            {% endfor %}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                // Initialize charts when DOM is loaded
                document.addEventListener('DOMContentLoaded', function() {
                    // Summary Chart
                    const summaryCtx = document.getElementById('summaryChart').getContext('2d');
                    new Chart(summaryCtx, {
                        type: 'bar',
                        data: {
                            labels: ['Functions', 'Classes', 'Methods', 'Orphans', 'Variants'],
                            datasets: [{
                                label: 'Count',
                                data: [
                                    {{ metrics.function_count }},
                                    {{ metrics.class_count }},
                                    {{ metrics.method_count }},
                                    {{ metrics.orphan_count }},
                                    {{ metrics.variant_count }}
                                ],
                                backgroundColor: [
                                    'rgba(54, 162, 235, 0.7)',
                                    'rgba(255, 99, 132, 0.7)',
                                    'rgba(255, 206, 86, 0.7)',
                                    'rgba(75, 192, 192, 0.7)',
                                    'rgba(153, 102, 255, 0.7)'
                                ]
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: {
                                    display: false
                                }
                            }
                        }
                    });
                    
                    // Complexity Chart
                    const complexityCtx = document.getElementById('complexityChart').getContext('2d');
                    new Chart(complexityCtx, {
                        type: 'scatter',
                        data: {{ complexity_data.chart_data|tojson }},
                        options: {
                            responsive: true,
                            plugins: {
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            return `${context.raw.name}: ${context.raw.y} complexity, ${context.raw.x} lines`;
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Lines of Code'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Cyclomatic Complexity'
                                    }
                                }
                            }
                        }
                    });
                    
                    // Orphan Chart
                    const orphanCtx = document.getElementById('orphanChart').getContext('2d');
                    new Chart(orphanCtx, {
                        type: 'pie',
                        data: {
                            labels: ['Functions', 'Classes', 'Variables'],
                            datasets: [{
                                data: [
                                    {{ orphan_data.functions|length }},
                                    {{ orphan_data.classes|length }},
                                    {{ orphan_data.variables|length }}
                                ],
                                backgroundColor: [
                                    'rgba(255, 99, 132, 0.7)',
                                    'rgba(54, 162, 235, 0.7)',
                                    'rgba(255, 206, 86, 0.7)'
                                ]
                            }]
                        },
                        options: {
                            responsive: true
                        }
                    });
                    
                    // Variable Chart
                    const variableCtx = document.getElementById('variableChart').getContext('2d');
                    new Chart(variableCtx, {
                        type: 'bar',
                        data: {
                            labels: {{ variable_data.chart_labels|tojson }},
                            datasets: [
                                {
                                    label: 'Definitions',
                                    data: {{ variable_data.definition_counts|tojson }},
                                    backgroundColor: 'rgba(54, 162, 235, 0.7)'
                                },
                                {
                                    label: 'Usages',
                                    data: {{ variable_data.usage_counts|tojson }},
                                    backgroundColor: 'rgba(75, 192, 192, 0.7)'
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                x: {
                                    stacked: false
                                },
                                y: {
                                    stacked: false
                                }
                            }
                        }
                    });
                });
            </script>
        </body>
        </html>
        """

        # Write the template to file
        with open(template_path, "w") as f:
            f.write(template_content)

        return template_path

    def _prepare_metrics(self, analysis_data: Dict) -> Dict:
        """Prepare metrics data for the dashboard."""
        # Count definitions by category
        function_count = 0
        class_count = 0
        method_count = 0

        for file_data in analysis_data.get("files", []):
            for definition in file_data.get("definitions", []):
                category = definition.get("category", "")
                if category == "function":
                    function_count += 1
                elif category == "class":
                    class_count += 1
                elif category == "method":
                    method_count += 1

        # Calculate average complexity
        total_complexity = 0
        complexity_count = 0

        for file_data in analysis_data.get("files", []):
            for definition in file_data.get("definitions", []):
                if definition.get("metrics", {}).get("cyclomatic") is not None:
                    total_complexity += definition["metrics"]["cyclomatic"]
                    complexity_count += 1

        avg_complexity = (
            total_complexity / complexity_count if complexity_count > 0 else 0
        )

        # Calculate documentation coverage
        total_defs = 0
        documented_defs = 0

        for file_data in analysis_data.get("files", []):
            for definition in file_data.get("definitions", []):
                total_defs += 1
                if definition.get("docstring"):
                    documented_defs += 1

        doc_coverage = documented_defs / total_defs if total_defs > 0 else 0

        # Count orphans
        orphan_count = (
            len(analysis_data.get("orphans", {}).get("functions", []))
            + len(analysis_data.get("orphans", {}).get("classes", []))
            + len(analysis_data.get("orphans", {}).get("variables", []))
        )

        # Count variants
        variant_count = len(analysis_data.get("variants", {}))

        return {
            "total_files": len(analysis_data.get("files", [])),
            "total_definitions": function_count + class_count + method_count,
            "function_count": function_count,
            "class_count": class_count,
            "method_count": method_count,
            "avg_complexity": avg_complexity,
            "doc_coverage": doc_coverage,
            "orphan_count": orphan_count,
            "variant_count": variant_count,
        }

    def _prepare_variant_groups(self, analysis_data: Dict) -> List[Dict]:
        """Prepare variant groups data for the dashboard."""
        groups = []

        for name, data in analysis_data.get("variants", {}).items():
            implementations = data.get("implementations", [])
            if len(implementations) <= 1:
                continue

            base_impl = implementations[0]
            base_code = base_impl.get("source_text", "# No source available")

            variants = []
            for impl in implementations[1:]:
                variants.append(
                    {
                        "path": impl.get("path", "unknown"),
                        "similarity": 0.8,  # Placeholder, would be calculated in real implementation
                        "diff": impl.get("diff_from_base", []),
                    }
                )

            groups.append({"name": name, "base_code": base_code, "variants": variants})

        return groups

    def _prepare_complexity_data(self, analysis_data: Dict) -> Dict:
        """Prepare complexity data for the dashboard."""
        high_complexity = []
        chart_data = {"datasets": [{"label": "Functions", "data": []}]}

        for file_data in analysis_data.get("files", []):
            file_name = file_data.get("path", "unknown")

            for definition in file_data.get("definitions", []):
                name = definition.get("name", "unknown")
                complexity = definition.get("metrics", {}).get("cyclomatic")
                line_count = definition.get("line_count", 0)

                if complexity and complexity > 10:
                    high_complexity.append(
                        {
                            "name": name,
                            "file": file_name,
                            "complexity": complexity,
                            "lines": line_count,
                        }
                    )

                if complexity is not None:
                    chart_data["datasets"][0]["data"].append(
                        {"x": line_count, "y": complexity, "name": name}
                    )

        return {
            "high_complexity": sorted(
                high_complexity, key=lambda x: x["complexity"], reverse=True
            )[:10],
            "chart_data": chart_data,
        }

    def _prepare_dependency_data(self, analysis_data: Dict) -> Dict:
        """Prepare dependency data for the dashboard."""
        # Count references to each module
        references = {}
        for src, dst in analysis_data.get("dependencies", []):
            if dst not in references:
                references[dst] = 0
            references[dst] += 1

        most_referenced = [
            {"name": module, "references": count}
            for module, count in sorted(
                references.items(), key=lambda x: x[1], reverse=True
            )
        ][:10]

        # Count external dependencies
        external_deps = {}
        for file_data in analysis_data.get("files", []):
            for imp in file_data.get("imports", []):
                if imp not in external_deps:
                    external_deps[imp] = 0
                external_deps[imp] += 1

        external = [
            {"name": pkg, "count": count}
            for pkg, count in sorted(
                external_deps.items(), key=lambda x: x[1], reverse=True
            )
        ][:10]

        # Prepare graph data
        nodes = []
        links = []

        # Add nodes
        for file_data in analysis_data.get("files", []):
            file_name = file_data.get("path", "unknown")
            nodes.append({"id": file_name, "group": 1})

        # Add edges
        for src, dst in analysis_data.get("dependencies", []):
            links.append({"source": src, "target": dst, "value": 1})

        graph = {"nodes": nodes, "links": links}

        return {
            "most_referenced": most_referenced,
            "external": external,
            "graph": graph,
        }

    def _prepare_documentation_data(self, analysis_data: Dict) -> Dict:
        """Prepare documentation data for the dashboard."""
        undocumented = []
        documented_count = 0
        undocumented_count = 0

        for file_data in analysis_data.get("files", []):
            file_name = file_data.get("path", "unknown")

            for definition in file_data.get("definitions", []):
                if definition.get("docstring"):
                    documented_count += 1
                else:
                    undocumented_count += 1
                    undocumented.append(
                        {
                            "name": definition.get("name", "unknown"),
                            "type": definition.get("category", "unknown"),
                            "file": file_name,
                        }
                    )

        chart_data = {
            "labels": ["Documented", "Undocumented"],
            "datasets": [
                {
                    "data": [documented_count, undocumented_count],
                    "backgroundColor": ["#36a2eb", "#ff6384"],
                }
            ],
        }

        return {
            "undocumented": sorted(undocumented, key=lambda x: x["file"])[:20],
            "chart_data": chart_data,
        }

    def _prepare_orphan_data(self, analysis_data: Dict) -> Dict:
        """Prepare orphan data for the dashboard."""
        return analysis_data.get(
            "orphans", {"functions": [], "classes": [], "variables": []}
        )

    def _prepare_variable_data(self, analysis_data: Dict) -> Dict:
        """Prepare variable data for the dashboard."""
        # Find variables with multiple definitions or high usage
        multi_defined = []

        for var_name, var_data in analysis_data.get("variables", {}).items():
            if var_data.get("definition_count", 0) > 1:
                multi_defined.append(
                    {
                        "name": var_name,
                        "definition_count": var_data.get("definition_count", 0),
                        "usage_count": var_data.get("usage_count", 0),
                    }
                )

        # Sort by definition count
        multi_defined.sort(key=lambda x: x["definition_count"], reverse=True)

        # Prepare chart data
        chart_labels = []
        definition_counts = []
        usage_counts = []

        for var in multi_defined[:10]:  # Top 10 for chart
            chart_labels.append(var["name"])
            definition_counts.append(var["definition_count"])
            usage_counts.append(var["usage_count"])

        return {
            "multi_defined": multi_defined[:15],  # Top 15 for table
            "chart_labels": chart_labels,
            "definition_counts": definition_counts,
            "usage_counts": usage_counts,
        }

    def _prepare_prerequisite_data(self, analysis_data: Dict) -> Dict:
        """Prepare prerequisite data for the dashboard."""
        return {
            "initialization_order": analysis_data.get("initialization_order", []),
            "cycles": analysis_data.get("cycles", []),
        }
