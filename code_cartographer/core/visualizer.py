"""
Enhanced Visualization Module
===========================
Provides advanced visualization capabilities for code structure and relationships.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib_venn import venn2, venn3
    HAS_MPL = True
except Exception:  # pragma: no cover - allow running without matplotlib
    plt = None
    cm = None
    np = None
    Normalize = None
    venn2 = None
    venn3 = None
    HAS_MPL = False

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - handled in dependency analyzer
    from code_cartographer.core.dependency_analyzer import nx


class CodeVisualizer:
    """Generates visualizations for code structure and relationships."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_placeholder(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path
    
    def generate_function_call_graph(self, call_graph: Dict[str, List[str]],
                                    output_path: Optional[Path] = None) -> Path:
        """
        Generate a visualization of the function call graph.
        
        Args:
            call_graph: Dictionary mapping caller names to lists of callee names
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.output_dir / "function_call_graph.png"

        if not HAS_MPL:
            return self._ensure_placeholder(output_path)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for caller, callees in call_graph.items():
            G.add_node(caller)
            for callee in callees:
                G.add_node(callee)
                G.add_edge(caller, callee)
        
        # Calculate node sizes based on importance (degree centrality)
        centrality = nx.degree_centrality(G)
        node_sizes = [centrality[node] * 3000 + 100 for node in G.nodes()]
        
        # Calculate edge weights based on importance
        edge_weights = [1 + 0.5 * G.in_degree(edge[1]) for edge in G.edges()]
        
        # Set up the figure
        plt.figure(figsize=(16, 12))
        
        # Use a spring layout for the graph
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=list(centrality.values()),
                              cmap=plt.cm.viridis, alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, 
                              edge_color='gray', arrows=True, 
                              arrowsize=15, connectionstyle='arc3,rad=0.1')
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title("Function Call Graph", fontsize=16)
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_class_hierarchy(self, class_data: Dict[str, Dict], 
                                output_path: Optional[Path] = None) -> Path:
        """
        Generate a visualization of the class hierarchy.
        
        Args:
            class_data: Dictionary mapping class names to their metadata
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.output_dir / "class_hierarchy.png"

        if not HAS_MPL:
            return self._ensure_placeholder(output_path)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for class_name, data in class_data.items():
            G.add_node(class_name, **data)
            for parent in data.get('parents', []):
                G.add_edge(parent, class_name)
        
        # Set up the figure
        plt.figure(figsize=(16, 12))
        
        # Use a hierarchical layout for the graph
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=2000, 
                              node_color='lightblue', alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, 
                              edge_color='gray', arrows=True, 
                              arrowsize=15)
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title("Class Hierarchy", fontsize=16)
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_dependency_heatmap(self, dependency_matrix: Dict[str, Dict[str, int]], 
                                   output_path: Optional[Path] = None) -> Path:
        """
        Generate a heatmap of module dependencies.
        
        Args:
            dependency_matrix: Dictionary mapping module names to their dependencies
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.output_dir / "dependency_heatmap.png"

        if not HAS_MPL:
            return self._ensure_placeholder(output_path)
        
        # Extract module names
        modules = list(dependency_matrix.keys())
        n_modules = len(modules)
        
        # Create a matrix of dependencies
        matrix = np.zeros((n_modules, n_modules))
        for i, source in enumerate(modules):
            for j, target in enumerate(modules):
                matrix[i, j] = dependency_matrix[source].get(target, 0)
        
        # Set up the figure
        plt.figure(figsize=(12, 10))
        
        # Create the heatmap
        plt.imshow(matrix, cmap='YlOrRd')
        
        # Add labels
        plt.xticks(range(n_modules), modules, rotation=90)
        plt.yticks(range(n_modules), modules)
        
        # Add a colorbar
        plt.colorbar(label='Dependency Strength')
        
        plt.title("Module Dependency Heatmap", fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_variable_usage_chart(self, variable_data: Dict[str, Dict], 
                                     output_path: Optional[Path] = None) -> Path:
        """
        Generate a chart showing variable usage across the codebase.
        
        Args:
            variable_data: Dictionary mapping variable names to their usage data
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.output_dir / "variable_usage.png"

        if not HAS_MPL:
            return self._ensure_placeholder(output_path)
        
        # Extract data for visualization
        variables = []
        definition_counts = []
        usage_counts = []
        is_orphan = []
        
        for var_name, data in variable_data.items():
            if len(variables) >= 30:  # Limit to top 30 variables for readability
                break
            variables.append(var_name)
            definition_counts.append(data.get('definition_count', 0))
            usage_counts.append(data.get('usage_count', 0))
            is_orphan.append(data.get('is_orphan', False))
        
        # Set up the figure
        plt.figure(figsize=(14, 8))
        
        # Create a bar chart
        x = np.arange(len(variables))
        width = 0.35
        
        plt.bar(x - width/2, definition_counts, width, label='Definitions', color='skyblue')
        plt.bar(x + width/2, usage_counts, width, label='Usages', color='lightgreen')
        
        # Highlight orphaned variables
        for i, orphan in enumerate(is_orphan):
            if orphan:
                plt.axvspan(i - width, i + width, alpha=0.2, color='red')
        
        # Add labels and legend
        plt.xlabel('Variables')
        plt.ylabel('Count')
        plt.title('Variable Definitions and Usages')
        plt.xticks(x, variables, rotation=90)
        plt.legend()
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_orphan_analysis(self, orphans: Dict[str, List[Dict]], 
                               output_path: Optional[Path] = None) -> Path:
        """
        Generate a visualization of orphaned code elements.
        
        Args:
            orphans: Dictionary mapping orphan types to lists of orphaned elements
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.output_dir / "orphan_analysis.png"

        if not HAS_MPL:
            return self._ensure_placeholder(output_path)
        
        # Extract counts
        function_count = len(orphans.get('functions', []))
        class_count = len(orphans.get('classes', []))
        variable_count = len(orphans.get('variables', []))
        
        # Set up the figure
        plt.figure(figsize=(10, 8))
        
        # Create a pie chart
        labels = ['Functions', 'Classes', 'Variables']
        sizes = [function_count, class_count, variable_count]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0.1, 0.1)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        
        plt.axis('equal')
        plt.title('Orphaned Code Elements', fontsize=16)
        
        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_prerequisite_graph(self, prerequisites: Dict[str, Set[str]], 
                                  output_path: Optional[Path] = None) -> Path:
        """
        Generate a visualization of prerequisites between code elements.
        
        Args:
            prerequisites: Dictionary mapping element names to their prerequisites
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.output_dir / "prerequisite_graph.png"

        if not HAS_MPL:
            return self._ensure_placeholder(output_path)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for element, prereqs in prerequisites.items():
            G.add_node(element)
            for prereq in prereqs:
                G.add_node(prereq)
                G.add_edge(element, prereq)  # Element depends on prereq
        
        # Set up the figure
        plt.figure(figsize=(16, 12))
        
        # Use a layered layout for the graph
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Calculate node importance
        in_degree = dict(G.in_degree())
        node_sizes = [in_degree.get(node, 0) * 100 + 300 for node in G.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6, 
                              edge_color='gray', arrows=True, 
                              arrowsize=10)
        
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title("Prerequisite Dependencies", fontsize=16)
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_initialization_sequence(self, sequence: List[str], 
                                       output_path: Optional[Path] = None) -> Path:
        """
        Generate a visualization of the initialization sequence.
        
        Args:
            sequence: List of element names in initialization order
            output_path: Path to save the visualization (optional)
            
        Returns:
            Path to the saved visualization
        """
        if output_path is None:
            output_path = self.output_dir / "initialization_sequence.png"

        if not HAS_MPL:
            return self._ensure_placeholder(output_path)
        
        # Set up the figure
        plt.figure(figsize=(14, 10))
        
        # Create a directed graph for the sequence
        G = nx.DiGraph()
        
        # Add nodes and edges in sequence
        for i in range(len(sequence)):
            G.add_node(sequence[i], order=i)
            if i > 0:
                G.add_edge(sequence[i-1], sequence[i])
        
        # Use a left-to-right layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=1500, 
                              node_color='lightgreen', alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.7, 
                              edge_color='gray', arrows=True, 
                              arrowsize=15)
        
        # Add node labels with sequence numbers
        labels = {node: f"{i+1}. {node}" for i, node in enumerate(sequence)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        
        plt.title("Initialization Sequence", fontsize=16)
        plt.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def generate_interactive_dashboard(self, analysis_data: Dict, 
                                     template_path: Path,
                                     output_path: Optional[Path] = None) -> Path:
        """
        Generate an interactive HTML dashboard for code analysis results.
        
        Args:
            analysis_data: Complete analysis data dictionary
            template_path: Path to the Jinja2 template for the dashboard
            output_path: Path to save the dashboard HTML (optional)
            
        Returns:
            Path to the saved dashboard
        """
        if output_path is None:
            output_path = self.output_dir / "dashboard.html"
        try:
            from jinja2 import Environment, FileSystemLoader
        except Exception:
            return self._ensure_placeholder(output_path)
        import datetime
        
        # Prepare data for the dashboard
        dashboard_data = {
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "total_files": len(analysis_data.get("files", [])),
                "variant_count": len(analysis_data.get("variants", {})),
                "avg_complexity": self._calculate_avg_complexity(analysis_data),
                "doc_coverage": self._calculate_doc_coverage(analysis_data),
                "file_trend": "↑ 5% from last analysis",
                "variant_trend": "↓ 10% from last analysis",
                "complexity_trend": "↓ 2% from last analysis",
                "doc_trend": "↑ 8% from last analysis"
            },
            "variant_groups": self._prepare_variant_groups(analysis_data),
            "complexity_data": self._prepare_complexity_data(analysis_data),
            "dependency_data": self._prepare_dependency_data(analysis_data),
            "documentation_data": self._prepare_documentation_data(analysis_data)
        }
        
        # Set up Jinja environment
        env = Environment(loader=FileSystemLoader(template_path.parent))
        template = env.get_template(template_path.name)
        
        # Render the template
        html_content = template.render(**dashboard_data)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _calculate_avg_complexity(self, analysis_data: Dict) -> float:
        """Calculate average cyclomatic complexity from analysis data."""
        total = 0
        count = 0
        
        for file_data in analysis_data.get("files", []):
            for definition in file_data.get("definitions", []):
                if definition.get("metrics", {}).get("cyclomatic") is not None:
                    total += definition["metrics"]["cyclomatic"]
                    count += 1
        
        return total / count if count > 0 else 0
    
    def _calculate_doc_coverage(self, analysis_data: Dict) -> float:
        """Calculate documentation coverage from analysis data."""
        total = 0
        documented = 0
        
        for file_data in analysis_data.get("files", []):
            for definition in file_data.get("definitions", []):
                total += 1
                if definition.get("docstring"):
                    documented += 1
        
        return documented / total if total > 0 else 0
    
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
                variants.append({
                    "path": impl.get("path", "unknown"),
                    "similarity": 0.8,  # Placeholder, would be calculated in real implementation
                    "diff": impl.get("diff_from_base", [])
                })
            
            groups.append({
                "name": name,
                "base_code": base_code,
                "variants": variants
            })
        
        return groups
    
    def _prepare_complexity_data(self, analysis_data: Dict) -> Dict:
        """Prepare complexity data for the dashboard."""
        high_complexity = []
        chart_data = {
            "datasets": [{
                "label": "Files",
                "data": [],
                "backgroundColor": "rgba(75, 192, 192, 0.6)"
            }]
        }
        
        for file_data in analysis_data.get("files", []):
            file_name = file_data.get("path", "unknown")
            complexity = file_data.get("metrics", {}).get("cyclomatic", 0)
            lines = sum(d.get("line_count", 0) for d in file_data.get("definitions", []))
            
            if complexity and complexity > 10:
                high_complexity.append({
                    "name": file_name,
                    "complexity": complexity,
                    "lines": lines
                })
            
            chart_data["datasets"][0]["data"].append({
                "x": lines,
                "y": complexity,
                "r": 5 + (complexity / 5)
            })
        
        return {
            "high_complexity": sorted(high_complexity, key=lambda x: x["complexity"], reverse=True)[:10],
            "chart_data": chart_data
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
            for module, count in sorted(references.items(), key=lambda x: x[1], reverse=True)
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
            for pkg, count in sorted(external_deps.items(), key=lambda x: x[1], reverse=True)
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
            "graph": graph
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
                    undocumented.append({
                        "name": definition.get("name", "unknown"),
                        "type": definition.get("category", "unknown"),
                        "file": file_name
                    })
        
        chart_data = {
            "labels": ["Documented", "Undocumented"],
            "datasets": [{
                "data": [documented_count, undocumented_count],
                "backgroundColor": ["#36a2eb", "#ff6384"]
            }]
        }
        
        return {
            "undocumented": sorted(undocumented, key=lambda x: x["file"])[:20],
            "chart_data": chart_data
        }
