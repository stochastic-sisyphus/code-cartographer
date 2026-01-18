"""
Immersive Dashboard Generator
==============================
Generates data and HTML for the immersive, dynamic code visualization.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader

# Constants for complexity normalization
MAX_CYCLOMATIC_COMPLEXITY = 50  # Cap for normalization to 0-100 scale


class ImmersiveDashboardGenerator:
    """Generates immersive, interactive visualization dashboards."""
    
    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize the dashboard generator.
        
        Args:
            template_dir: Directory containing Jinja2 templates
        """
        if template_dir is None:
            # Default to templates directory in package
            template_dir = Path(__file__).parent.parent.parent / 'templates'
        
        self.template_dir = Path(template_dir)
        self.env = Environment(loader=FileSystemLoader(str(self.template_dir)))
    
    def generate(
        self,
        analysis_data: Dict[str, Any],
        output_path: Path,
        temporal_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate an immersive dashboard from analysis data.
        
        Args:
            analysis_data: Code analysis results
            output_path: Path to save the dashboard HTML
            temporal_data: Optional temporal evolution data
            
        Returns:
            Path to the generated dashboard
        """
        # Transform analysis data into visualization-friendly format
        viz_data = self._prepare_visualization_data(analysis_data, temporal_data)
        
        # Load template
        template = self.env.get_template('immersive_dashboard.html.j2')
        
        # Render template
        html_content = template.render(data=viz_data)
        
        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content)
        
        return output_path
    
    def _prepare_visualization_data(
        self,
        analysis_data: Dict[str, Any],
        temporal_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Transform analysis data into format suitable for D3.js visualization.
        
        Args:
            analysis_data: Raw analysis data
            temporal_data: Optional temporal evolution data
            
        Returns:
            Visualization-ready data structure
        """
        nodes = []
        links = []
        
        # Extract files - handle both dict and list formats
        files_data = analysis_data.get('files', [])
        if isinstance(files_data, list):
            # Convert list of file metadata to dict keyed by path
            files = {f['path']: f for f in files_data}
        else:
            files = files_data
        
        node_map = {}  # Map file paths to node indices
        
        for idx, (file_path, file_data) in enumerate(files.items()):
            # Determine node type
            node_type = self._determine_node_type(file_data)
            
            # Calculate complexity
            complexity = self._calculate_complexity_score(file_data)
            
            # Calculate importance (based on connections)
            importance = len(file_data.get('internal_imports', []))
            
            node = {
                'id': file_path,
                'name': Path(file_path).stem,
                'type': node_type,
                'complexity': complexity,
                'importance': importance,
                'description': file_data.get('module_docstring', ''),
                'metrics': {
                    'loc': file_data.get('line_count', 0),
                    'functions': len(file_data.get('definitions', [])),
                }
            }
            
            nodes.append(node)
            node_map[file_path] = idx
        
        # Create links based on imports and dependencies
        for file_path, file_data in files.items():
            source_id = file_path
            
            # Internal imports create links
            for import_path in file_data.get('internal_imports', []):
                if import_path in node_map:
                    # Determine relationship type
                    link_type = self._determine_link_type(
                        files[file_path],
                        files.get(import_path, {})
                    )
                    
                    # Calculate strength based on usage
                    strength = self._calculate_link_strength(
                        file_data,
                        files.get(import_path, {})
                    )
                    
                    link = {
                        'source': source_id,
                        'target': import_path,
                        'type': link_type,
                        'strength': strength
                    }
                    
                    links.append(link)
        
        # Add temporal data if available
        timeline = []
        if temporal_data:
            timeline = temporal_data.get('timeline', [])
        
        return {
            'nodes': nodes,
            'links': links,
            'timeline': timeline,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_nodes': len(nodes),
                'total_links': len(links),
                'avg_complexity': sum(n['complexity'] for n in nodes) / len(nodes) if nodes else 0
            }
        }
    
    def _determine_node_type(self, file_data: Dict[str, Any]) -> str:
        """
        Determine the primary type of a code file.
        
        Args:
            file_data: File analysis data
            
        Returns:
            Node type string
        """
        definitions = file_data.get('definitions', [])
        
        # Count different definition types
        class_count = sum(1 for d in definitions if d.get('category') == 'class')
        function_count = sum(1 for d in definitions if d.get('category') == 'function')
        
        if class_count > function_count:
            return 'class'
        elif function_count > 0:
            return 'function'
        else:
            return 'module'
    
    def _calculate_complexity_score(self, file_data: Dict[str, Any]) -> float:
        """
        Calculate a normalized complexity score for a file.
        
        Args:
            file_data: File analysis data
            
        Returns:
            Complexity score (0-100)
        """
        metrics = file_data.get('metrics', {})
        
        # Use cyclomatic complexity if available
        cyclomatic = metrics.get('cyclomatic', 0)
        
        # Normalize to 0-100 scale using defined constant
        normalized = min(100, (cyclomatic / MAX_CYCLOMATIC_COMPLEXITY) * 100)
        
        return normalized
    
    def _determine_link_type(
        self,
        source_file: Dict[str, Any],
        target_file: Dict[str, Any]
    ) -> str:
        """
        Determine the type of relationship between two files.
        
        Args:
            source_file: Source file data
            target_file: Target file data
            
        Returns:
            Link type string ('harmony', 'tension', or 'neutral')
        """
        # Files with similar complexity are in harmony
        source_complexity = self._calculate_complexity_score(source_file)
        target_complexity = self._calculate_complexity_score(target_file)
        
        complexity_diff = abs(source_complexity - target_complexity)
        
        if complexity_diff < 20:
            return 'harmony'
        elif complexity_diff > 50:
            return 'tension'
        else:
            return 'neutral'
    
    def _calculate_link_strength(
        self,
        source_file: Dict[str, Any],
        target_file: Dict[str, Any]
    ) -> float:
        """
        Calculate the strength of the relationship between files.
        
        Args:
            source_file: Source file data
            target_file: Target file data
            
        Returns:
            Strength value (0.0-1.0)
        """
        # Count how many definitions from target are used in source
        # This is simplified - could be enhanced with actual call graph
        source_defs = len(source_file.get('definitions', []))
        target_defs = len(target_file.get('definitions', []))
        
        if target_defs == 0:
            return 0.3
        
        # Estimate based on definition counts
        strength = min(1.0, source_defs / (target_defs * 2))
        
        return max(0.1, strength)
    
    def generate_with_temporal(
        self,
        current_analysis: Dict[str, Any],
        temporal_snapshots: List[Dict[str, Any]],
        output_path: Path
    ) -> Path:
        """
        Generate dashboard with temporal evolution data.
        
        Args:
            current_analysis: Current code analysis
            temporal_snapshots: List of historical snapshots
            output_path: Output path for dashboard
            
        Returns:
            Path to generated dashboard
        """
        # Build timeline data
        timeline = []
        for snapshot in temporal_snapshots:
            timeline.append({
                'timestamp': snapshot.get('timestamp'),
                'commit': snapshot.get('commit', '')[:8],
                'element_count': len(snapshot.get('elements', {})),
                'complexity': snapshot.get('avg_complexity', 0)
            })
        
        temporal_data = {
            'timeline': timeline,
            'snapshots': temporal_snapshots
        }
        
        return self.generate(current_analysis, output_path, temporal_data)
    
    def export_json_data(
        self,
        analysis_data: Dict[str, Any],
        output_path: Path,
        temporal_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Export visualization data as JSON for standalone use.
        
        Args:
            analysis_data: Analysis results
            output_path: Path to save JSON
            temporal_data: Optional temporal data
            
        Returns:
            Path to exported JSON file
        """
        viz_data = self._prepare_visualization_data(analysis_data, temporal_data)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open('w') as f:
            json.dump(viz_data, f, indent=2)
        
        return output_path
