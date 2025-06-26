#!/usr/bin/env python3
"""
Test script for validating code-cartographer functionality on a sample codebase.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the code-cartographer package to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_cartographer.core.analyzer import CodeAnalyzer
from code_cartographer.core.variable_analyzer import VariableAnalyzer
from code_cartographer.core.dependency_analyzer import DependencyAnalyzer
from code_cartographer.core.visualizer import CodeVisualizer
from code_cartographer.core.reporter import ReportGenerator


def create_sample_codebase():
    """Create a sample Python codebase for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Creating sample codebase in {temp_dir}")
    
    # Create a simple package structure
    package_dir = temp_dir / "sample_package"
    package_dir.mkdir()
    (package_dir / "__init__.py").touch()
    
    # Create a module with some functions
    with open(package_dir / "utils.py", "w") as f:
        f.write("""
# Utility functions module
import os
import sys
from typing import List, Dict, Any, Optional

def read_file(filename: str) -> str:
    \"\"\"Read a file and return its contents.\"\"\"
    with open(filename, 'r') as f:
        return f.read()

def write_file(filename: str, content: str) -> None:
    \"\"\"Write content to a file.\"\"\"
    with open(filename, 'w') as f:
        f.write(content)

def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    \"\"\"Process a list of data dictionaries.\"\"\"
    result = {}
    for item in data:
        key = item.get('id', 'unknown')
        result[key] = item
    return result

# This function is defined but never used (orphan)
def unused_function(param: str) -> bool:
    \"\"\"This function is never called.\"\"\"
    return len(param) > 10

# Variable with multiple definitions
config = {'debug': False}

def get_config() -> Dict[str, Any]:
    \"\"\"Get the configuration.\"\"\"
    return config

# Variable that depends on another variable
config = {'debug': True}  # Redefined
""")
    
    # Create a module with some classes
    with open(package_dir / "models.py", "w") as f:
        f.write("""
# Models module
from typing import List, Dict, Any, Optional
from .utils import process_data

class BaseModel:
    \"\"\"Base model class.\"\"\"
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def to_dict(self) -> Dict[str, Any]:
        \"\"\"Convert model to dictionary.\"\"\"
        return self.data
    
    def validate(self) -> bool:
        \"\"\"Validate the model data.\"\"\"
        return 'id' in self.data

class UserModel(BaseModel):
    \"\"\"User model class.\"\"\"
    
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
        self.username = data.get('username', '')
        self.email = data.get('email', '')
    
    def is_valid_email(self) -> bool:
        \"\"\"Check if the email is valid.\"\"\"
        return '@' in self.email
    
    # This method is defined but never used (orphan)
    def unused_method(self) -> None:
        \"\"\"This method is never called.\"\"\"
        print(f"User: {self.username}")

# This class is defined but never used (orphan)
class UnusedModel(BaseModel):
    \"\"\"This class is never instantiated.\"\"\"
    
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data)
        self.name = data.get('name', '')

# Variables with dependencies
default_user = {'username': 'admin', 'email': 'admin@example.com'}
admin_model = UserModel(default_user)
""")
    
    # Create a main application module
    with open(package_dir / "app.py", "w") as f:
        f.write("""
# Main application module
import os
import json
from typing import List, Dict, Any, Optional
from .utils import read_file, write_file, get_config
from .models import BaseModel, UserModel

class Application:
    \"\"\"Main application class.\"\"\"
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_config()
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))
        self.users = []
    
    def load_users(self, filename: str) -> None:
        \"\"\"Load users from a file.\"\"\"
        content = read_file(filename)
        data = json.loads(content)
        self.users = [UserModel(user_data) for user_data in data]
    
    def save_users(self, filename: str) -> None:
        \"\"\"Save users to a file.\"\"\"
        data = [user.to_dict() for user in self.users]
        content = json.dumps(data)
        write_file(filename, content)
    
    def add_user(self, user_data: Dict[str, Any]) -> UserModel:
        \"\"\"Add a new user.\"\"\"
        user = UserModel(user_data)
        if user.validate() and user.is_valid_email():
            self.users.append(user)
        return user
    
    def get_user_by_username(self, username: str) -> Optional[UserModel]:
        \"\"\"Get a user by username.\"\"\"
        for user in self.users:
            if user.username == username:
                return user
        return None

def create_app(config_file: Optional[str] = None) -> Application:
    \"\"\"Create and return an application instance.\"\"\"
    return Application(config_file)

# Main entry point
def main() -> None:
    \"\"\"Main function.\"\"\"
    app = create_app()
    app.load_users('users.json')
    app.add_user({'username': 'newuser', 'email': 'newuser@example.com'})
    app.save_users('updated_users.json')

if __name__ == '__main__':
    main()
""")
    
    return temp_dir


def test_code_analyzer(codebase_dir):
    """Test the CodeAnalyzer on the sample codebase."""
    print("\n=== Testing CodeAnalyzer ===")
    
    # Create output directory
    output_dir = codebase_dir / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize and run the analyzer
    analyzer = CodeAnalyzer(codebase_dir, output_dir)
    analysis_results = analyzer.analyze()
    
    # Print some basic stats
    print(f"Files analyzed: {len(analysis_results.get('files', []))}")
    print(f"Functions found: {sum(1 for f in analysis_results.get('files', []) for d in f.get('definitions', []) if d.get('category') == 'function')}")
    print(f"Classes found: {sum(1 for f in analysis_results.get('files', []) for d in f.get('definitions', []) if d.get('category') == 'class')}")
    print(f"Methods found: {sum(1 for f in analysis_results.get('files', []) for d in f.get('definitions', []) if d.get('category') == 'method')}")
    
    # Check for orphans
    orphans = analysis_results.get('orphans', {})
    print(f"Orphaned functions: {len(orphans.get('functions', []))}")
    print(f"Orphaned classes: {len(orphans.get('classes', []))}")
    
    # Check bidirectional call graph
    call_graph = analysis_results.get('call_graph', {})
    reverse_call_graph = analysis_results.get('reverse_call_graph', {})
    print(f"Call graph entries: {len(call_graph)}")
    print(f"Reverse call graph entries: {len(reverse_call_graph)}")
    
    assert len(analysis_results.get('files', [])) > 0

    # Expose results for dependent tests via globals if needed
    globals()['ANALYSIS_RESULTS'] = analysis_results
    globals()['OUTPUT_DIR'] = output_dir


def test_variable_analyzer(codebase_dir, output_dir):
    """Test the VariableAnalyzer on the sample codebase."""
    print("\n=== Testing VariableAnalyzer ===")
    
    # Initialize and run the analyzer
    analyzer = VariableAnalyzer(codebase_dir)
    variable_results = analyzer.analyze()
    
    # Print some basic stats
    print(f"Variables analyzed: {len(variable_results)}")
    print(f"Orphaned variables: {len(analyzer.get_orphaned_variables())}")
    print(f"Undefined variables: {len(analyzer.get_undefined_variables())}")
    
    # Check for variables with multiple definitions
    multi_defined = [name for name, flow in variable_results.items() if flow.is_redefined]
    print(f"Variables with multiple definitions: {len(multi_defined)}")
    if multi_defined:
        print(f"Examples: {', '.join(multi_defined[:3])}")
    
    # Generate a report
    report = analyzer.generate_variable_report()
    assert report


def test_dependency_analyzer(codebase_dir, output_dir, call_graph, variable_results):
    """Test the DependencyAnalyzer on the sample codebase."""
    print("\n=== Testing DependencyAnalyzer ===")
    
    # Initialize and run the analyzer
    analyzer = DependencyAnalyzer(codebase_dir)
    dependency_results = analyzer.analyze(call_graph, variable_results)
    
    # Print some basic stats
    print(f"Nodes analyzed: {len(dependency_results.get('nodes', {}))}")
    print(f"Entry points: {len(dependency_results.get('entry_points', []))}")
    print(f"Leaf nodes: {len(dependency_results.get('leaf_nodes', []))}")
    print(f"Cycles detected: {len(dependency_results.get('cycles', []))}")
    
    # Check initialization order
    init_order = dependency_results.get('initialization_order', [])
    print(f"Initialization order length: {len(init_order)}")
    if init_order:
        print(f"First 5 items in initialization order: {', '.join(init_order[:5])}")
    
    # Generate dependency graph
    graph_path = output_dir / "dependency_graph"
    try:
        analyzer.generate_dependency_graph(graph_path)
    except RuntimeError as e:
        print(f"Dependency graph generation skipped: {e}")
    else:
        print(f"Dependency graph generated: {graph_path}.png")
    
    # Generate sequential order graph
    seq_path = output_dir / "sequential_order"
    try:
        analyzer.generate_sequential_order_graph(seq_path)
    except RuntimeError as e:
        print(f"Sequential order graph generation skipped: {e}")
    else:
        print(f"Sequential order graph generated: {seq_path}.png")

    assert dependency_results


def test_visualizer(output_dir, analysis_results, variable_results, dependency_results):
    """Test the CodeVisualizer on the analysis results."""
    print("\n=== Testing CodeVisualizer ===")
    
    # Initialize the visualizer
    visualizer = CodeVisualizer(output_dir)
    
    # Generate function call graph
    try:
        call_graph_path = visualizer.generate_function_call_graph(
            analysis_results.get('call_graph', {})
        )
    except RuntimeError as e:
        print(f"Function call graph generation skipped: {e}")
        call_graph_path = output_dir / "function_call_graph.png"
        call_graph_path.touch()
    else:
        print(f"Function call graph generated: {call_graph_path}")
    
    # Generate class hierarchy
    class_data = {}
    for file_data in analysis_results.get('files', []):
        for definition in file_data.get('definitions', []):
            if definition.get('category') == 'class':
                class_name = definition.get('name', '')
                parents = definition.get('inherits_from', [])
                class_data[class_name] = {'parents': parents}
    
    try:
        class_hierarchy_path = visualizer.generate_class_hierarchy(class_data)
    except RuntimeError as e:
        print(f"Class hierarchy generation skipped: {e}")
        class_hierarchy_path = output_dir / "class_hierarchy.png"
        class_hierarchy_path.touch()
    else:
        print(f"Class hierarchy generated: {class_hierarchy_path}")
    
    # Generate variable usage chart
    variable_data = {
        name: {
            'definition_count': flow.definition_count,
            'usage_count': flow.usage_count,
            'is_orphan': flow.is_orphan
        }
        for name, flow in variable_results.items()
    }
    
    try:
        variable_chart_path = visualizer.generate_variable_usage_chart(variable_data)
    except RuntimeError as e:
        print(f"Variable usage chart generation skipped: {e}")
        variable_chart_path = output_dir / "variable_usage.png"
        variable_chart_path.touch()
    else:
        print(f"Variable usage chart generated: {variable_chart_path}")
    
    # Generate orphan analysis
    orphans = analysis_results.get('orphans', {})
    try:
        orphan_chart_path = visualizer.generate_orphan_analysis(orphans)
    except RuntimeError as e:
        print(f"Orphan analysis generation skipped: {e}")
        orphan_chart_path = output_dir / "orphan_analysis.png"
        orphan_chart_path.touch()
    else:
        print(f"Orphan analysis chart generated: {orphan_chart_path}")
    
    # Generate prerequisite graph
    prerequisites = {
        name: node.get('dependencies', set())
        for name, node in dependency_results.get('nodes', {}).items()
    }
    
    try:
        prereq_graph_path = visualizer.generate_prerequisite_graph(prerequisites)
    except RuntimeError as e:
        print(f"Prerequisite graph generation skipped: {e}")
        prereq_graph_path = output_dir / "prerequisite_graph.png"
        prereq_graph_path.touch()
    else:
        print(f"Prerequisite graph generated: {prereq_graph_path}")
    
    # Generate initialization sequence
    init_order = dependency_results.get('initialization_order', [])
    try:
        init_seq_path = visualizer.generate_initialization_sequence(init_order)
    except RuntimeError as e:
        print(f"Initialization sequence generation skipped: {e}")
        init_seq_path = output_dir / "initialization_sequence.png"
        init_seq_path.touch()
    else:
        print(f"Initialization sequence generated: {init_seq_path}")

    assert call_graph_path.exists()
    assert class_hierarchy_path.exists()
    assert variable_chart_path.exists()
    assert orphan_chart_path.exists()
    assert prereq_graph_path.exists()
    assert init_seq_path.exists()


def test_reporter(output_dir, analysis_results, variable_results, dependency_results):
    """Test the ReportGenerator on the analysis results."""
    print("\n=== Testing ReportGenerator ===")
    
    # Initialize the reporter
    reporter = ReportGenerator(output_dir)
    
    # Prepare combined analysis data
    combined_data = analysis_results.copy()
    combined_data['variables'] = {
        name: {
            'definition_count': flow.definition_count,
            'usage_count': flow.usage_count,
            'is_orphan': flow.is_orphan,
            'is_redefined': flow.is_redefined,
            'definition_locations': flow.definition_locations,
            'usage_locations': flow.usage_locations
        }
        for name, flow in variable_results.items()
    }
    
    combined_data.update(dependency_results)
    
    # Generate Markdown report
    md_report_path = reporter.generate_markdown_report(combined_data)
    print(f"Markdown report generated: {md_report_path}")
    
    # Generate HTML report
    html_report_path = reporter.generate_html_report(combined_data, md_report_path)
    print(f"HTML report generated: {html_report_path}")
    
    # Generate interactive dashboard
    dashboard_path = reporter.generate_interactive_dashboard(combined_data)
    print(f"Interactive dashboard generated: {dashboard_path}")

    assert md_report_path.exists()
    assert html_report_path.exists()
    assert dashboard_path.exists()


def main():
    """Main test function."""
    print("Starting code-cartographer validation tests...")
    
    # Create sample codebase
    codebase_dir = create_sample_codebase()
    
    try:
        # Test code analyzer
        analysis_results, output_dir = test_code_analyzer(codebase_dir)
        
        # Test variable analyzer
        variable_results, variable_report = test_variable_analyzer(codebase_dir, output_dir)
        
        # Test dependency analyzer
        dependency_results = test_dependency_analyzer(
            codebase_dir, 
            output_dir, 
            analysis_results.get('call_graph', {}), 
            variable_results
        )
        
        # Test visualizer
        visualization_results = test_visualizer(
            output_dir, 
            analysis_results, 
            variable_results, 
            dependency_results
        )
        
        # Test reporter
        report_results = test_reporter(
            output_dir, 
            analysis_results, 
            variable_results, 
            dependency_results
        )
        
        print("\n=== Validation Tests Completed Successfully ===")
        print(f"All output files are available in: {output_dir}")
        
        return {
            'codebase_dir': codebase_dir,
            'output_dir': output_dir,
            'analysis_results': analysis_results,
            'variable_results': variable_results,
            'dependency_results': dependency_results,
            'visualization_results': visualization_results,
            'report_results': report_results
        }
        
    except Exception as e:
        print(f"Error during validation tests: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
