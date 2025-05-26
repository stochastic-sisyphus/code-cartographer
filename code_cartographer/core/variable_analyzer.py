"""
Variable Usage Analyzer
======================
Specialized module for tracking variable definitions and usages across a Python codebase.
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class VariableDefinition:
    """Detailed information about a variable definition."""
    name: str
    file_path: str
    line_number: int
    scope: str  # 'global', 'function', 'class', 'method'
    definition_type: str  # 'assignment', 'parameter', 'import', 'class', 'function'
    type_hint: Optional[str] = None
    value_source: Optional[str] = None  # Source code of the assigned value if available


@dataclass
class VariableUsage:
    """Information about a variable usage."""
    name: str
    file_path: str
    line_number: int
    scope: str  # 'global', 'function', 'class', 'method'
    usage_type: str  # 'read', 'write', 'delete', 'import'


@dataclass
class VariableFlow:
    """Tracks the flow of a variable through the codebase."""
    name: str = ""  # Default empty string to allow instantiation without arguments
    definitions: List[VariableDefinition] = field(default_factory=list)
    usages: List[VariableUsage] = field(default_factory=list)
    is_orphan: bool = True  # Default to True, will be updated during analysis
    
    @property
    def definition_count(self) -> int:
        return len(self.definitions)
    
    @property
    def usage_count(self) -> int:
        return len(self.usages)
    
    @property
    def is_redefined(self) -> bool:
        return len(self.definitions) > 1
    
    @property
    def definition_locations(self) -> List[Tuple[str, int]]:
        """Get all file paths and line numbers where this variable is defined."""
        return [(d.file_path, d.line_number) for d in self.definitions]
    
    @property
    def usage_locations(self) -> List[Tuple[str, int]]:
        """Get all file paths and line numbers where this variable is used."""
        return [(u.file_path, u.line_number) for u in self.usages]


# Factory function for defaultdict to create VariableFlow instances
def variable_flow_factory():
    """Factory function for creating VariableFlow instances with defaultdict."""
    return VariableFlow()


class VariableVisitor(ast.NodeVisitor):
    """AST visitor that tracks variable definitions and usages."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.definitions: List[VariableDefinition] = []
        self.usages: List[VariableUsage] = []
        # Initialize scope_stack first before any property access
        self.scope_stack: List[str] = ["global"]
        # Now it's safe to set current_scope
        self._current_scope = "global"
    
    @property
    def current_scope(self) -> str:
        # Ensure scope_stack is initialized and has at least one element
        if not hasattr(self, 'scope_stack') or not self.scope_stack:
            self.scope_stack = ["global"]
        return self.scope_stack[-1]
    
    @current_scope.setter
    def current_scope(self, scope: str):
        # Ensure scope_stack is initialized
        if not hasattr(self, 'scope_stack'):
            self.scope_stack = ["global"]
        
        # Safely modify the stack
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
        self.scope_stack.append(scope)
    
    def push_scope(self, scope: str):
        # Ensure scope_stack is initialized
        if not hasattr(self, 'scope_stack'):
            self.scope_stack = ["global"]
        self.scope_stack.append(scope)
    
    def pop_scope(self):
        # Ensure scope_stack is initialized and has more than one element
        if not hasattr(self, 'scope_stack'):
            self.scope_stack = ["global"]
        elif len(self.scope_stack) > 1:
            self.scope_stack.pop()
    
    def visit_Name(self, node):
        """Process variable names."""
        if isinstance(node.ctx, ast.Store):
            # Variable definition
            self.definitions.append(
                VariableDefinition(
                    name=node.id,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    scope=self.current_scope,
                    definition_type="assignment"
                )
            )
        elif isinstance(node.ctx, ast.Load):
            # Variable usage
            self.usages.append(
                VariableUsage(
                    name=node.id,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    scope=self.current_scope,
                    usage_type="read"
                )
            )
        elif isinstance(node.ctx, ast.Del):
            # Variable deletion
            self.usages.append(
                VariableUsage(
                    name=node.id,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    scope=self.current_scope,
                    usage_type="delete"
                )
            )
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Process function definitions and their parameters."""
        old_scope = self.current_scope
        self.push_scope(f"function:{node.name}")
        
        # Add function name as a definition in the parent scope
        self.definitions.append(
            VariableDefinition(
                name=node.name,
                file_path=self.file_path,
                line_number=node.lineno,
                scope=old_scope,
                definition_type="function"
            )
        )
        
        # Process parameters
        for arg in node.args.args:
            type_hint = None
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation)
                
            self.definitions.append(
                VariableDefinition(
                    name=arg.arg,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    scope=self.current_scope,
                    definition_type="parameter",
                    type_hint=type_hint
                )
            )
        
        # Process function body
        self.generic_visit(node)
        self.pop_scope()
    
    def visit_AsyncFunctionDef(self, node):
        """Process async function definitions."""
        self.visit_FunctionDef(node)  # Reuse the same logic
    
    def visit_ClassDef(self, node):
        """Process class definitions."""
        old_scope = self.current_scope
        self.push_scope(f"class:{node.name}")
        
        # Add class name as a definition in the parent scope
        self.definitions.append(
            VariableDefinition(
                name=node.name,
                file_path=self.file_path,
                line_number=node.lineno,
                scope=old_scope,
                definition_type="class"
            )
        )
        
        # Process class body
        self.generic_visit(node)
        self.pop_scope()
    
    def visit_Import(self, node):
        """Process import statements."""
        for alias in node.names:
            name = alias.asname or alias.name
            self.definitions.append(
                VariableDefinition(
                    name=name,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    scope=self.current_scope,
                    definition_type="import",
                    value_source=alias.name
                )
            )
        
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Process from-import statements."""
        for alias in node.names:
            name = alias.asname or alias.name
            module_path = f"{node.module}.{alias.name}" if node.module else alias.name
            self.definitions.append(
                VariableDefinition(
                    name=name,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    scope=self.current_scope,
                    definition_type="import",
                    value_source=module_path
                )
            )
        
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        """Process assignment statements with value extraction."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Try to extract the assigned value as source code
                try:
                    value_source = ast.unparse(node.value)
                except Exception:
                    value_source = None
                
                # Update the most recent definition with the value source
                for definition in reversed(self.definitions):
                    if definition.name == target.id and definition.line_number == target.lineno:
                        definition.value_source = value_source
                        break
        
        self.generic_visit(node)
    
    def visit_AnnAssign(self, node):
        """Process annotated assignments."""
        if isinstance(target, ast.Name):
            type_hint = ast.unparse(node.annotation) if node.annotation else None
            
            # Try to extract the assigned value as source code
            value_source = None
            if node.value:
                try:
                    value_source = ast.unparse(node.value)
                except Exception:
                    pass
            
            self.definitions.append(
                VariableDefinition(
                    name=node.target.id,
                    file_path=self.file_path,
                    line_number=node.lineno,
                    scope=self.current_scope,
                    definition_type="assignment",
                    type_hint=type_hint,
                    value_source=value_source
                )
            )
        
        self.generic_visit(node)


class VariableAnalyzer:
    """Analyzes variable usage across a Python codebase."""
    
    def __init__(self, root: Path, exclude_patterns: List[str] = None):
        self.root = root
        self.exclude_patterns = exclude_patterns or []
        # Use the factory function to create VariableFlow instances
        self.variable_flows: Dict[str, VariableFlow] = defaultdict(variable_flow_factory)
    
    def analyze_file(self, file_path: Path) -> Tuple[List[VariableDefinition], List[VariableUsage]]:
        """Analyze a single Python file for variable definitions and usages."""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            
            relative_path = str(file_path.relative_to(self.root))
            visitor = VariableVisitor(relative_path)
            visitor.visit(tree)
            
            return visitor.definitions, visitor.usages
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return [], []
    
    def analyze(self) -> Dict[str, VariableFlow]:
        """Analyze all Python files in the project."""
        for file_path in self._find_python_files():
            definitions, usages = self.analyze_file(file_path)
            
            # Process definitions
            for definition in definitions:
                var_flow = self.variable_flows[definition.name]
                var_flow.name = definition.name
                var_flow.definitions.append(definition)
            
            # Process usages
            for usage in usages:
                var_flow = self.variable_flows[usage.name]
                var_flow.name = usage.name
                var_flow.usages.append(usage)
        
        # Update orphan status
        for var_name, var_flow in self.variable_flows.items():
            var_flow.is_orphan = var_flow.usage_count == 0
        
        return dict(self.variable_flows)
    
    def _find_python_files(self):
        """Find all Python files in the project, respecting exclusion patterns."""
        import re
        exclude_patterns = [re.compile(pattern) for pattern in self.exclude_patterns]
        
        for file_path in self.root.rglob("*.py"):
            relative_path = str(file_path.relative_to(self.root))
            if any(pattern.search(relative_path) for pattern in exclude_patterns):
                continue
            yield file_path
    
    def get_orphaned_variables(self) -> List[str]:
        """Get a list of orphaned variables (defined but never used)."""
        return [
            name for name, flow in self.variable_flows.items()
            if flow.is_orphan and not name.startswith("_")  # Exclude private variables
        ]
    
    def get_undefined_variables(self) -> List[str]:
        """Get a list of variables that are used but never defined."""
        undefined = []
        for name, flow in self.variable_flows.items():
            if not flow.definitions and flow.usages:
                undefined.append(name)
        return undefined
    
    def get_variable_dependencies(self) -> Dict[str, Set[str]]:
        """Get dependencies between variables based on their definitions."""
        dependencies = defaultdict(set)
        
        for name, flow in self.variable_flows.items():
            for definition in flow.definitions:
                if definition.value_source:
                    # Extract variable names from the value source
                    try:
                        value_tree = ast.parse(definition.value_source)
                        for node in ast.walk(value_tree):
                            if isinstance(node, ast.Name):
                                dependencies[name].add(node.id)
                    except Exception:
                        pass
        
        return dict(dependencies)
    
    def generate_variable_report(self) -> Dict:
        """Generate a comprehensive report of variable usage."""
        report = {
            "variables": {},
            "orphaned_count": len(self.get_orphaned_variables()),
            "undefined_count": len(self.get_undefined_variables()),
            "total_count": len(self.variable_flows),
            "dependencies": self.get_variable_dependencies()
        }
        
        for name, flow in self.variable_flows.items():
            report["variables"][name] = {
                "definition_count": flow.definition_count,
                "usage_count": flow.usage_count,
                "is_orphan": flow.is_orphan,
                "is_redefined": flow.is_redefined,
                "definition_locations": flow.definition_locations,
                "usage_locations": flow.usage_locations
            }
        
        return report


# Define a VariableTracker class that wraps VariableVisitor for compatibility with imports
class VariableTracker(VariableVisitor):
    """Compatibility wrapper for VariableVisitor to maintain API compatibility."""
    pass


# Define VariableMetadata as an alias for VariableDefinition for compatibility with imports
VariableMetadata = VariableDefinition


# Export all public classes and functions
__all__ = [
    'VariableDefinition',
    'VariableUsage',
    'VariableFlow',
    'VariableVisitor',
    'VariableTracker',
    'VariableMetadata',  # Added for compatibility with imports
    'VariableAnalyzer',
    'variable_flow_factory'
]
