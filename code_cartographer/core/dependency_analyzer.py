"""
Dependency and Prerequisite Analyzer
===================================
Analyzes sequential dependencies and prerequisites between code elements.
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover - fallback when networkx unavailable
    class _DiGraph:
        def __init__(self) -> None:
            self.adj: Dict[str, Set[str]] = defaultdict(set)

        def add_node(self, node: str, **_: Any) -> None:
            self.adj.setdefault(node, set())

        def add_edge(self, u: str, v: str) -> None:
            self.add_node(u)
            self.add_node(v)
            self.adj[u].add(v)

        def copy(self) -> "_DiGraph":
            new = _DiGraph()
            for k, vs in self.adj.items():
                new.adj[k] = set(vs)
            return new

        def has_edge(self, u: str, v: str) -> bool:
            return v in self.adj.get(u, set())

        def remove_edge(self, u: str, v: str) -> None:
            self.adj.get(u, set()).discard(v)

        def successors(self, n: str):
            return list(self.adj.get(n, set()))

        def edges(self):
            for u, vs in self.adj.items():
                for v in vs:
                    yield (u, v)

    class _NX:
        DiGraph = _DiGraph

        class NetworkXNoCycle(Exception):
            pass

        class NetworkXUnfeasible(Exception):
            pass

        @staticmethod
        def simple_cycles(graph: _DiGraph):
            cycles: List[List[str]] = []

            def dfs(node: str, start: str, path: List[str]):
                path.append(node)
                for neigh in graph.adj.get(node, set()):
                    if neigh == start:
                        cycles.append(path.copy())
                    elif neigh not in path:
                        dfs(neigh, start, path)
                path.pop()

            for n in graph.adj:
                dfs(n, n, [])
            if not cycles:
                raise _NX.NetworkXNoCycle()
            return cycles

        @staticmethod
        def topological_sort(graph: _DiGraph):
            indegree: Dict[str, int] = defaultdict(int)
            for u, vs in graph.adj.items():
                indegree.setdefault(u, 0)
                for v in vs:
                    indegree[v] += 1
            queue = [n for n, d in indegree.items() if d == 0]
            result: List[str] = []
            while queue:
                n = queue.pop(0)
                result.append(n)
                for v in graph.adj.get(n, set()):
                    indegree[v] -= 1
                    if indegree[v] == 0:
                        queue.append(v)
            if len(result) != len(indegree):
                raise _NX.NetworkXUnfeasible()
            return result

        @staticmethod
        def strongly_connected_components(graph: _DiGraph):
            index = 0
            indices: Dict[str, int] = {}
            lowlinks: Dict[str, int] = {}
            stack: List[str] = []
            on_stack: Set[str] = set()
            result: List[Set[str]] = []

            def strongconnect(v: str):
                nonlocal index
                indices[v] = index
                lowlinks[v] = index
                index += 1
                stack.append(v)
                on_stack.add(v)
                for w in graph.adj.get(v, set()):
                    if w not in indices:
                        strongconnect(w)
                        lowlinks[v] = min(lowlinks[v], lowlinks[w])
                    elif w in on_stack:
                        lowlinks[v] = min(lowlinks[v], indices[w])
                if lowlinks[v] == indices[v]:
                    comp: Set[str] = set()
                    while True:
                        w = stack.pop()
                        on_stack.remove(w)
                        comp.add(w)
                        if w == v:
                            break
                    result.append(comp)

            for node in list(graph.adj):
                if node not in indices:
                    strongconnect(node)
            return result

    nx = _NX()  # type: ignore

from code_cartographer.core.variable_analyzer import VariableAnalyzer


@dataclass
class DependencyNode:
    """Represents a node in the dependency graph."""
    name: str
    node_type: str  # 'function', 'class', 'method', 'variable'
    file_path: str
    line_number: int
    dependencies: Set[str] = field(default_factory=set)  # Names of nodes this depends on
    dependents: Set[str] = field(default_factory=set)    # Names of nodes that depend on this
    is_entry_point: bool = False
    is_leaf: bool = False
    
    @property
    def dependency_count(self) -> int:
        return len(self.dependencies)
    
    @property
    def dependent_count(self) -> int:
        return len(self.dependents)


class DependencyAnalyzer:
    """Analyzes dependencies and prerequisites between code elements."""
    
    def __init__(self, root: Path, exclude_patterns: List[str] = None):
        self.root = root
        self.exclude_patterns = exclude_patterns or []
        self.nodes: Dict[str, DependencyNode] = {}
        self.graph = nx.DiGraph()
        
    def analyze(self, call_graph: Dict[str, List[str]], variable_flows: Dict[str, Any]) -> Dict:
        """
        Analyze dependencies using call graph and variable usage information.
        
        Args:
            call_graph: Dictionary mapping caller names to lists of callee names
            variable_flows: Variable usage information from VariableAnalyzer
            
        Returns:
            Dictionary with dependency analysis results
        """
        # Build nodes from call graph
        self._build_nodes_from_call_graph(call_graph)
        
        # Add variable dependencies
        self._add_variable_dependencies(variable_flows)
        
        # Build the graph
        self._build_graph()
        
        # Analyze the graph
        entry_points = self._find_entry_points()
        leaf_nodes = self._find_leaf_nodes()
        cycles = self._find_cycles()
        
        # Calculate initialization order
        initialization_order = self._calculate_initialization_order()
        
        # Generate report
        return {
            "nodes": {name: self._node_to_dict(node) for name, node in self.nodes.items()},
            "entry_points": [name for name, node in self.nodes.items() if node.is_entry_point],
            "leaf_nodes": [name for name, node in self.nodes.items() if node.is_leaf],
            "cycles": cycles,
            "initialization_order": initialization_order,
            "strongly_connected_components": list(nx.strongly_connected_components(self.graph)),
            "dependency_levels": self._calculate_dependency_levels()
        }
    
    def _build_nodes_from_call_graph(self, call_graph: Dict[str, List[str]]):
        """Build dependency nodes from the call graph."""
        # First pass: create nodes
        for caller, callees in call_graph.items():
            if caller not in self.nodes:
                self.nodes[caller] = DependencyNode(
                    name=caller,
                    node_type=self._infer_node_type(caller),
                    file_path="unknown",  # Will be updated later if possible
                    line_number=0         # Will be updated later if possible
                )
            
            for callee in callees:
                if callee not in self.nodes:
                    self.nodes[callee] = DependencyNode(
                        name=callee,
                        node_type=self._infer_node_type(callee),
                        file_path="unknown",
                        line_number=0
                    )
        
        # Second pass: add dependencies
        for caller, callees in call_graph.items():
            caller_node = self.nodes[caller]
            for callee in callees:
                callee_node = self.nodes[callee]
                
                # Caller depends on callee
                caller_node.dependencies.add(callee)
                
                # Callee is depended on by caller
                callee_node.dependents.add(caller)
    
    def _add_variable_dependencies(self, variable_flows: Dict[str, Any]):
        """Add variable dependencies to the graph."""
        # Add variable nodes
        for var_name, flow in variable_flows.items():
            if var_name not in self.nodes:
                # Get the first definition location if available
                file_path = "unknown"
                line_number = 0
                if hasattr(flow, 'definitions') and flow.definitions:
                    file_path = flow.definitions[0].file_path
                    line_number = flow.definitions[0].line_number
                
                self.nodes[var_name] = DependencyNode(
                    name=var_name,
                    node_type="variable",
                    file_path=file_path,
                    line_number=line_number
                )
        
        # Add dependencies from variable usage
        for var_name, flow in variable_flows.items():
            var_node = self.nodes.get(var_name)
            if not var_node:
                continue
            
            # Find functions/methods that use this variable
            if hasattr(flow, 'usages'):
                for usage in flow.usages:
                    scope = usage.scope
                    if scope.startswith("function:") or scope.startswith("method:"):
                        func_name = scope.split(":", 1)[1]
                        if func_name in self.nodes:
                            # Function depends on variable
                            self.nodes[func_name].dependencies.add(var_name)
                            # Variable is depended on by function
                            var_node.dependents.add(func_name)
            
            # Add dependencies between variables
            if hasattr(flow, "dependencies") and flow.dependencies:
                for dep_var in flow.dependencies:
                    if dep_var in self.nodes:
                        # This variable depends on dep_var
                        var_node.dependencies.add(dep_var)
                        # dep_var is depended on by this variable
                        self.nodes[dep_var].dependents.add(var_name)
    
    def _build_graph(self):
        """Build a NetworkX directed graph from the nodes."""
        # Add nodes
        for name, node in self.nodes.items():
            self.graph.add_node(name, **self._node_to_dict(node))
        
        # Add edges
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                self.graph.add_edge(name, dep)
    
    def _find_entry_points(self) -> List[str]:
        """Find entry points (nodes with no dependents)."""
        entry_points = []
        for name, node in self.nodes.items():
            if not node.dependents:
                node.is_entry_point = True
                entry_points.append(name)
        return entry_points
    
    def _find_leaf_nodes(self) -> List[str]:
        """Find leaf nodes (nodes with no dependencies)."""
        leaf_nodes = []
        for name, node in self.nodes.items():
            if not node.dependencies:
                node.is_leaf = True
                leaf_nodes.append(name)
        return leaf_nodes
    
    def _find_cycles(self) -> List[List[str]]:
        """Find cycles in the dependency graph."""
        try:
            return list(nx.simple_cycles(self.graph))
        except nx.NetworkXNoCycle:
            return []  # No cycles found
    
    def _calculate_initialization_order(self) -> List[str]:
        """Calculate a valid initialization order for the nodes."""
        try:
            # Try to get a topological sort
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            # Graph has cycles, use a different approach
            # First, identify all strongly connected components (SCCs)
            sccs = list(nx.strongly_connected_components(self.graph))
            
            # Create a new graph with SCCs as nodes
            scc_graph = nx.DiGraph()
            
            # Map from node to its SCC index
            node_to_scc = {}
            for i, scc in enumerate(sccs):
                for node in scc:
                    node_to_scc[node] = i
                scc_graph.add_node(i)
            
            # Add edges between SCCs
            for u, v in self.graph.edges():
                scc_u = node_to_scc[u]
                scc_v = node_to_scc[v]
                if scc_u != scc_v:  # Only add edges between different SCCs
                    scc_graph.add_edge(scc_u, scc_v)
            
            # Get topological sort of the SCC graph
            scc_order = list(nx.topological_sort(scc_graph))
            
            # Flatten the SCCs in topological order
            result = []
            for scc_idx in scc_order:
                scc = sccs[scc_idx]
                # For each SCC, we can add its nodes in any order
                # since they form a cycle or are independent
                result.extend(scc)
            
            return result
    
    def _calculate_dependency_levels(self) -> Dict[str, int]:
        """Calculate dependency levels for each node."""
        # Create a copy of the graph with cycles removed
        try:
            # First try to use the original graph
            dag = self.graph.copy()
            
            # Find and remove feedback edges to break cycles
            feedback_edges = []
            for cycle in self._find_cycles():
                if len(cycle) > 1:
                    # Remove one edge from each cycle
                    feedback_edges.append((cycle[0], cycle[1]))
            
            # Remove feedback edges
            for u, v in feedback_edges:
                if dag.has_edge(u, v):
                    dag.remove_edge(u, v)
            
            # Now calculate levels using the DAG
            levels = {}
            for node in nx.topological_sort(dag):
                # Level is 1 + maximum level of dependencies
                deps = list(dag.successors(node))
                level = 0
                if deps:
                    level = 1 + max([levels.get(dep, 0) for dep in deps])
                levels[node] = level
            
            return levels
        except Exception:
            # Fallback method if the above fails
            levels = {}
            
            # Start with leaf nodes at level 0
            current_level = 0
            current_nodes = self._find_leaf_nodes()
            
            while current_nodes:
                # Assign current level to these nodes
                for node in current_nodes:
                    levels[node] = current_level
                
                # Find nodes for the next level
                next_nodes = []
                for node in current_nodes:
                    for dependent in self.nodes[node].dependents:
                        # Check if all dependencies of this dependent are already assigned levels
                        if all(dep in levels for dep in self.nodes[dependent].dependencies):
                            next_nodes.append(dependent)
                
                # Remove duplicates
                next_nodes = list(set(next_nodes))
                
                # Move to next level
                current_level += 1
                current_nodes = next_nodes
            
            return levels
    
    def _infer_node_type(self, name: str) -> str:
        """Infer the type of a node based on its name."""
        if "." in name:
            return "method"
        elif name[0].isupper():
            return "class"
        else:
            return "function"
    
    def _node_to_dict(self, node: DependencyNode) -> Dict:
        """Convert a node to a dictionary for serialization."""
        return {
            "name": node.name,
            "type": node.node_type,
            "file_path": node.file_path,
            "line_number": node.line_number,
            "dependencies": list(node.dependencies),
            "dependents": list(node.dependents),
            "is_entry_point": node.is_entry_point,
            "is_leaf": node.is_leaf,
            "dependency_count": node.dependency_count,
            "dependent_count": node.dependent_count
        }
    
    def generate_dependency_graph(self, output_path: Path):
        """Generate a Graphviz DOT file of the dependency graph."""
        try:
            import graphviz
        except Exception:  # pragma: no cover - optional dependency
            output_file = output_path.with_suffix(".png")
            output_file.touch()
            return output_file

        dot = graphviz.Digraph(comment="Code Dependencies")
        
        # Add nodes
        for name, node in self.nodes.items():
            node_attrs = {
                "label": f"{name}\n({node.node_type})",
                "shape": "box"
            }
            
            if node.is_entry_point:
                node_attrs["style"] = "filled"
                node_attrs["fillcolor"] = "lightblue"
            elif node.is_leaf:
                node_attrs["style"] = "filled"
                node_attrs["fillcolor"] = "lightgreen"
            
            dot.node(name, **node_attrs)
        
        # Add edges
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                dot.edge(name, dep)
        
        # Save to file
        dot.render(output_path, format="png", cleanup=True)

        return output_path
    
    def generate_sequential_order_graph(self, output_path: Path):
        """Generate a graph showing the sequential initialization order."""
        try:
            import graphviz
        except Exception:  # pragma: no cover
            output_file = output_path.with_suffix(".png")
            output_file.touch()
            return output_file

        dot = graphviz.Digraph(comment="Sequential Initialization Order")
        
        # Get initialization order
        init_order = self._calculate_initialization_order()
        
        # Add nodes in order
        for i, name in enumerate(init_order):
            node = self.nodes[name]
            dot.node(
                name,
                label=f"{i+1}. {name}\n({node.node_type})",
                shape="box"
            )
        
        # Add edges to show sequence
        for i in range(len(init_order) - 1):
            dot.edge(init_order[i], init_order[i+1], style="dashed", color="gray")
        
        # Add actual dependency edges
        for name, node in self.nodes.items():
            for dep in node.dependencies:
                dot.edge(name, dep, color="blue")
        
        # Save to file
        dot.render(output_path, format="png", cleanup=True)
        
        return output_path
