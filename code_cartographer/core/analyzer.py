"""
Enhanced Code Analysis Engine
============================
A sophisticated static analysis engine that performs deep inspection of Python codebases
to generate rich structural and qualitative insights, with bidirectional call graph and orphan detection.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import hashlib
import json
import os
import re
import sys
import textwrap
import tokenize
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import cached_property
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional advanced metrics
try:
    from radon.complexity import cc_visit  # type: ignore
    from radon.metrics import mi_visit  # type: ignore

    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


@dataclass
class ComplexityMetrics:
    cyclomatic: Optional[int] = None
    maintainability_index: Optional[float] = None
    risk_flag: bool = False  # True if exceeds thresholds


@dataclass
class DefinitionMetadata:
    name: str
    category: str  # function, class, method
    start_line: int
    end_line: int
    line_count: int
    is_async: bool
    decorators: List[str]
    docstring: Optional[str]
    outbound_calls: Set[str]
    source_text: str
    source_hash: str
    has_type_hints: bool
    metrics: ComplexityMetrics
    inbound_calls: Set[str] = field(default_factory=set)  # Who calls this definition
    is_orphan: bool = True  # Default to True, will be updated during analysis
    prerequisites: Set[str] = field(default_factory=set)  # Definitions this depends on
    variable_uses: Set[str] = field(default_factory=set)  # Variables used by this definition


@dataclass
class VariableMetadata:
    name: str
    defined_in: str  # File path or definition name
    line_number: int
    is_global: bool = False
    is_parameter: bool = False
    is_imported: bool = False
    type_hint: Optional[str] = None
    used_in: Set[str] = field(default_factory=set)  # Definitions that use this variable
    is_orphan: bool = True  # Default to True, will be updated during analysis


@dataclass
class FileMetadata:
    path: str
    imports: List[str]
    internal_imports: List[str]
    stdlib_imports: List[str]
    comprehension_patterns: Dict[str, int]
    loop_count: int
    branch_count: int
    exception_blocks: int
    exception_raises: int
    regex_patterns: int
    generator_count: int
    uses_async: bool
    oop_patterns: Dict[str, bool]
    concurrent_libs: Set[str]
    module_docstring: Optional[str]
    comment_count: int
    definitions: List[DefinitionMetadata]
    metrics: ComplexityMetrics
    declared_variables: List[VariableMetadata] = field(default_factory=list)
    orphaned_code: List[str] = field(default_factory=list)  # Definitions never called


class VariableTracker(ast.NodeVisitor):
    """Track variable definitions and usages."""
    
    def __init__(self, source: str, file_path: str):
        self.source = source
        self.file_path = file_path
        self.variables: List[VariableMetadata] = []
        self.current_definition: Optional[str] = None
        self.variable_uses: Dict[str, Set[str]] = defaultdict(set)
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            # Variable definition
            var_meta = VariableMetadata(
                name=node.id,
                defined_in=self.current_definition or self.file_path,
                line_number=node.lineno,
                is_global=self.current_definition is None,
            )
            self.variables.append(var_meta)
        elif isinstance(node.ctx, ast.Load):
            # Variable usage
            if self.current_definition:
                self.variable_uses[node.id].add(self.current_definition)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        old_definition = self.current_definition
        self.current_definition = node.name
        
        # Track parameters as variables
        for arg in node.args.args:
            var_meta = VariableMetadata(
                name=arg.arg,
                defined_in=self.current_definition,
                line_number=node.lineno,
                is_parameter=True,
                type_hint=ast.unparse(arg.annotation) if arg.annotation else None
            )
            self.variables.append(var_meta)
            
        self.generic_visit(node)
        self.current_definition = old_definition
        
    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)  # Reuse the same logic
        
    def visit_ClassDef(self, node):
        old_definition = self.current_definition
        self.current_definition = node.name
        self.generic_visit(node)
        self.current_definition = old_definition
        
    def visit_Import(self, node):
        for alias in node.names:
            var_meta = VariableMetadata(
                name=alias.asname or alias.name,
                defined_in=self.file_path,
                line_number=node.lineno,
                is_global=True,
                is_imported=True
            )
            self.variables.append(var_meta)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        for alias in node.names:
            var_meta = VariableMetadata(
                name=alias.asname or alias.name,
                defined_in=self.file_path,
                line_number=node.lineno,
                is_global=True,
                is_imported=True
            )
            self.variables.append(var_meta)
        self.generic_visit(node)
        
    def finalize(self):
        """Update variable usage information."""
        for var in self.variables:
            var.used_in = self.variable_uses.get(var.name, set())
            var.is_orphan = len(var.used_in) == 0


class CodeInspector(ast.NodeVisitor):
    """Advanced AST visitor for deep code analysis."""

    def __init__(self, source: str, project_root: Path):
        self._src = source
        self._lines = source.splitlines()
        self.project_root = project_root

        # Analysis containers
        self.imports: List[str] = []
        self.internal_imports: List[str] = []
        self.variables: List[str] = []
        self.comprehension_patterns: Dict[str, int] = defaultdict(int)
        self.loop_count = 0
        self.branch_count = 0
        self.exception_blocks = 0
        self.exception_raises = 0
        self.regex_patterns = 0
        self.generator_count = 0
        self.uses_async = False
        self.oop_flags = {"inheritance": False, "polymorphism": False, "dunder": False}
        self.concurrent_libs: Set[str] = set()
        self.definitions: List[DefinitionMetadata] = []
        
        # Track all function/method calls for bidirectional mapping
        self.all_calls: Dict[str, Set[str]] = defaultdict(set)
        
        # Track variable definitions and usages
        self.variable_tracker = None

        # Pre-scan for regex usage
        if re.search(r"\bimport\s+re\b", source):
            self.regex_patterns = 1

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._process_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        self._process_import(mod)
        self.generic_visit(node)

    def _process_import(self, mod_path: str):
        self.imports.append(mod_path)
        if mod_path in {
            "threading",
            "multiprocessing",
            "asyncio",
            "concurrent.futures",
        }:
            self.concurrent_libs.add(mod_path)

        # Check for internal project imports
        candidate = self.project_root / (mod_path.replace(".", os.sep) + ".py")
        if candidate.exists():
            self.internal_imports.append(str(candidate.relative_to(self.project_root)))

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.append(target.id)
        self.generic_visit(node)

    def visit_For(self, node):
        self.loop_count += 1
        self.generic_visit(node)

    visit_While = visit_For

    def visit_If(self, node):
        self.branch_count += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.exception_blocks += 1
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.exception_raises += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.comprehension_patterns["list"] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.comprehension_patterns["dict"] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.comprehension_patterns["set"] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.comprehension_patterns["generator"] += 1
        self.generator_count += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_definition(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self.uses_async = True
        self._process_definition(node, "function", is_async=True)

    def visit_ClassDef(self, node):
        self.oop_flags["inheritance"] |= bool(node.bases)
        self._process_definition(node, "class")

        # Process methods
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._process_definition(
                    child,
                    "method",
                    parent=node.name,
                    is_async=isinstance(child, ast.AsyncFunctionDef),
                )
        self.generic_visit(node)

    def _extract_segment(self, node):
        """Extract source code segment with error handling."""
        try:
            return ast.get_source_segment(self._src, node) or ""
        except Exception:
            return "\n".join(self._lines[node.lineno - 1 : node.end_lineno])

    def _extract_calls(self, node):
        """Extract all function/method calls from an AST node."""
        calls = set()
        caller_name = getattr(node, "name", "<unknown>")
        
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute):
                    call_name = n.func.attr
                    calls.add(call_name)
                    # Record this call for bidirectional mapping
                    self.all_calls[call_name].add(caller_name)
                elif isinstance(n.func, ast.Name):
                    call_name = n.func.id
                    calls.add(call_name)
                    # Record this call for bidirectional mapping
                    self.all_calls[call_name].add(caller_name)
        return calls

    def _compute_metrics(self, node) -> ComplexityMetrics:
        """Compute complexity metrics if radon is available."""
        if not _HAS_METRICS:
            return ComplexityMetrics()

        text = ast.unparse(node)
        cc_blocks = cc_visit(text)
        cyclo = cc_blocks[0].complexity if cc_blocks else None
        mi = mi_visit(text, multi=True)
        risk = bool((cyclo and cyclo > 10) or (mi and mi < 65))

        return ComplexityMetrics(
            cyclomatic=cyclo, maintainability_index=mi, risk_flag=risk
        )

    def _process_definition(self, node, category, parent=None, is_async=False):
        """Process a function/class definition with full metadata extraction."""
        source = self._extract_segment(node)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        docstring = ast.get_docstring(node)

        # Check for type hints
        try:
            has_hints = bool(getattr(node, "returns", None)) or any(
                isinstance(a.annotation, ast.AST)
                for a in getattr(node.args, "args", [])
            )
        except Exception:
            has_hints = False

        metrics = self._compute_metrics(node)
        name = getattr(node, "name", "<lambda>")

        if parent and category == "method":
            name = f"{parent}.{name}"

        # Extract outbound calls
        outbound_calls = self._extract_calls(node)
        
        # Extract variable uses
        variable_uses = set()
        for var_name in self.variable_tracker.variable_uses:
            if name in self.variable_tracker.variable_uses[var_name]:
                variable_uses.add(var_name)

        self.definitions.append(
            DefinitionMetadata(
                name=name,
                category=category,
                start_line=node.lineno,
                end_line=node.end_lineno,
                line_count=node.end_lineno - node.lineno + 1,
                is_async=is_async,
                decorators=[
                    ast.unparse(d) for d in getattr(node, "decorator_list", [])
                ],
                docstring=docstring,
                outbound_calls=outbound_calls,
                source_text=source,
                source_hash=source_hash,
                has_type_hints=has_hints,
                metrics=metrics,
                variable_uses=variable_uses
            )
        )


class ProjectAnalyzer:
    """Orchestrates full-project code analysis."""

    def __init__(self, root: Path, exclude_patterns: List[str] | None = None):
        self.root = root
        # Preserve provided patterns for tests
        self.exclude_patterns: List[str] = exclude_patterns or []
        # Compiled regex patterns used internally
        self._exclude_regex: List[re.Pattern] = []

        for pattern in self.exclude_patterns:
            try:
                self._exclude_regex.append(re.compile(pattern))
            except re.error:
                regex_pattern = (
                    pattern.replace(".", "\\.")
                    .replace("*", ".*")
                    .replace("?", ".")
                )
                self._exclude_regex.append(re.compile(regex_pattern))
        
        self.file_data: List[FileMetadata] = []
        self.definition_index: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
        self.dependency_graph: Set[Tuple[str, str]] = set()
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.variable_index: Dict[str, List[VariableMetadata]] = defaultdict(list)

    def execute(self):
        """Run full project analysis."""
        for py_file in self._find_python_files():
            try:
                self._analyze_file(py_file)
            except SyntaxError as err:
                print(f"[WARN] {py_file}: {err}")
            except Exception as exc:
                print(f"[ERR ] Failed on {py_file}: {exc}")

        # Post-process to identify orphans and build bidirectional call graph
        self._build_call_graphs()
        self._identify_orphans()
        self._analyze_prerequisites()

        files = []
        for fs in self.file_data:
            fd = asdict(fs)
            fd["functions"] = [d.name for d in fs.definitions if d.category == "function"] # Extract function names
            files.append(fd)

        return {
            "files": files,
            "variants": self._generate_variant_report(),
            "dependencies": list(self.dependency_graph),
            "call_graph": {k: list(v) for k, v in self.call_graph.items()},
            "reverse_call_graph": {k: list(v) for k, v in self.reverse_call_graph.items()},
            "orphans": self._get_orphans(),
            "variables": {k: [asdict(v) for v in vars] for k, vars in self.variable_index.items()},
            # Add backward compatibility for tests
            "functions": self._get_function_list()
        }

    def _find_python_files(self):
        """Find all Python files, respecting exclusion patterns."""
        for p in self.root.rglob("*.py"):
            if any(rx.search(str(p)) for rx in self._exclude_regex):
                continue
            yield p

    def _analyze_file(self, path: Path):
        """Perform deep analysis of a single Python file."""
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        
        # First track variables
        variable_tracker = VariableTracker(source, str(path.relative_to(self.root)))
        variable_tracker.visit(tree)
        variable_tracker.finalize()
        
        # Store variables in the index
        for var in variable_tracker.variables:
            self.variable_index[var.name].append(var)
        
        # Then do full code inspection
        inspector = CodeInspector(source, self.root)
        inspector.variable_tracker = variable_tracker
        inspector.visit(tree)
        
        # Extract module docstring
        module_docstring = ast.get_docstring(tree)
        
        # Count comments
        comment_count = 0
        try:
            with tokenize.open(path) as f:
                tokens = tokenize.generate_tokens(f.readline)
                comment_count = sum(1 for tok in tokens if tok.type == tokenize.COMMENT)
        except Exception:
            pass
        
        # Compute file-level metrics
        metrics = ComplexityMetrics()
        if _HAS_METRICS:
            try:
                mi = mi_visit(source, multi=True)
                metrics.maintainability_index = mi
                metrics.risk_flag = mi < 65
            except Exception:
                pass
        
        # Create file metadata
        rel_path = str(path.relative_to(self.root))
        file_meta = FileMetadata(
            path=rel_path,
            imports=inspector.imports,
            internal_imports=inspector.internal_imports,
            stdlib_imports=[i for i in inspector.imports if i in sys.stdlib_module_names],
            comprehension_patterns=dict(inspector.comprehension_patterns),
            loop_count=inspector.loop_count,
            branch_count=inspector.branch_count,
            exception_blocks=inspector.exception_blocks,
            exception_raises=inspector.exception_raises,
            regex_patterns=inspector.regex_patterns,
            generator_count=inspector.generator_count,
            uses_async=inspector.uses_async,
            oop_patterns=dict(inspector.oop_flags),
            concurrent_libs=inspector.concurrent_libs,
            module_docstring=module_docstring,
            comment_count=comment_count,
            definitions=inspector.definitions,
            metrics=metrics,
            declared_variables=[v for v in variable_tracker.variables if not v.is_imported]
        )
        
        self.file_data.append(file_meta)
        
        # Index definitions for cross-referencing
        for defn in inspector.definitions:
            self.definition_index[defn.name][rel_path] = (defn.category, defn.source_hash)
            
            # Record outbound calls for dependency graph
            for call in defn.outbound_calls:
                self.dependency_graph.add((defn.name, call))
                self.call_graph[defn.name].add(call)

    def _build_call_graphs(self):
        """Build bidirectional call graphs."""
        # Forward call graph is already built during analysis
        
        # Build reverse call graph
        for caller, callees in self.call_graph.items():
            for callee in callees:
                self.reverse_call_graph[callee].add(caller)

    def _identify_orphans(self):
        """Identify orphaned functions and variables."""
        # Mark functions/methods that are called
        for file_meta in self.file_data:
            for defn in file_meta.definitions:
                if defn.name in self.reverse_call_graph:
                    defn.is_orphan = False
                    
                    # Also update inbound calls
                    defn.inbound_calls = self.reverse_call_graph[defn.name]
        
        # Identify orphaned variables
        for var_list in self.variable_index.values():
            for var in var_list:
                var.is_orphan = len(var.used_in) == 0
        
        # Update file metadata with orphaned code
        for file_meta in self.file_data:
            file_meta.orphaned_code = [
                d.name for d in file_meta.definitions if d.is_orphan
            ]

    def _analyze_prerequisites(self):
        """Analyze prerequisites for each definition."""
        for file_meta in self.file_data:
            for defn in file_meta.definitions:
                # Add outbound calls as prerequisites
                defn.prerequisites.update(defn.outbound_calls)
                
                # Add variable dependencies
                for var_name in defn.variable_uses:
                    if var_name in self.variable_index:
                        for var in self.variable_index[var_name]:
                            if var.defined_in != defn.name:  # Avoid self-reference
                                defn.prerequisites.add(var.defined_in)

    def _generate_variant_report(self):
        """Generate a report of code variants."""
        # This is a placeholder for the variant analysis
        # In a real implementation, this would use the variant_analyzer module
        return {}

    def _get_orphans(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get structured information on orphaned code elements."""
        result = {"functions": [], "classes": [], "variables": []}
        for file_meta in self.file_data:
            for defn in file_meta.definitions:
                if defn.is_orphan:
                    entry = {"name": defn.name, "file": file_meta.path, "line": defn.start_line}
                    if defn.category in ["function", "method"]:
                        result["functions"].append(entry)
                    elif defn.category == "class":
                        result["classes"].append(entry)
            for var in file_meta.declared_variables:
                if var.is_orphan:
                    result["variables"].append({"name": var.name, "file": file_meta.path, "line": var.line_number})
        return result
    
    def _get_function_list(self):
        """Get a list of all functions for backward compatibility with tests."""
        result = []
        for file_meta in self.file_data:
            functions = [d.name for d in file_meta.definitions if d.category == "function"]
            result.append({"path": file_meta.path, "functions": functions})
        return result


class CodeAnalyzer:
    """Main entry point for code analysis."""
    
    def __init__(self, project_path: str, output_dir: str = None, exclude: List[str] = None):
        self.project_path = Path(project_path).absolute()
        self.output_dir = Path(output_dir or self.project_path / "analysis_output")
        self.exclude = exclude or ["__pycache__", "*.pyc", "venv", ".git", ".venv"]
        
    def analyze(self):
        """Run the analysis and generate reports."""
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Run the analysis
        analyzer = ProjectAnalyzer(self.project_path, self.exclude)
        results = analyzer.execute()
        
        # Save raw results
        with open(self.output_dir / "analysis_data.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
