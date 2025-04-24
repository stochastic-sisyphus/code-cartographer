"""
Advanced Code Analysis Engine
============================
A sophisticated static analysis engine that performs deep inspection of Python codebases
to generate rich structural and qualitative insights.
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
from dataclasses import dataclass, asdict
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


@dataclass
class FileMetadata:
    path: str
    imports: List[str]
    internal_imports: List[str]
    stdlib_imports: List[str]
    declared_variables: List[str]
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
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute):
                    calls.add(n.func.attr)
                elif isinstance(n.func, ast.Name):
                    calls.add(n.func.id)
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
                outbound_calls=self._extract_calls(node),
                source_text=source,
                source_hash=source_hash,
                has_type_hints=has_hints,
                metrics=metrics,
            )
        )


class ProjectAnalyzer:
    """Orchestrates full-project code analysis."""

    def __init__(self, root: Path, exclude_patterns: List[str]):
        self.root = root
        self.exclude = [re.compile(p) for p in exclude_patterns]
        self.file_data: List[FileMetadata] = []
        self.definition_index: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)
        self.dependency_graph: Set[Tuple[str, str]] = set()

    def execute(self):
        """Run full project analysis."""
        for py_file in self._find_python_files():
            try:
                self._analyze_file(py_file)
            except SyntaxError as err:
                print(f"[WARN] {py_file}: {err}")
            except Exception as exc:
                print(f"[ERR ] Failed on {py_file}: {exc}")

        return {
            "files": [asdict(fs) for fs in self.file_data],
            "variants": self._generate_variant_report(),
            "dependencies": list(self.dependency_graph),
        }

    def _find_python_files(self):
        """Find all Python files, respecting exclusion patterns."""
        for p in self.root.rglob("*.py"):
            if any(rx.search(str(p)) for rx in self.exclude):
                continue
            yield p

    def _analyze_file(self, path: Path):
        """Perform deep analysis of a single Python file."""
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        inspector = CodeInspector(source, self.root)
        inspector.visit(tree)

        # Count comments
        comments = sum(
            tok.type == tokenize.COMMENT
            for tok in tokenize.generate_tokens(StringIO(source).readline)
        )

        metrics = inspector._compute_metrics(tree)

        metadata = FileMetadata(
            path=str(path.relative_to(self.root)),
            imports=inspector.imports,
            internal_imports=inspector.internal_imports,
            stdlib_imports=[
                imp for imp in inspector.imports if imp in sys.stdlib_module_names
            ],
            declared_variables=sorted(set(inspector.variables)),
            comprehension_patterns=dict(inspector.comprehension_patterns),
            loop_count=inspector.loop_count,
            branch_count=inspector.branch_count,
            exception_blocks=inspector.exception_blocks,
            exception_raises=inspector.exception_raises,
            regex_patterns=inspector.regex_patterns,
            generator_count=inspector.generator_count,
            uses_async=inspector.uses_async,
            oop_patterns=inspector.oop_flags,
            concurrent_libs=inspector.concurrent_libs,
            module_docstring=ast.get_docstring(tree),
            comment_count=comments,
            definitions=inspector.definitions,
            metrics=metrics,
        )
        self.file_data.append(metadata)

        # Build indices
        for d in inspector.definitions:
            self.definition_index[d.name][d.source_hash] = (
                metadata.path,
                d.source_text,
            )

        # Record dependencies
        for dep in inspector.internal_imports:
            self.dependency_graph.add((metadata.path, dep))

    def _generate_variant_report(self):
        """Generate detailed variant analysis with LLM-ready refactor prompts."""
        variants = {}
        for name, implementations in self.definition_index.items():
            if len(implementations) <= 1:
                continue

            entries = []
            hashes = list(implementations.keys())
            base_hash = hashes[0]
            base_source = implementations[base_hash][1]

            for hash_val, (path, source) in implementations.items():
                diff = (
                    list(
                        difflib.unified_diff(
                            base_source.splitlines(), source.splitlines(), lineterm=""
                        )
                    )
                    if hash_val != base_hash
                    else []
                )

                entries.append({"hash": hash_val, "path": path, "diff_from_base": diff})

            # Generate LLM-ready refactoring prompt
            prompt = textwrap.dedent(
                f"""
                Multiple implementations of `{name}` detected across the codebase.
                Recommended action: Refactor into a single canonical implementation.
                Strategy:
                1. Review all variants and their differences
                2. Identify superset of functionality
                3. Merge implementations preserving all edge cases
                4. Add comprehensive tests covering all use cases
                5. Update all call sites to use new implementation
            """
            ).strip()

            variants[name] = {"implementations": entries, "refactor_prompt": prompt}

        return variants


def generate_markdown(analysis: Dict[str, Any], output_path: Path):
    """Generate detailed Markdown report from analysis results."""
    lines = ["# Code Analysis Report\n"]

    for fs in analysis["files"]:
        lines.extend(
            [
                f"## {fs['path']}",
                f"* Maintainability Index: {fs['metrics']['maintainability_index']}",
                f"* Cyclomatic Complexity: {fs['metrics']['cyclomatic']}",
                "\n### Definitions\n",
            ]
        )

        for d in fs["definitions"]:
            risk = "⚠️" if d["metrics"]["risk_flag"] else ""
            lines.append(
                f"* `{d['category']}` **{d['name']}** "
                f"({d['line_count']} lines) {risk}"
            )

    output_path.write_text("\n".join(lines))


def generate_dependency_graph(edges: List[Tuple[str, str]], output_path: Path):
    """Generate Graphviz DOT file of project dependencies."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("digraph project_dependencies {\n")
        f.write("  node [style=filled,fillcolor=lightgray];\n")
        f.write("  edge [color=navy];\n\n")

        for src, dst in edges:
            f.write(f'  "{src}" -> "{dst}";\n')

        f.write("}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Python codebase analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        required=True,
        help="Root directory of the project to analyze",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("code_analysis.json"),
        help="Path for JSON output",
    )

    parser.add_argument(
        "-e",
        "--exclude",
        nargs="*",
        default=[],
        help="Regex patterns for files/dirs to exclude",
    )

    parser.add_argument(
        "--markdown", type=Path, help="Generate Markdown report at specified path"
    )

    parser.add_argument(
        "--graphviz",
        type=Path,
        help="Generate Graphviz dependency graph at specified path",
    )

    parser.add_argument("--no-git", action="store_true", help="Skip git SHA tagging")

    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level")

    args = parser.parse_args()

    # Run analysis
    analyzer = ProjectAnalyzer(args.dir, args.exclude)
    analysis = analyzer.execute()

    # Add git context if available
    if not args.no_git and (args.dir / ".git").exists():
        import subprocess

        with contextlib.suppress(Exception):
            sha = subprocess.check_output(
                ["git", "-C", str(args.dir), "rev-parse", "HEAD"], text=True
            ).strip()
            analysis["git_revision"] = sha

    # Write outputs
    args.output.write_text(
        json.dumps(
            analysis,
            indent=args.indent,
            default=lambda o: (
                asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)
            ),
        )
    )
    print(f"[JSON] {args.output.resolve()}")

    if args.markdown:
        generate_markdown(analysis, args.markdown)
        print(f"[MD  ] {args.markdown.resolve()}")

    if args.graphviz:
        generate_dependency_graph(analysis["dependencies"], args.graphviz)
        print(f"[DOT ] {args.graphviz.resolve()}")


if __name__ == "__main__":
    main()
