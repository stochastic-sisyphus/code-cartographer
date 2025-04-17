#!/usr/bin/env python3
"""
Deep Code Analyzer – v2
======================
A one‑stop static‑analysis powerhouse that crawls a Python project tree and
produces a *multi‑artifact* report:

* **deep_code_summary.json** – richly structured, machine‑readable summary
* **deep_code_summary.md**   – human‑friendly Markdown digest (optional)
* **dependencies.dot**       – Graphviz DOT of intra‑project import graph (optional)

Key capabilities
----------------
1.  File‑level & definition‑level metadata (functions, classes, methods) with
    full source, SHA‑256, radon complexity, decorators, async flags, calls, etc.
2.  Duplicate/variant detection *plus* inline diffs between variants.
3.  Dependency graph: maps internal imports between project files.
4.  Complexity thresholds – flags "at‑risk" hotspots (MI < 65 or CC > 10).
5.  Automated *refactor prompts* you can feed directly to an LLM.
6.  CLI flags for exclusion patterns, Markdown & Graphviz output, Git context,
    JSON indentation, and diff granularity.

Why JSON *and* Markdown?
—————————
JSON is ideal for downstream tooling; Markdown is great for a quick skim or PR
review.  Both are generated from the same canonical data so they never drift.

Usage
-----
```bash
python deep_code_analyzer.py \
  -d /path/to/project \
  --markdown report.md \
  --graphviz deps.dot \
  --exclude "tests/.* " "build/.*"
```

Run `-h` for the full option list.
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
from dataclasses import dataclass, asdict, field
from functools import cached_property
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional dependencies -------------------------------------------------------
try:
    from radon.complexity import cc_visit  # type: ignore
    from radon.metrics import mi_visit  # type: ignore
    _HAS_RADON = True
except ImportError:
    _HAS_RADON = False

@dataclass
class Complexity:
    cyclomatic: Optional[int] = None
    maintainability_index: Optional[float] = None
    at_risk: bool = False  # flagged if exceeds threshold

@dataclass
class DefinitionSummary:
    name: str
    kind: str
    lineno: int
    end_lineno: int
    line_count: int
    is_async: bool
    decorators: List[str]
    docstring: Optional[str]
    calls: Set[str]
    source: str
    sha256: str
    type_hints: bool
    complexity: Complexity

@dataclass
class FileSummary:
    path: str
    imports: List[str]
    internal_imports: List[str]
    stdlib_imports: List[str]
    variables: List[str]
    comprehension_counts: Dict[str, int]
    loops: int
    conditionals: int
    try_blocks: int
    raises: int
    regex_usage: int
    generators: int
    asyncio_usage: bool
    ood_principles: Dict[str, bool]
    concurrency_libs: Set[str]
    file_docstring: Optional[str]
    comments: int
    definitions: List[DefinitionSummary]
    complexity: Complexity

# -----------------------------------------------------------------------------
# AST Visitor
# -----------------------------------------------------------------------------
class _Analyzer(ast.NodeVisitor):
    def __init__(self, source: str, project_root: Path):
        self._src = source
        self._lines = source.splitlines()
        self.project_root = project_root
        # containers
        self.imports: List[str] = []
        self.internal_imports: List[str] = []
        self.variables: List[str] = []
        self.comprehension_counts: Dict[str, int] = defaultdict(int)
        self.loops = 0
        self.conditionals = 0
        self.try_blocks = 0
        self.raises = 0
        self.regex_usage = 0
        self.generators = 0
        self.asyncio_usage = False
        self.ood_flags = {"inheritance": False, "polymorphism": False, "dunder": False}
        self.conc_libs: Set[str] = set()
        self.definitions: List[DefinitionSummary] = []
        if re.search(r"\bimport\s+re\b", source):
            self.regex_usage = 1

    # ---------- visit methods ----------
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._register_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mod = node.module or ""
        self._register_import(mod)
        self.generic_visit(node)

    def _register_import(self, mod_path: str):
        self.imports.append(mod_path)
        if mod_path in {
            "threading",
            "multiprocessing",
            "asyncio",
            "concurrent.futures",
        }:
            self.conc_libs.add(mod_path)
        # internal dependency?
        candidate = self.project_root / (mod_path.replace(".", os.sep) + ".py")
        if candidate.exists():
            self.internal_imports.append(str(candidate.relative_to(self.project_root)))

    def visit_Assign(self, node):
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                self.variables.append(tgt.id)
        self.generic_visit(node)

    def visit_For(self, node):
        self.loops += 1
        self.generic_visit(node)

    visit_While = visit_For

    def visit_If(self, node):
        self.conditionals += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.try_blocks += 1
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.raises += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.comprehension_counts["listcomp"] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.comprehension_counts["dictcomp"] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.comprehension_counts["setcomp"] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.comprehension_counts["genexp"] += 1
        self.generators += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._add_def(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self.asyncio_usage = True
        self._add_def(node, "function", is_async=True)

    def visit_ClassDef(self, node):
        self.ood_flags["inheritance"] |= bool(node.bases)
        self._add_def(node, "class")
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._add_def(child, "method", parent=node.name, is_async=isinstance(child, ast.AsyncFunctionDef))
        self.generic_visit(node)

    # ---------- helpers ----------
    def _segment(self, node):
        try:
            return ast.get_source_segment(self._src, node) or ""
        except Exception:
            return "\n".join(self._lines[node.lineno-1: node.end_lineno])

    def _calls(self, node):
        calls = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute):
                    calls.add(n.func.attr)
                elif isinstance(n.func, ast.Name):
                    calls.add(n.func.id)
        return calls

    def _complexity(self, node) -> Complexity:
        if not _HAS_RADON:
            return Complexity()
        text = ast.unparse(node)
        cc_blocks = cc_visit(text)
        cyclo = cc_blocks[0].complexity if cc_blocks else None
        mi = mi_visit(text, multi=True)
        at_risk = bool((cyclo and cyclo > 10) or (mi and mi < 65))
        return Complexity(cyclomatic=cyclo, maintainability_index=mi, at_risk=at_risk)

    def _add_def(self, node, kind, parent=None, is_async=False):
        src = self._segment(node)
        sha = hashlib.sha256(src.encode()).hexdigest()
        doc = ast.get_docstring(node)
        try:
            hints = bool(getattr(node, "returns", None)) or any(
                isinstance(a.annotation, ast.AST) for a in getattr(node.args, "args", [])
            )
        except Exception:
            hints = False
        comp = self._complexity(node)
        name = getattr(node, "name", "<lambda>")
        if parent and kind == "method":
            name = f"{parent}.{name}"
        self.definitions.append(
            DefinitionSummary(
                name=name,
                kind=kind,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                line_count=node.end_lineno - node.lineno + 1,
                is_async=is_async,
                decorators=[ast.unparse(d) for d in getattr(node, "decorator_list", [])],
                docstring=doc,
                calls=self._calls(node),
                source=src,
                sha256=sha,
                type_hints=hints,
                complexity=comp,
            )
        )

# -----------------------------------------------------------------------------
# Project Analyzer
# -----------------------------------------------------------------------------
class ProjectAnalyzer:
    def __init__(self, root: Path, exclude: List[str]):
        self.root = root
        self.exclude = [re.compile(p) for p in exclude]
        self.file_summaries: List[FileSummary] = []
        self.def_index: Dict[str, Dict[str, Tuple[str, str]]] = defaultdict(dict)  # name -> sha -> (path, src)
        self.dep_edges: Set[Tuple[str, str]] = set()

    def run(self):
        for py_file in self._py_files():
            try:
                self._analyze_file(py_file)
            except SyntaxError as err:
                print(f"[WARN] {py_file}: {err}")
            except Exception as exc:
                print(f"[ERR ] Failed on {py_file}: {exc}")
        return {
            "summary": [asdict(fs) for fs in self.file_summaries],
            "duplicates": self._duplicate_report(),
            "dependency_graph": list(self.dep_edges),
        }

    # -------------- helpers --------------
    def _py_files(self):
        for p in self.root.rglob("*.py"):
            if any(rx.search(str(p)) for rx in self.exclude):
                continue
            yield p

    def _analyze_file(self, path: Path):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        v = _Analyzer(src, self.root)
        v.visit(tree)

        comments = sum(
            tok.type == tokenize.COMMENT
            for tok in tokenize.generate_tokens(StringIO(src).readline)
        )
        comp = v._complexity(tree)
        fs = FileSummary(
            path=str(path.relative_to(self.root)),
            imports=v.imports,
            internal_imports=v.internal_imports,
            stdlib_imports=[imp for imp in v.imports if imp in sys.stdlib_module_names],
            variables=sorted(set(v.variables)),
            comprehension_counts=dict(v.comprehension_counts),
            loops=v.loops,
            conditionals=v.conditionals,
            try_blocks=v.try_blocks,
            raises=v.raises,
            regex_usage=v.regex_usage,
            generators=v.generators,
            asyncio_usage=v.asyncio_usage,
            ood_principles=v.ood_flags,
            concurrency_libs=v.conc_libs,
            file_docstring=ast.get_docstring(tree),
            comments=comments,
            definitions=v.definitions,
            complexity=comp,
        )
        self.file_summaries.append(fs)

        # index for duplicates & diffs
        for d in v.definitions:
            self.def_index[d.name][d.sha256] = (fs.path, d.source)

        # dependency graph
        for dep in v.internal_imports:
            self.dep_edges.add((fs.path, dep))

    def _duplicate_report(self):
        out = {}
        for name, variants in self.def_index.items():
            if len(variants) <= 1:
                continue
            entries = []
            hashes = list(variants.keys())
            base_sha = hashes[0]
            base_src = variants[base_sha][1]
            for sha, (path, src) in variants.items():
                diff = list(difflib.unified_diff(base_src.splitlines(), src.splitlines(), lineterm="")) if sha != base_sha else []
                entries.append({"sha256": sha, "path": path, "diff_vs_first": diff})
            # prompt for LLM
            prompt = textwrap.dedent(f"""
                The function/class `{name}` has multiple variants across the codebase.  Refactor them into a single, canonical implementation where appropriate.  Consult the diffs and choose the superset of functionality.  Ensure unit tests cover edge cases for all previous usages.
            """)
            out[name] = {"variants": entries, "refactor_prompt": prompt.strip()}
        return out

# -----------------------------------------------------------------------------
# Markdown & Graphviz helpers
# -----------------------------------------------------------------------------

def _write_markdown(report: Dict[str, Any], out_path: Path):
    lines = ["# Deep Code Summary\n"]
    for fs in report["summary"]:
        lines.extend(
            (
                f"## {fs['path']}",
                f"* MI: {fs['complexity']['maintainability_index']}  *CC avg*: {fs['complexity']['cyclomatic']}",
                "### Definitions\n",
            )
        )
        for d in fs['definitions']:
            risk = "⚠️" if d['complexity']['at_risk'] else ""
            lines.append(f"* `{d['kind']}` **{d['name']}** ({d['line_count']} lines) {risk}")
    out_path.write_text("\n".join(lines))


def _write_graphviz(edges: List[Tuple[str, str]], out_path: Path):
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("digraph dependencies {\n")
        for a, b in edges:
            fh.write(f"  \"{a}\" -> \"{b}\";\n")
        fh.write("}\n")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Deep static analyzer for Python projects")
    p.add_argument("-d", "--dir", type=Path, required=True, help="Project root directory")
    p.add_argument("-o", "--output", type=Path, default=Path("deep_code_summary.json"), help="Path for JSON output")
    p.add_argument("-e", "--exclude", nargs="*", default=[], help="Regex patterns to skip")
    p.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    p.add_argument("--graphviz", type=Path, help="Optional Graphviz .dot output path")
    p.add_argument("--no-git", action="store_true", help="Skip git SHA tagging")
    p.add_argument("--indent", type=int, default=2, help="JSON indent level")
    args = p.parse_args()

    analyzer = ProjectAnalyzer(args.dir, args.exclude)
    report = analyzer.run()

    # git context -------------------------------------------------------
    if not args.no_git and (args.dir / ".git").exists():
        import subprocess, shlex
        with contextlib.suppress(Exception):
            sha = subprocess.check_output(["git", "-C", str(args.dir), "rev-parse", "HEAD"], text=True).strip()
            report["git_head"] = sha
    args.output.write_text(json.dumps(report, indent=args.indent, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
    print(f"[JSON]  {args.output.resolve()}")

    if args.markdown:
        _write_markdown(report, args.markdown)
        print(f"[MD  ]  {args.markdown.resolve()}")

    if args.graphviz:
        _write_graphviz(report["dependency_graph"], args.graphviz)
        print(f"[DOT ]  {args.graphviz.resolve()}")

if __name__ == "__main__":
    _cli()
