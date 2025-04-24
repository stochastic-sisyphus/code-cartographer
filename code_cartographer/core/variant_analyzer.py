"""
Code Variant Analysis Engine
===========================
Specialized engine for detecting and analyzing code variants and duplicates
across a Python codebase, with support for semantic similarity detection.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

@dataclass
class CodeVariant:
    """Represents a specific variant of a code entity."""
    path: str
    source: str
    hash: str
    start_line: int
    end_line: int
    similarity_score: Optional[float] = None
    
@dataclass
class VariantCluster:
    """Group of related code variants."""
    name: str
    category: str  # function, class, method
    variants: List[CodeVariant]
    base_variant: CodeVariant
    semantic_matches: bool = False
    
    @property
    def variant_count(self) -> int:
        return len(self.variants)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "variant_count": self.variant_count,
            "semantic_matches": self.semantic_matches,
            "base_variant": asdict(self.base_variant),
            "variants": [asdict(v) for v in self.variants]
        }

class CodeNormalizer:
    """Normalizes Python code for semantic comparison."""
    
    @staticmethod
    def normalize(node: ast.AST) -> str:
        """Convert AST to normalized form for comparison."""
        if isinstance(node, ast.Name):
            return "NAME"
        elif isinstance(node, ast.Num):
            return "NUMBER"
        elif isinstance(node, ast.Str):
            return "STRING"
        elif isinstance(node, ast.List):
            return f"[{', '.join(CodeNormalizer.normalize(e) for e in node.elts)}]"
        elif isinstance(node, ast.Dict):
            keys = [CodeNormalizer.normalize(k) for k in node.keys if k is not None]
            values = [CodeNormalizer.normalize(v) for v in node.values]
            return f"{{{', '.join(f'{k}: {v}' for k, v in zip(keys, values))}}}"
        elif isinstance(node, ast.Call):
            func = CodeNormalizer.normalize(node.func)
            args = [CodeNormalizer.normalize(a) for a in node.args]
            kwargs = [f"{k.arg}={CodeNormalizer.normalize(k.value)}" for k in node.keywords]
            return f"{func}({', '.join(args + kwargs)})"
        elif isinstance(node, ast.Attribute):
            return f"{CodeNormalizer.normalize(node.value)}.{node.attr}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [CodeNormalizer.normalize(n) for n in node.body]
            return f"def {node.name}({CodeNormalizer.normalize(node.args)}):\n{' '.join(body)}"
        elif isinstance(node, ast.arguments):
            parts = []
            # Handle positional args
            parts.extend(CodeNormalizer.normalize(a) for a in node.posonlyargs)
            parts.extend(CodeNormalizer.normalize(a) for a in node.args)
            # Handle *args
            if node.vararg:
                parts.append(f"*{node.vararg.arg}")
            # Handle keyword args
            parts.extend(CodeNormalizer.normalize(a) for a in node.kwonlyargs)
            # Handle **kwargs
            if node.kwarg:
                parts.append(f"**{node.kwarg.arg}")
            return ", ".join(parts)
        elif isinstance(node, ast.arg):
            return node.arg
        else:
            # Default handling for other node types
            return ast.unparse(node)

class SemanticAnalyzer:
    """Analyzes semantic similarity between code variants."""
    
    def __init__(self):
        if not _HAS_ML:
            raise ImportError(
                "Semantic analysis requires scikit-learn and numpy. "
                "Install with: pip install scikit-learn numpy"
            )
        
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            min_df=1,
            stop_words=None
        )
    
    def compute_similarity(self, base_code: str, variant_code: str) -> float:
        """Compute semantic similarity score between two code snippets."""
        # Normalize and vectorize
        texts = [base_code, variant_code]
        try:
            vectors = self.vectorizer.fit_transform(texts)
            # Compute cosine similarity
            similarity = cosine_similarity(vectors[:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
    
    def find_semantic_variants(
        self,
        base_code: str,
        candidates: List[Tuple[str, str, int, int]],
        threshold: float = 0.8
    ) -> List[CodeVariant]:
        """Find semantically similar variants from candidates."""
        variants = []
        base_normalized = self._normalize_code(base_code)
        
        for source, path, start, end in candidates:
            try:
                normalized = self._normalize_code(source)
                similarity = self.compute_similarity(base_normalized, normalized)
                
                if similarity >= threshold:
                    variants.append(
                        CodeVariant(
                            path=path,
                            source=source,
                            hash=hashlib.sha256(source.encode()).hexdigest(),
                            start_line=start,
                            end_line=end,
                            similarity_score=similarity
                        )
                    )
            except Exception:
                continue
                
        return variants
    
    def _normalize_code(self, source: str) -> str:
        """Normalize code for semantic comparison."""
        try:
            tree = ast.parse(source)
            return CodeNormalizer.normalize(tree)
        except Exception:
            return source

class VariantAnalyzer:
    """Main engine for variant analysis across a codebase."""
    
    def __init__(
        self,
        root: Path,
        semantic_threshold: float = 0.8,
        min_lines: int = 5,
        exclude_patterns: Optional[List[str]] = None
    ):
        self.root = root
        self.semantic_threshold = semantic_threshold
        self.min_lines = min_lines
        self.exclude = [re.compile(p) for p in (exclude_patterns or [])]
        
        # Analysis results
        self.exact_variants: Dict[str, VariantCluster] = {}
        self.semantic_variants: Dict[str, VariantCluster] = {}
        
        # Optional semantic analyzer
        self.semantic_analyzer = (
            SemanticAnalyzer() if _HAS_ML else None
        )
    
    def analyze(self) -> Dict[str, Any]:
        """Run full variant analysis on the codebase."""
        # First pass: collect all code entities
        entities: Dict[str, List[Tuple[str, str, int, int]]] = defaultdict(list)
        
        for py_file in self._find_python_files():
            try:
                self._collect_entities(py_file, entities)
            except Exception as e:
                print(f"[ERR] Failed to analyze {py_file}: {e}")
        
        # Second pass: analyze variants
        self._analyze_variants(entities)
        
        return self._generate_report()
    
    def _find_python_files(self):
        """Find all Python files, respecting exclusion patterns."""
        for p in self.root.rglob("*.py"):
            if any(rx.search(str(p)) for rx in self.exclude):
                continue
            yield p
    
    def _collect_entities(
        self,
        path: Path,
        entities: Dict[str, List[Tuple[str, str, int, int]]]
    ):
        """Collect code entities from a single file."""
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip small entities
                if node.end_lineno - node.lineno < self.min_lines:
                    continue
                
                source_segment = ast.get_source_segment(source, node) or ""
                if not source_segment:
                    continue
                
                key = f"{type(node).__name__}:{node.name}"
                entities[key].append((
                    source_segment,
                    str(path.relative_to(self.root)),
                    node.lineno,
                    node.end_lineno
                ))
    
    def _analyze_variants(
        self,
        entities: Dict[str, List[Tuple[str, str, int, int]]]
    ):
        """Analyze collected entities for variants."""
        for key, implementations in entities.items():
            if len(implementations) <= 1:
                continue

            category, name = key.split(":", 1)
            base_src, base_path, base_start, base_end = implementations[0]
            base_hash = hashlib.sha256(base_src.encode()).hexdigest()

            base_variant = CodeVariant(
                path=base_path,
                source=base_src,
                hash=base_hash,
                start_line=base_start,
                end_line=base_end
            )

            # Track exact duplicates
            exact_matches = []
            semantic_candidates = []

            for src, path, start, end in implementations[1:]:
                curr_hash = hashlib.sha256(src.encode()).hexdigest()

                if curr_hash == base_hash:
                    exact_matches.append(
                        CodeVariant(
                            path=path,
                            source=src,
                            hash=curr_hash,
                            start_line=start,
                            end_line=end
                        )
                    )
                else:
                    semantic_candidates.append((src, path, start, end))

            # Record exact variants if found
            if exact_matches:
                self.exact_variants[name] = VariantCluster(
                    name=name,
                    category=category.lower(),
                    variants=exact_matches,
                    base_variant=base_variant
                )

            # Analyze semantic variants if enabled
            if self.semantic_analyzer and semantic_candidates:
                if semantic_matches := self.semantic_analyzer.find_semantic_variants(
                    base_src, semantic_candidates, self.semantic_threshold
                ):
                    self.semantic_variants[name] = VariantCluster(
                        name=name,
                        category=category.lower(),
                        variants=semantic_matches,
                        base_variant=base_variant,
                        semantic_matches=True
                    )
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive variant analysis report."""
        report = {
            "summary": {
                "total_exact_variants": len(self.exact_variants),
                "total_semantic_variants": len(self.semantic_variants),
                "semantic_analysis_enabled": bool(self.semantic_analyzer)
            },
            "exact_variants": {},
            "semantic_variants": {}
        }
        
        # Process exact variants
        for name, cluster in self.exact_variants.items():
            report["exact_variants"][name] = {
                **cluster.to_dict(),
                "refactoring_recommendation": self._generate_refactor_prompt(
                    name,
                    cluster.category,
                    "exact"
                )
            }
        
        # Process semantic variants
        for name, cluster in self.semantic_variants.items():
            report["semantic_variants"][name] = {
                **cluster.to_dict(),
                "refactoring_recommendation": self._generate_refactor_prompt(
                    name,
                    cluster.category,
                    "semantic"
                )
            }
        
        return report
    
    def _generate_refactor_prompt(
        self,
        name: str,
        category: str,
        variant_type: str
    ) -> str:
        """Generate targeted refactoring recommendation."""
        base = f"Multiple {variant_type} variants of {category} `{name}` detected."
        
        if variant_type == "exact":
            action = textwrap.dedent("""
                Recommended Action:
                1. Extract shared implementation to a common module
                2. Replace duplicates with imports
                3. Add tests to verify identical behavior
                4. Document the canonical implementation
            """).strip()
        else:
            action = textwrap.dedent("""
                Recommended Action:
                1. Review semantic differences between variants
                2. Identify superset of functionality
                3. Create unified implementation supporting all use cases
                4. Add parameterization for variant-specific behavior
                5. Add comprehensive test suite
                6. Migrate all variants to new implementation
            """).strip()
        
        return f"{base}\n\n{action}"

def main():
    """CLI entry point for variant analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze Python code variants and duplicates",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-d", "--dir",
        type=Path,
        required=True,
        help="Root directory to analyze"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("variant_analysis.json"),
        help="Output JSON file path"
    )
    
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for semantic variants (0.0-1.0)"
    )
    
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum lines for variant consideration"
    )
    
    parser.add_argument(
        "-e", "--exclude",
        nargs="*",
        default=[],
        help="Regex patterns for paths to exclude"
    )
    
    args = parser.parse_args()
    
    analyzer = VariantAnalyzer(
        root=args.dir,
        semantic_threshold=args.semantic_threshold,
        min_lines=args.min_lines,
        exclude_patterns=args.exclude
    )
    
    analysis = analyzer.analyze()
    
    args.output.write_text(
        json.dumps(analysis, indent=2)
    )
    print(f"[INFO] Analysis complete: {args.output.resolve()}")

if __name__ == "__main__":
    main() 