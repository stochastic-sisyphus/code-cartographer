"""
Code Variant Analyzer
====================
Detects and analyzes code variants and duplicates in Python codebases.
"""

import ast
import hashlib
import logging
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class CodeNormalizer:
    """Normalizes Python code for variant analysis by removing syntax noise."""

    @staticmethod
    def normalize(code: str) -> str:
        """Normalize code by removing comments, docstrings, and standardizing formatting.

        Args:
            code: Python source code to normalize

        Returns:
            Normalized code string
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Apply transformations
            tree = CodeNormalizer._remove_docstrings(tree)
            tree = CodeNormalizer._standardize_names(tree)
            tree = CodeNormalizer._normalize_literals(tree)
            tree = CodeNormalizer._normalize_imports(tree)
            tree = CodeNormalizer._normalize_function_args(tree)

            return ast.unparse(tree)
        except SyntaxError:
            return code

    @staticmethod
    def _remove_docstrings(tree: ast.AST) -> ast.AST:
        """Remove docstrings and comments from AST."""

        class DocstringRemover(ast.NodeTransformer):
            def visit_Expr(self, node):
                return None if isinstance(node.value, ast.Str) else node

        return DocstringRemover().visit(tree)

    @staticmethod
    def _standardize_names(tree: ast.AST) -> ast.AST:
        """Standardize variable, function, and class names."""
        name_map = {}
        counter = 0

        class NameStandardizer(ast.NodeTransformer):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in name_map:
                        name_map[node.id] = f"var_{counter}"
                        counter += 1
                    node.id = name_map[node.id]
                elif isinstance(node.ctx, ast.Load) and node.id in name_map:
                    node.id = name_map[node.id]
                return node

            def visit_FunctionDef(self, node):
                return self._extracted_from_visit_FunctionDef_19(node, "func_")

            def _extracted_from_visit_FunctionDef_19(self, node, arg1):
                if node.name not in name_map:
                    name_map[node.name] = f"{arg1}{counter}"
                    counter += 1
                node.name = name_map[node.name]
                return self.generic_visit(node)

            def visit_ClassDef(self, node):
                return self._extracted_from_visit_FunctionDef_19(node, "class_")

        return NameStandardizer().visit(tree)

    @staticmethod
    def _normalize_literals(tree: ast.AST) -> ast.AST:
        """Normalize string and numeric literals."""

        class LiteralNormalizer(ast.NodeTransformer):
            def visit_Str(self, node):
                return ast.Constant(value="<str>")

            def visit_Num(self, node):
                if isinstance(node.n, int):
                    return ast.Constant(value=0)
                return ast.Constant(value=0.0)

            def visit_List(self, node):
                if not node.elts:
                    return node
                return ast.List(elts=[ast.Constant(value="<item>")], ctx=node.ctx)

            def visit_Dict(self, node):
                if not node.keys:
                    return node
                return ast.Dict(
                    keys=[ast.Constant(value="<key>")],
                    values=[ast.Constant(value="<value>")],
                )

        return LiteralNormalizer().visit(tree)

    @staticmethod
    def _normalize_imports(tree: ast.AST) -> ast.AST:
        """Normalize import statements."""

        class ImportNormalizer(ast.NodeTransformer):
            def visit_Import(self, node):
                return ast.Import(names=[ast.alias(name="<module>", asname=None)])

            def visit_ImportFrom(self, node):
                return ast.ImportFrom(
                    module="<module>",
                    names=[ast.alias(name="<name>", asname=None)],
                    level=0,
                )

        return ImportNormalizer().visit(tree)

    @staticmethod
    def _normalize_function_args(tree: ast.AST) -> ast.AST:
        """Normalize function arguments and their defaults."""

        class ArgNormalizer(ast.NodeTransformer):
            def visit_arguments(self, node):
                # Normalize argument names
                if node.args:
                    for i, arg in enumerate(node.args):
                        arg.arg = f"arg_{i}"

                # Normalize defaults to None
                if node.defaults:
                    node.defaults = [ast.Constant(value=None) for _ in node.defaults]

                # Normalize kwonlyargs
                if node.kwonlyargs:
                    for i, arg in enumerate(node.kwonlyargs):
                        arg.arg = f"kwarg_{i}"

                # Normalize kw_defaults to None
                if node.kw_defaults:
                    node.kw_defaults = [
                        ast.Constant(value=None) for _ in node.kw_defaults
                    ]

                return node

        return ArgNormalizer().visit(tree)


@dataclass
class CodeBlock:
    """Represents a block of code with metadata."""

    path: Path
    start_line: int
    end_line: int
    content: str
    hash: str = field(init=False)
    normalized: str = field(init=False)
    tokens: List[str] = field(init=False)

    def __post_init__(self):
        """Initialize derived fields."""
        self.hash = hashlib.sha256(self.content.encode()).hexdigest()
        self.normalized = CodeNormalizer.normalize(self.content)
        self.tokens = word_tokenize(self.normalized.lower())

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "path": str(self.path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "hash": self.hash,
        }


@dataclass
class VariantGroup:
    """Group of related code variants."""

    blocks: List[CodeBlock]
    similarity: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "similarity": self.similarity,
        }


@dataclass
class VariantMatch:
    """Represents a match between two code variants."""

    source_block: CodeBlock
    target_block: CodeBlock
    similarity: float
    diff: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "source": self.source_block.to_dict(),
            "target": self.target_block.to_dict(),
            "similarity": self.similarity,
            "diff": self.diff,
        }


class SemanticAnalyzer:
    """Analyzes semantic similarity between code blocks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the semantic analyzer.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.normalizer = CodeNormalizer()

    def compare_blocks(self, block1: CodeBlock, block2: CodeBlock) -> float:
        """Compare two code blocks and return their semantic similarity.

        Args:
            block1: First code block
            block2: Second code block

        Returns:
            Similarity score between 0 and 1
        """
        # Get embeddings for both normalized and original code
        emb1_norm = self.model.encode(block1.normalized)
        emb2_norm = self.model.encode(block2.normalized)
        emb1_orig = self.model.encode(block1.content)
        emb2_orig = self.model.encode(block2.content)

        # Calculate similarities
        norm_sim = self._cosine_similarity(emb1_norm, emb2_norm)
        orig_sim = self._cosine_similarity(emb1_orig, emb2_orig)

        # Weight normalized similarity higher
        return 0.7 * norm_sim + 0.3 * orig_sim

    def find_variants(
        self, blocks: List[CodeBlock], threshold: float = 0.8
    ) -> List[VariantGroup]:
        """Find semantically similar code blocks.

        Args:
            blocks: List of code blocks to analyze
            threshold: Minimum similarity threshold

        Returns:
            List of variant groups
        """
        # Pre-compute embeddings for efficiency
        norm_embeddings = self.model.encode([b.normalized for b in blocks])
        orig_embeddings = self.model.encode([b.content for b in blocks])

        # Find variant groups
        groups = []
        used = set()

        for i, block1 in enumerate(blocks):
            if i in used:
                continue

            group = []
            for j, block2 in enumerate(blocks[i + 1 :], i + 1):
                if j in used:
                    continue

                # Calculate combined similarity
                norm_sim = self._cosine_similarity(
                    norm_embeddings[i], norm_embeddings[j]
                )
                orig_sim = self._cosine_similarity(
                    orig_embeddings[i], orig_embeddings[j]
                )
                similarity = 0.7 * norm_sim + 0.3 * orig_sim

                if similarity >= threshold:
                    if not group:
                        group = [block1]
                    group.append(block2)
                    used.add(j)

            if group:
                used.add(i)
                groups.append(VariantGroup(blocks=group, similarity=similarity))

        return groups

    def analyze_variants(
        self, blocks: List[CodeBlock], threshold: float = 0.8
    ) -> List[VariantMatch]:
        """Analyze variants and generate detailed matches.

        Args:
            blocks: List of code blocks to analyze
            threshold: Minimum similarity threshold

        Returns:
            List of variant matches with detailed analysis
        """
        matches = []

        for i, block1 in enumerate(blocks):
            for block2 in blocks[i + 1 :]:
                similarity = self.compare_blocks(block1, block2)

                if similarity >= threshold:
                    # Generate diff for visualization
                    diff = list(
                        difflib.unified_diff(
                            block1.content.splitlines(),
                            block2.content.splitlines(),
                            lineterm="",
                        )
                    )

                    matches.append(
                        VariantMatch(
                            source_block=block1,
                            target_block=block2,
                            similarity=similarity,
                            diff=diff,
                        )
                    )

        return matches

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(v1 @ v2 / ((v1**2).sum() ** 0.5 * (v2**2).sum() ** 0.5))


@dataclass
class MergeResult:
    """Result of merging code variants."""

    merged_content: str
    original_blocks: List[CodeBlock]
    affected_files: Set[Path]
    changes: List[Dict[str, str]]
    similarity_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "merged_content": self.merged_content,
            "original_blocks": [b.to_dict() for b in self.original_blocks],
            "affected_files": [str(p) for p in self.affected_files],
            "changes": self.changes,
            "similarity_score": self.similarity_score,
        }


class VariantMerger:
    """Handles merging of code variants into a single canonical version."""

    def __init__(self):
        """Initialize the variant merger."""
        self.normalizer = CodeNormalizer()

    def merge_variants(self, group: VariantGroup) -> MergeResult:
        """Merge a group of code variants into a single best version.

        Args:
            group: Group of code variants to merge

        Returns:
            Merge result containing the merged code and change details
        """
        blocks = group.blocks
        if not blocks:
            raise ValueError("Cannot merge empty variant group")

        # Find the most complex version as base
        base_block = self._find_base_block(blocks)

        # Parse all blocks
        trees = []
        for block in blocks:
            try:
                tree = ast.parse(block.content)
                trees.append((block, tree))
            except SyntaxError as e:
                logger.warning(f"Failed to parse {block.path}: {e}")

        if not trees:
            raise ValueError("No valid code blocks to merge")

        # Merge the ASTs
        merged_tree = self._merge_asts(base_block, trees)

        # Generate merged code
        merged_content = ast.unparse(merged_tree)

        # Track changes for each file
        changes = []
        affected_files = {b.path for b in blocks}

        for block in blocks:
            if block != base_block:
                diff = list(
                    difflib.unified_diff(
                        block.content.splitlines(),
                        merged_content.splitlines(),
                        lineterm="",
                    )
                )
                changes.append(
                    {
                        "file": str(block.path),
                        "original": block.content,
                        "replacement": merged_content,
                        "diff": diff,
                    }
                )

        return MergeResult(
            merged_content=merged_content,
            original_blocks=blocks,
            affected_files=affected_files,
            changes=changes,
            similarity_score=group.similarity,
        )

    def _find_base_block(self, blocks: List[CodeBlock]) -> CodeBlock:
        """Find the most complex version to use as merge base."""

        def complexity_score(block: CodeBlock) -> int:
            try:
                tree = ast.parse(block.content)
                return sum(1 for _ in ast.walk(tree))
            except SyntaxError:
                return 0

        return max(blocks, key=complexity_score)

    def _merge_asts(
        self, base_block: CodeBlock, trees: List[Tuple[CodeBlock, ast.AST]]
    ) -> ast.AST:
        """Merge multiple ASTs into one, preserving unique functionality."""
        base_tree = ast.parse(base_block.content)

        # Collect all unique function/method bodies
        bodies = {}
        for block, tree in trees:
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    normalized = self.normalizer.normalize(
                        ast.get_source_segment(block.content, node)
                    )
                    if normalized not in bodies:
                        bodies[normalized] = node

        # Create merged tree by combining unique elements
        class TreeMerger(ast.NodeTransformer):
            def visit_Module(self, node):
                # Replace module body with all unique definitions
                node.body = list(bodies.values())
                return node

        return TreeMerger().visit(base_tree)


class CodePatcher:
    """Handles automatic patching of code files with merged variants."""

    def __init__(self, root: Path):
        """Initialize the code patcher.

        Args:
            root: Root directory of the project
        """
        self.root = root

    def apply_merge(self, merge_result: MergeResult) -> Dict[str, str]:
        """Apply merged changes to the original files.

        Args:
            merge_result: Result of merging variants

        Returns:
            Dictionary mapping file paths to their patched content
        """
        patched_files = {}

        for change in merge_result.changes:
            file_path = Path(change["file"])
            try:
                # Read original file
                content = file_path.read_text()

                # Find and replace the variant block
                new_content = content.replace(change["original"], change["replacement"])

                if new_content != content:
                    patched_files[str(file_path)] = new_content

            except Exception as e:
                logger.error(f"Failed to patch {file_path}: {e}")

        return patched_files

    def write_patches(self, patches: Dict[str, str], backup: bool = True) -> None:
        """Write patches to files.

        Args:
            patches: Dictionary mapping file paths to their patched content
            backup: Whether to create backup files
        """
        for file_path, content in patches.items():
            path = Path(file_path)
            try:
                if backup:
                    backup_path = path.with_suffix(f"{path.suffix}.bak")
                    path.rename(backup_path)

                path.write_text(content)
                logger.info(f"Patched {path}")

            except Exception as e:
                logger.error(f"Failed to write patch for {path}: {e}")
                if backup and "backup_path" in locals():
                    try:
                        backup_path.rename(path)
                    except Exception:
                        logger.error(f"Failed to restore backup for {path}")


class VariantAnalyzer:
    """Analyzes Python code for variants and duplicates."""

    def __init__(
        self,
        root: Path,
        semantic_threshold: float = 0.8,
        min_lines: int = 5,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """Initialize the analyzer.

        Args:
            root: Root directory to analyze
            semantic_threshold: Minimum similarity for semantic variants
            min_lines: Minimum lines for variant consideration
            exclude_patterns: Regex patterns for paths to exclude
        """
        self.root = Path(root)
        self.semantic_threshold = semantic_threshold
        self.min_lines = min_lines
        self.exclude_patterns = exclude_patterns or []

        # Initialize components
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.semantic_analyzer = SemanticAnalyzer()
        self.merger = VariantMerger()
        self.patcher = CodePatcher(root)

    def analyze(self) -> Dict:
        """Run the variant analysis.

        Returns:
            Dictionary containing analysis results with:
            - exact_duplicates: List of exact code duplicates
            - semantic_variants: List of semantically similar code blocks
            - variant_matches: List of pairwise variant matches with diffs
            - merged_variants: List of automatically merged variants
            - statistics: Analysis statistics and metrics
        """
        # Extract code blocks
        blocks = self._extract_blocks()
        if not blocks:
            return {
                "exact_duplicates": [],
                "semantic_variants": [],
                "variant_matches": [],
                "merged_variants": [],
                "statistics": self._empty_stats(),
            }

        # Find exact duplicates
        exact_duplicates = self._find_exact_duplicates(blocks)

        # Find semantic variants
        semantic_variants = self.semantic_analyzer.find_variants(
            blocks, self.semantic_threshold
        )

        # Generate detailed variant matches
        variant_matches = self.semantic_analyzer.analyze_variants(
            blocks, self.semantic_threshold
        )

        # Attempt to merge variants
        merged_variants = []
        for group in semantic_variants:
            try:
                merge_result = self.merger.merge_variants(group)
                merged_variants.append(merge_result)
            except Exception as e:
                logger.warning(f"Failed to merge variant group: {e}")

        # Compute statistics
        stats = self._compute_statistics(blocks, exact_duplicates, semantic_variants)

        return {
            "exact_duplicates": [g.to_dict() for g in exact_duplicates],
            "semantic_variants": [g.to_dict() for g in semantic_variants],
            "variant_matches": [m.to_dict() for m in variant_matches],
            "merged_variants": [m.to_dict() for m in merged_variants],
            "statistics": stats,
        }

    def apply_merged_variants(self, backup: bool = True) -> None:
        """Apply all merged variants to the codebase.

        Args:
            backup: Whether to create backup files before patching
        """
        analysis = self.analyze()
        merged_variants = analysis.get("merged_variants", [])

        for merge_result in merged_variants:
            # Convert dict back to MergeResult
            result = MergeResult(
                merged_content=merge_result["merged_content"],
                original_blocks=[
                    CodeBlock(**b) for b in merge_result["original_blocks"]
                ],
                affected_files={Path(p) for p in merge_result["affected_files"]},
                changes=merge_result["changes"],
                similarity_score=merge_result["similarity_score"],
            )

            # Apply patches
            patches = self.patcher.apply_merge(result)
            self.patcher.write_patches(patches, backup=backup)

    def _extract_blocks(self) -> List[CodeBlock]:
        """Extract code blocks from Python files."""
        blocks = []

        for path in self.root.rglob("*.py"):
            # Skip excluded paths
            if any(pattern in str(path) for pattern in self.exclude_patterns):
                continue

            try:
                content = path.read_text()
                tree = ast.parse(content)

                # Extract function and class definitions
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        code = ast.get_source_segment(content, node)
                        if code and len(code.splitlines()) >= self.min_lines:
                            blocks.append(
                                CodeBlock(
                                    path=path,
                                    start_line=node.lineno,
                                    end_line=node.end_lineno,
                                    content=code,
                                )
                            )

            except Exception as e:
                logger.warning(f"Error processing {path}: {e}")
                continue

        return blocks

    def _find_exact_duplicates(self, blocks: List[CodeBlock]) -> List[VariantGroup]:
        """Find exactly matching code blocks."""
        duplicates = {}

        for block in blocks:
            if block.hash in duplicates:
                duplicates[block.hash].append(block)
            else:
                duplicates[block.hash] = [block]

        return [
            VariantGroup(blocks=group, similarity=1.0)
            for group in duplicates.values()
            if len(group) > 1
        ]

    def _compute_statistics(
        self,
        blocks: List[CodeBlock],
        exact_dupes: List[VariantGroup],
        semantic_vars: List[VariantGroup],
    ) -> Dict:
        """Compute analysis statistics."""
        total_lines = sum(b.end_line - b.start_line + 1 for b in blocks)
        duplicate_lines = sum(
            sum(b.end_line - b.start_line + 1 for b in g.blocks) for g in exact_dupes
        )
        variant_lines = sum(
            sum(b.end_line - b.start_line + 1 for b in g.blocks) for g in semantic_vars
        )

        return {
            "total_blocks": len(blocks),
            "total_lines": total_lines,
            "duplicate_groups": len(exact_dupes),
            "duplicate_blocks": sum(len(g.blocks) for g in exact_dupes),
            "duplicate_lines": duplicate_lines,
            "variant_groups": len(semantic_vars),
            "variant_blocks": sum(len(g.blocks) for g in semantic_vars),
            "variant_lines": variant_lines,
            "duplicate_percentage": (
                round(duplicate_lines / total_lines * 100, 2)
                if total_lines > 0
                else 0.0
            ),
            "variant_percentage": (
                round(variant_lines / total_lines * 100, 2) if total_lines > 0 else 0.0
            ),
        }

    def _empty_stats(self) -> Dict:
        """Return empty statistics structure."""
        return {
            "total_blocks": 0,
            "total_lines": 0,
            "duplicate_groups": 0,
            "duplicate_blocks": 0,
            "duplicate_lines": 0,
            "variant_groups": 0,
            "variant_blocks": 0,
            "variant_lines": 0,
            "duplicate_percentage": 0.0,
            "variant_percentage": 0.0,
        }


"""
Code Variant Analyzer
====================
Detects and analyzes code variants and duplicates in Python codebases.
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

# Make NLP dependencies optional
HAS_NLP = False
try:
    import nltk
    from nltk.tokenize import word_tokenize

    HAS_NLP = True
except ImportError:

    def word_tokenize(text):
        return text.split()


import contextlib

HAS_TRANSFORMERS = False
with contextlib.suppress(ImportError):
    from sentence_transformers import SentenceTransformer

    HAS_TRANSFORMERS = True
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class CodeNormalizer:
    """Normalizes Python code for variant analysis by removing syntax noise."""

    @staticmethod
    def normalize(code: str) -> str:
        """Normalize code by removing comments, docstrings, and standardizing formatting.

        Args:
            code: Python source code to normalize

        Returns:
            Normalized code string
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Apply transformations
            tree = CodeNormalizer._remove_docstrings(tree)
            tree = CodeNormalizer._standardize_names(tree)
            tree = CodeNormalizer._normalize_literals(tree)
            tree = CodeNormalizer._normalize_imports(tree)
            tree = CodeNormalizer._normalize_function_args(tree)

            return ast.unparse(tree)
        except SyntaxError:
            return code

    @staticmethod
    def _remove_docstrings(tree: ast.AST) -> ast.AST:
        """Remove docstrings and comments from AST."""

        class DocstringRemover(ast.NodeTransformer):
            def visit_Expr(self, node):
                return None if isinstance(node.value, ast.Str) else node

        return DocstringRemover().visit(tree)

    @staticmethod
    def _standardize_names(tree: ast.AST) -> ast.AST:
        """Standardize variable, function, and class names."""
        name_map = {}
        counter = 0

        class NameStandardizer(ast.NodeTransformer):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in name_map:
                        name_map[node.id] = f"var_{counter}"
                        counter += 1
                    node.id = name_map[node.id]
                elif isinstance(node.ctx, ast.Load) and node.id in name_map:
                    node.id = name_map[node.id]
                return node

            def visit_FunctionDef(self, node):
                return self._extracted_from_visit_FunctionDef_19(node, "func_")

            def _extracted_from_visit_FunctionDef_19(self, node, arg1):
                if node.name not in name_map:
                    name_map[node.name] = f"{arg1}{counter}"
                    counter += 1
                node.name = name_map[node.name]
                return self.generic_visit(node)

            def visit_ClassDef(self, node):
                return self._extracted_from_visit_FunctionDef_19(node, "class_")

        return NameStandardizer().visit(tree)

    def _normalize_literals(self) -> ast.AST:
        """Normalize string and numeric literals."""

        class LiteralNormalizer(ast.NodeTransformer):
            def visit_Str(self, node):
                return ast.Constant(value="<str>")

            def visit_Num(self, node):
                if isinstance(node.n, int):
                    return ast.Constant(value=0)
                return ast.Constant(value=0.0)

            def visit_List(self, node):
                if not node.elts:
                    return node
                return ast.List(elts=[ast.Constant(value="<item>")], ctx=node.ctx)

            def visit_Dict(self, node):
                if not node.keys:
                    return node
                return ast.Dict(
                    keys=[ast.Constant(value="<key>")],
                    values=[ast.Constant(value="<value>")],
                )

        return LiteralNormalizer().visit(self)

    @staticmethod
    def _normalize_imports(tree: ast.AST) -> ast.AST:
        """Normalize import statements."""

        class ImportNormalizer(ast.NodeTransformer):
            def visit_Import(self, node):
                return ast.Import(names=[ast.alias(name="<module>", asname=None)])

            def visit_ImportFrom(self, node):
                return ast.ImportFrom(
                    module="<module>",
                    names=[ast.alias(name="<n>", asname=None)],
                    level=0,
                )

        return ImportNormalizer().visit(tree)

    @staticmethod
    def _normalize_function_args(tree: ast.AST) -> ast.AST:
        """Normalize function arguments and their defaults."""

        class ArgNormalizer(ast.NodeTransformer):
            def visit_arguments(self, node):
                # Normalize argument names
                if node.args:
                    for i, arg in enumerate(node.args):
                        arg.arg = f"arg_{i}"

                # Normalize defaults to None
                if node.defaults:
                    node.defaults = [ast.Constant(value=None) for _ in node.defaults]

                # Normalize kwonlyargs
                if node.kwonlyargs:
                    for i, arg in enumerate(node.kwonlyargs):
                        arg.arg = f"kwarg_{i}"

                # Normalize kw_defaults to None
                if node.kw_defaults:
                    node.kw_defaults = [
                        ast.Constant(value=None) for _ in node.kw_defaults
                    ]

                return node

        return ArgNormalizer().visit(tree)


@dataclass
class CodeBlock:
    """Represents a block of code with metadata."""

    path: Path
    start_line: int
    end_line: int
    content: str
    hash: str = field(init=False)
    normalized: str = field(init=False)
    tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize derived fields."""
        self.hash = hashlib.sha256(self.content.encode()).hexdigest()
        self.normalized = CodeNormalizer.normalize(self.content)
        if HAS_NLP:
            self.tokens = word_tokenize(self.normalized.lower())
        else:
            self.tokens = self.normalized.lower().split()

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "path": str(self.path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "hash": self.hash,
        }


@dataclass
class VariantGroup:
    """Group of related code variants."""

    blocks: List[CodeBlock]
    similarity: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "blocks": [b.to_dict() for b in self.blocks],
            "similarity": self.similarity,
        }


@dataclass
class VariantMatch:
    """Represents a match between two code variants."""

    source_block: CodeBlock
    target_block: CodeBlock
    similarity: float
    diff: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "source": self.source_block.to_dict(),
            "target": self.target_block.to_dict(),
            "similarity": self.similarity,
            "diff": self.diff,
        }


class SemanticAnalyzer:
    """Analyzes semantic similarity between code blocks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the semantic analyzer.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = None
        self.normalizer = CodeNormalizer()

        if HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")

    def compare_blocks(self, block1: CodeBlock, block2: CodeBlock) -> float:
        """Compare two code blocks and return their semantic similarity.

        Args:
            block1: First code block
            block2: Second code block

        Returns:
            Similarity score between 0 and 1
        """
        if not self.model:
            # Fallback to simple token-based similarity if no model is available
            return self._token_similarity(block1, block2)

        # Get embeddings for both normalized and original code
        emb1_norm = self.model.encode(block1.normalized)
        emb2_norm = self.model.encode(block2.normalized)
        emb1_orig = self.model.encode(block1.content)
        emb2_orig = self.model.encode(block2.content)

        # Calculate similarities
        norm_sim = self._cosine_similarity(emb1_norm, emb2_norm)
        orig_sim = self._cosine_similarity(emb1_orig, emb2_orig)

        # Weight normalized similarity higher
        return 0.7 * norm_sim + 0.3 * orig_sim

    def _token_similarity(self, block1: CodeBlock, block2: CodeBlock) -> float:
        """Calculate token-based similarity as fallback."""
        # Use Jaccard similarity on tokens
        tokens1 = set(block1.tokens)
        tokens2 = set(block2.tokens)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        return intersection / union if union > 0 else 0.0

    def find_variants(
        self, blocks: List[CodeBlock], threshold: float = 0.8
    ) -> List[VariantGroup]:
        """Find semantically similar code blocks.

        Args:
            blocks: List of code blocks to analyze
            threshold: Minimum similarity threshold

        Returns:
            List of variant groups
        """
        if not self.model and not HAS_NLP:
            logger.warning(
                "NLP dependencies not available, variant analysis will be limited"
            )
            # Return empty list if NLP features are not available
            return []

        if not self.model:
            # Use token-based similarity as fallback
            return self._find_variants_token_based(blocks, threshold)

        # Pre-compute embeddings for efficiency
        norm_embeddings = self.model.encode([b.normalized for b in blocks])
        orig_embeddings = self.model.encode([b.content for b in blocks])

        # Find variant groups
        groups = []
        used = set()

        for i, block1 in enumerate(blocks):
            if i in used:
                continue

            group = []
            for j, block2 in enumerate(blocks[i + 1 :], i + 1):
                if j in used:
                    continue

                # Calculate combined similarity
                norm_sim = self._cosine_similarity(
                    norm_embeddings[i], norm_embeddings[j]
                )
                orig_sim = self._cosine_similarity(
                    orig_embeddings[i], orig_embeddings[j]
                )
                similarity = 0.7 * norm_sim + 0.3 * orig_sim

                if similarity >= threshold:
                    if not group:
                        group = [block1]
                    group.append(block2)
                    used.add(j)

            if group:
                used.add(i)
                groups.append(VariantGroup(blocks=group, similarity=similarity))

        return groups

    def _find_variants_token_based(
        self, blocks: List[CodeBlock], threshold: float = 0.8
    ) -> List[VariantGroup]:
        """Find variants using token-based similarity as fallback."""
        groups = []
        used = set()

        for i, block1 in enumerate(blocks):
            if i in used:
                continue

            group = []
            for j, block2 in enumerate(blocks[i + 1 :], i + 1):
                if j in used:
                    continue

                similarity = self._token_similarity(block1, block2)

                if similarity >= threshold:
                    if not group:
                        group = [block1]
                    group.append(block2)
                    used.add(j)

            if group:
                used.add(i)
                groups.append(VariantGroup(blocks=group, similarity=similarity))

        return groups

    def analyze_variants(
        self, blocks: List[CodeBlock], threshold: float = 0.8
    ) -> List[VariantMatch]:
        """Analyze variants and generate detailed matches.

        Args:
            blocks: List of code blocks to analyze
            threshold: Minimum similarity threshold

        Returns:
            List of variant matches with detailed analysis
        """
        if not self.model and not HAS_NLP:
            logger.warning(
                "NLP dependencies not available, variant analysis will be limited"
            )
            return []

        matches = []

        for i, block1 in enumerate(blocks):
            for block2 in blocks[i + 1 :]:
                similarity = self.compare_blocks(block1, block2)

                if similarity >= threshold:
                    # Generate diff for visualization
                    diff = list(
                        difflib.unified_diff(
                            block1.content.splitlines(),
                            block2.content.splitlines(),
                            lineterm="",
                        )
                    )

                    matches.append(
                        VariantMatch(
                            source_block=block1,
                            target_block=block2,
                            similarity=similarity,
                            diff=diff,
                        )
                    )

        return matches

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(v1 @ v2 / ((v1**2).sum() ** 0.5 * (v2**2).sum() ** 0.5))


@dataclass
class MergeResult:
    """Result of merging code variants."""

    merged_content: str
    original_blocks: List[CodeBlock]
    affected_files: Set[Path]
    changes: List[Dict[str, str]]
    similarity_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "merged_content": self.merged_content,
            "original_blocks": [b.to_dict() for b in self.original_blocks],
            "affected_files": [str(p) for p in self.affected_files],
            "changes": self.changes,
            "similarity_score": self.similarity_score,
        }


class VariantMerger:
    """Handles merging of code variants into a single canonical version."""

    def __init__(self):
        """Initialize the variant merger."""
        self.normalizer = CodeNormalizer()

    def merge_variants(self, group: VariantGroup) -> MergeResult:
        """Merge a group of code variants into a single best version.

        Args:
            group: Group of code variants to merge

        Returns:
            Merge result containing the merged code and change details
        """
        blocks = group.blocks
        if not blocks:
            raise ValueError("Cannot merge empty variant group")

        # Find the most complex version as base
        base_block = self._find_base_block(blocks)

        # Parse all blocks
        trees = []
        for block in blocks:
            try:
                tree = ast.parse(block.content)
                trees.append((block, tree))
            except SyntaxError as e:
                logger.warning(f"Failed to parse {block.path}: {e}")

        if not trees:
            raise ValueError("No valid code blocks to merge")

        # Merge the ASTs
        merged_tree = self._merge_asts(base_block, trees)

        # Generate merged code
        merged_content = ast.unparse(merged_tree)

        # Track changes for each file
        changes = []
        affected_files = {b.path for b in blocks}

        for block in blocks:
            if block != base_block:
                diff = list(
                    difflib.unified_diff(
                        block.content.splitlines(),
                        merged_content.splitlines(),
                        lineterm="",
                    )
                )
                changes.append(
                    {
                        "file": str(block.path),
                        "original": block.content,
                        "replacement": merged_content,
                        "diff": diff,
                    }
                )

        return MergeResult(
            merged_content=merged_content,
            original_blocks=blocks,
            affected_files=affected_files,
            changes=changes,
            similarity_score=group.similarity,
        )

    def _find_base_block(self, blocks: List[CodeBlock]) -> CodeBlock:
        """Find the most complex version to use as merge base."""

        def complexity_score(block: CodeBlock) -> int:
            try:
                tree = ast.parse(block.content)
                return sum(1 for _ in ast.walk(tree))
            except SyntaxError:
                return 0

        return max(blocks, key=complexity_score)

    def _merge_asts(
        self, base_block: CodeBlock, trees: List[Tuple[CodeBlock, ast.AST]]
    ) -> ast.AST:
        """Merge multiple ASTs into one, preserving unique functionality."""
        return ast.parse(base_block.content)


class VariantAnalyzer:
    """Analyzes code variants in a Python codebase."""

    def __init__(
        self,
        root: Path,
        semantic_threshold: float = 0.8,
        min_lines: int = 5,
        exclude_patterns: List[str] = None,
    ):
        """Initialize the variant analyzer.

        Args:
            root: Root directory of the codebase
            semantic_threshold: Minimum similarity threshold for semantic variants
            min_lines: Minimum number of lines for a code block to be considered
            exclude_patterns: List of regex patterns for paths to exclude
        """
        self.root = Path(root)
        self.semantic_threshold = semantic_threshold
        self.min_lines = min_lines
        self.exclude_patterns = exclude_patterns or []

        # Initialize analyzers
        self.semantic_analyzer = None
        if HAS_TRANSFORMERS:
            try:
                self.semantic_analyzer = SemanticAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize semantic analyzer: {e}")
        else:
            logger.warning(
                "Sentence transformers not available, semantic analysis will be limited"
            )
            # Create a basic analyzer without transformer model
            self.semantic_analyzer = SemanticAnalyzer()

        self.merger = VariantMerger()

    def analyze(self) -> Dict:
        """Analyze the codebase for code variants.

        Returns:
            Dictionary with analysis results
        """
        # Extract code blocks
        blocks = self._extract_code_blocks()

        # Find exact duplicates
        exact_duplicates = self._find_exact_duplicates(blocks)

        # Find semantic variants if NLP is available
        semantic_variants = []
        if self.semantic_analyzer and (HAS_TRANSFORMERS or HAS_NLP):
            semantic_variants = self.semantic_analyzer.find_variants(
                blocks, self.semantic_threshold
            )

        # Generate report
        return {
            "exact_duplicates": [
                {
                    "hash": hash_val,
                    "blocks": [b.to_dict() for b in blocks],
                }
                for hash_val, blocks in exact_duplicates.items()
                if len(blocks) > 1
            ],
            "semantic_variants": [group.to_dict() for group in semantic_variants],
            "stats": {
                "total_blocks": len(blocks),
                "exact_duplicate_count": sum(
                    len(blocks) > 1 for blocks in exact_duplicates.values()
                ),
                "semantic_variant_count": len(semantic_variants),
            },
        }

    def _extract_code_blocks(self) -> List[CodeBlock]:
        """Extract code blocks from the codebase."""
        blocks = []

        for py_file in self._find_python_files():
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)

                # Extract function and class definitions
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        # Get source segment
                        source = ast.get_source_segment(content, node)
                        if source and len(source.splitlines()) >= self.min_lines:
                            blocks.append(
                                CodeBlock(
                                    path=py_file.relative_to(self.root),
                                    start_line=node.lineno,
                                    end_line=node.end_lineno,
                                    content=source,
                                )
                            )
            except Exception as e:
                logger.warning(f"Failed to process {py_file}: {e}")

        return blocks

    def _find_exact_duplicates(
        self, blocks: List[CodeBlock]
    ) -> Dict[str, List[CodeBlock]]:
        """Find exact code duplicates based on hash."""
        duplicates = {}

        for block in blocks:
            if block.hash not in duplicates:
                duplicates[block.hash] = []
            duplicates[block.hash].append(block)

        return duplicates

    def _find_python_files(self):
        """Find all Python files in the project."""
        import re

        exclude_patterns = [re.compile(pattern) for pattern in self.exclude_patterns]

        for path in self.root.rglob("*.py"):
            rel_path = str(path.relative_to(self.root))
            if any(pattern.search(rel_path) for pattern in exclude_patterns):
                continue
            yield path

    def apply_merged_variants(self, backup: bool = True) -> Dict:
        """Apply merged variants to the codebase.

        Args:
            backup: Whether to create backup files

        Returns:
            Dictionary with merge results
        """
        if not HAS_TRANSFORMERS and not HAS_NLP:
            logger.warning(
                "NLP dependencies not available, cannot apply merged variants"
            )
            return {"error": "NLP dependencies not available"}

        # Extract code blocks
        blocks = self._extract_code_blocks()

        # Find semantic variants
        variants = self.semantic_analyzer.find_variants(blocks, self.semantic_threshold)

        # Merge and apply changes
        results = []
        for group in variants:
            try:
                merge_result = self.merger.merge_variants(group)

                # Apply changes to files
                for change in merge_result.changes:
                    file_path = self.root / change["file"]

                    # Create backup
                    if backup:
                        backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
                        backup_path.write_text(
                            file_path.read_text(encoding="utf-8"), encoding="utf-8"
                        )

                    # Apply change
                    content = file_path.read_text(encoding="utf-8")
                    new_content = content.replace(
                        change["original"], change["replacement"]
                    )
                    file_path.write_text(new_content, encoding="utf-8")

                results.append(merge_result.to_dict())
            except Exception as e:
                logger.error(f"Failed to apply merge: {e}")

        return {
            "merged_count": len(results),
            "results": results,
        }
