"""
Code Variant Analyzer
====================
Detects and analyzes code variants and duplicates in Python codebases.
"""

import ast
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

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
        self.hash = hashlib.sha256(
            self.content.encode()
        ).hexdigest()
        self.normalized = self._normalize_code()
        self.tokens = word_tokenize(self.normalized.lower())
    
    def _normalize_code(self) -> str:
        """Normalize code for comparison by removing syntax noise."""
        try:
            tree = ast.parse(self.content)
        except SyntaxError:
            return self.content
            
        # Remove comments and docstrings
        for node in ast.walk(tree):
            if isinstance(node, ast.Str):
                node.s = ""
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                node.value.s = ""
                
        return ast.unparse(tree)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "path": str(self.path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "hash": self.hash
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
            "similarity": self.similarity
        }

class VariantAnalyzer:
    """Analyzes Python code for variants and duplicates."""
    
    def __init__(
        self,
        root: Path,
        semantic_threshold: float = 0.8,
        min_lines: int = 5,
        exclude_patterns: Optional[List[str]] = None
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
        
        # Load models
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            
        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def analyze(self) -> Dict:
        """Run the variant analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        # Extract code blocks
        blocks = self._extract_blocks()
        
        # Find exact duplicates
        exact_duplicates = self._find_exact_duplicates(blocks)
        
        # Find semantic variants
        semantic_variants = self._find_semantic_variants(blocks)
        
        return {
            "exact_duplicates": [
                g.to_dict() for g in exact_duplicates
            ],
            "semantic_variants": [
                g.to_dict() for g in semantic_variants
            ]
        }
    
    def _extract_blocks(self) -> List[CodeBlock]:
        """Extract code blocks from Python files."""
        blocks = []
        
        for path in self.root.rglob("*.py"):
            # Skip excluded paths
            if any(
                pattern in str(path)
                for pattern in self.exclude_patterns
            ):
                continue
                
            try:
                content = path.read_text()
                lines = content.splitlines()
                
                # Extract blocks of sufficient size
                current_block = []
                for i, line in enumerate(lines, 1):
                    if line.strip():
                        current_block.append(line)
                    elif len(current_block) >= self.min_lines:
                        blocks.append(
                            CodeBlock(
                                path=path,
                                start_line=i - len(current_block),
                                end_line=i - 1,
                                content="\n".join(current_block)
                            )
                        )
                        current_block = []
                    else:
                        current_block = []
                        
                # Handle last block
                if len(current_block) >= self.min_lines:
                    blocks.append(
                        CodeBlock(
                            path=path,
                            start_line=len(lines) - len(current_block) + 1,
                            end_line=len(lines),
                            content="\n".join(current_block)
                        )
                    )
                    
            except Exception as e:
                logger.warning(
                    f"Error processing {path}: {e}"
                )
                continue
                
        return blocks
    
    def _find_exact_duplicates(
        self,
        blocks: List[CodeBlock]
    ) -> List[VariantGroup]:
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
    
    def _find_semantic_variants(
        self,
        blocks: List[CodeBlock]
    ) -> List[VariantGroup]:
        """Find semantically similar code blocks."""
        variants = []
        
        # Encode all blocks
        encodings = self.encoder.encode(
            [b.normalized for b in blocks]
        )
        
        # Compare all pairs
        for i, block1 in enumerate(blocks):
            group = []
            
            for j, block2 in enumerate(blocks[i + 1:], i + 1):
                # Skip exact duplicates
                if block1.hash == block2.hash:
                    continue
                    
                # Calculate cosine similarity
                similarity = float(
                    encodings[i] @ encodings[j].T
                    / (
                        (encodings[i] ** 2).sum() ** 0.5
                        * (encodings[j] ** 2).sum() ** 0.5
                    )
                )
                
                if similarity >= self.semantic_threshold:
                    if not group:
                        group = [block1]
                    group.append(block2)
                    
            if group:
                variants.append(
                    VariantGroup(
                        blocks=group,
                        similarity=similarity
                    )
                )
                
        return variants
