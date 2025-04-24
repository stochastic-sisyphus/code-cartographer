"""Tests for the variant analyzer module."""

from pathlib import Path

import pytest

from code_cartographer.core.variant_analyzer import CodeBlock, VariantAnalyzer


def test_variant_analyzer_init():
    """Test VariantAnalyzer initialization."""
    analyzer = VariantAnalyzer(root=Path("."))
    assert analyzer.root == Path(".")
    assert analyzer.exclude_patterns == []
    assert analyzer.semantic_threshold == 0.8
    assert analyzer.min_lines == 5


def test_code_block_init():
    """Test CodeBlock initialization and normalization."""
    code = '''
def test():
    """Test docstring."""
    # Test comment
    print("Hello")
'''
    block = CodeBlock(path=Path("test.py"), start_line=1, end_line=5, content=code)

    assert block.path == Path("test.py")
    assert block.start_line == 1
    assert block.end_line == 5
    assert block.content == code
    assert isinstance(block.hash, str)
    assert len(block.hash) == 64  # SHA-256 hash length


@pytest.fixture
def duplicate_code(tmp_path):
    """Create temporary Python files with duplicate code."""
    code1 = '''
def process_data(data):
    """Process the data."""
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
    code2 = '''
def transform_items(items):
    """Transform the items."""
    result = []
    for item in items:
        result.append(item * 2)
    return result
'''

    file1 = tmp_path / "module1.py"
    file2 = tmp_path / "module2.py"
    file1.write_text(code1)
    file2.write_text(code2)
    return tmp_path


def test_variant_analyzer_find_variants(duplicate_code):
    """Test finding code variants."""
    analyzer = VariantAnalyzer(root=duplicate_code, semantic_threshold=0.8, min_lines=3)

    result = analyzer.analyze()

    assert "semantic_variants" in result
    assert "exact_duplicates" in result

    # The core logic should be detected as semantically similar
    variants = result["semantic_variants"]
    assert len(variants) > 0

    # Check variant structure
    variant = variants[0]
    assert "blocks" in variant
    assert "similarity" in variant
    assert variant["similarity"] >= 0.8
