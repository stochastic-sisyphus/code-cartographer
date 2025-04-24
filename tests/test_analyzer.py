"""Tests for the code analyzer module."""

from pathlib import Path

import pytest

from code_cartographer.core.analyzer import ProjectAnalyzer


def test_project_analyzer_init():
    """Test ProjectAnalyzer initialization."""
    analyzer = ProjectAnalyzer(root=Path("."))
    assert analyzer.root == Path(".")
    assert analyzer.exclude_patterns == []


def test_project_analyzer_with_exclude():
    """Test ProjectAnalyzer with exclude patterns."""
    exclude = ["*.pyc", "__pycache__"]
    analyzer = ProjectAnalyzer(root=Path("."), exclude_patterns=exclude)
    assert analyzer.exclude_patterns == exclude


@pytest.fixture
def sample_code(tmp_path):
    """Create a temporary Python file for testing."""
    code = '''
def hello():
    """Say hello."""
    print("Hello, world!")

def goodbye():
    """Say goodbye."""
    print("Goodbye, world!")
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(code)
    return file_path


def test_project_analyzer_execute(sample_code):
    """Test basic code analysis execution."""
    analyzer = ProjectAnalyzer(root=sample_code.parent)
    result = analyzer.execute()

    assert "files" in result
    assert "dependencies" in result
    assert len(result["files"]) == 1

    file_info = result["files"][0]
    assert file_info["path"] == str(sample_code.relative_to(sample_code.parent))
    assert file_info["functions"] == ["hello", "goodbye"]
