"""
CLI Smoke Tests
===============
End-to-end tests to ensure CLI functionality works correctly.
"""

import json
import subprocess
import tempfile
from pathlib import Path



def test_cli_help():
    """Test that CLI help works without errors."""
    result = subprocess.run(
        ["python3", "-m", "code_cartographer.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "Code Cartographer" in result.stdout
    assert "analyze" in result.stdout
    assert "variants" in result.stdout


def test_analyze_subcommand_help():
    """Test that analyze subcommand help works."""
    result = subprocess.run(
        ["python3", "-m", "code_cartographer.cli", "analyze", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "Root directory to analyze" in result.stdout


def test_variants_subcommand_help():
    """Test that variants subcommand help works."""
    result = subprocess.run(
        ["python3", "-m", "code_cartographer.cli", "variants", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    assert result.returncode == 0
    assert "variants" in result.stdout


def test_analyze_command_on_test_repo():
    """Test analyze command on a small test repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a minimal test repository
        test_repo = temp_path / "test_repo"
        test_repo.mkdir()

        # Create a simple Python file
        (test_repo / "main.py").write_text(
            """
def hello_world():
    '''Simple hello world function.'''
    print("Hello, World!")

class Calculator:
    '''Simple calculator class.'''
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    hello_world()
    calc = Calculator()
    print(calc.add(2, 3))
"""
        )

        # Create another file
        (test_repo / "utils.py").write_text(
            """
def format_number(num):
    '''Format a number with commas.'''
    return f"{num:,}"

def is_even(num):
    '''Check if a number is even.'''
    return num % 2 == 0
"""
        )

        # Run analysis
        output_file = temp_path / "analysis.json"
        result = subprocess.run(
            [
                "python3",
                "-m",
                "code_cartographer.cli",
                "analyze",
                "--dir",
                str(test_repo),
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Check that command succeeded
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert "Analysis complete" in result.stdout

        # Check that output file was created
        assert output_file.exists()

        # Check that output is valid JSON
        with open(output_file) as f:
            analysis_data = json.load(f)

        # Basic validation of analysis structure
        assert isinstance(analysis_data, dict)
        # The exact structure may vary, but we should have some analysis data
        assert len(analysis_data) > 0


def test_variants_command_on_test_repo():
    """Test variants command on a small test repository."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a minimal test repository with some duplicate code
        test_repo = temp_path / "test_repo"
        test_repo.mkdir()

        # Create files with similar functions
        (test_repo / "file1.py").write_text(
            """
def process_data(data):
    '''Process some data.'''
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""
        )

        (test_repo / "file2.py").write_text(
            """
def handle_data(data):
    '''Handle some data.'''
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
        )

        # Run variants analysis
        output_file = temp_path / "variants.json"
        result = subprocess.run(
            [
                "python3",
                "-m",
                "code_cartographer.cli",
                "variants",
                "--dir",
                str(test_repo),
                "--output",
                str(output_file),
                "--semantic-threshold",
                "0.7",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Check that command succeeded
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert "Variant analysis complete" in result.stdout

        # Check that output file was created
        assert output_file.exists()

        # Check that output is valid JSON
        with open(output_file) as f:
            variants_data = json.load(f)

        # Basic validation of variants structure
        assert isinstance(variants_data, dict)
