"""
End-to-end tests for the CLI commands.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(
        ["python", "-m", "code_cartographer.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Code Cartographer" in result.stdout
    assert "analyze" in result.stdout
    assert "variants" in result.stdout


def test_cli_analyze_help():
    """Test that analyze command help works."""
    result = subprocess.run(
        ["python", "-m", "code_cartographer.cli", "analyze", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "analyze" in result.stdout


def test_cli_analyze_mini_repo():
    """Test analyzing the mini example repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "analysis.json"

        # Run the analyze command
        result = subprocess.run(
            [
                "python",
                "-m",
                "code_cartographer.cli",
                "analyze",
                "-d",
                "examples/mini_repo",
                "-o",
                str(output_json),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Check command succeeded
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check JSON output was created
        assert output_json.exists(), "JSON output file not created"

        # Check JSON is valid and has expected structure
        with open(output_json) as f:
            analysis = json.load(f)

        assert "files" in analysis
        assert len(analysis["files"]) > 0
        assert "call_graph" in analysis
        assert "orphans" in analysis

        # Check Markdown report was created
        markdown_report = output_json.parent / "code_analysis_report.md"
        assert markdown_report.exists(), "Markdown report not created"

        # Check HTML report was created
        html_report = output_json.parent / "code_analysis_report.html"
        assert html_report.exists(), "HTML report not created"

        # Verify HTML report content
        html_content = html_report.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "Code Analysis Report" in html_content


def test_cli_analyze_idempotent():
    """Test that running analyze twice produces the same results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json1 = Path(tmpdir) / "analysis1.json"
        output_json2 = Path(tmpdir) / "analysis2.json"

        # Run analyze first time
        subprocess.run(
            [
                "python",
                "-m",
                "code_cartographer.cli",
                "analyze",
                "-d",
                "examples/mini_repo",
                "-o",
                str(output_json1),
            ],
            capture_output=True,
            timeout=60,
        )

        # Run analyze second time
        subprocess.run(
            [
                "python",
                "-m",
                "code_cartographer.cli",
                "analyze",
                "-d",
                "examples/mini_repo",
                "-o",
                str(output_json2),
            ],
            capture_output=True,
            timeout=60,
        )

        # Compare the JSON outputs (ignoring potential timestamps)
        with open(output_json1) as f1, open(output_json2) as f2:
            analysis1 = json.load(f1)
            analysis2 = json.load(f2)

        # Remove any git_revision which might differ
        analysis1.pop("git_revision", None)
        analysis2.pop("git_revision", None)

        # Check that the core analysis is the same
        assert analysis1["files"] == analysis2["files"]
        assert analysis1["call_graph"] == analysis2["call_graph"]


def test_cli_analyze_current_directory():
    """Test analyzing current directory with default arguments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to a test directory with some Python files
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir("examples/mini_repo")

            # Run analyze with minimal arguments
            result = subprocess.run(
                ["python", "-m", "code_cartographer.cli", "analyze"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Should succeed
            assert result.returncode == 0

            # Check artifacts directory was created
            assert Path("artifacts").exists()
            assert Path("artifacts/code_analysis.json").exists()

            # Clean up
            import shutil

            shutil.rmtree("artifacts", ignore_errors=True)

        finally:
            os.chdir(old_cwd)


def test_html_report_standalone():
    """Test that HTML report can be opened without a server."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "analysis.json"

        # Run the analyze command
        subprocess.run(
            [
                "python",
                "-m",
                "code_cartographer.cli",
                "analyze",
                "-d",
                "examples/mini_repo",
                "-o",
                str(output_json),
            ],
            capture_output=True,
            timeout=60,
        )

        # Read the HTML report
        html_report = output_json.parent / "code_analysis_report.html"
        html_content = html_report.read_text()

        # Check that it doesn't require external resources
        # (no CDN links, all CSS/JS should be inline or local)
        assert "cdn.jsdelivr.net" not in html_content
        assert "cdnjs.cloudflare.com" not in html_content

        # Check that it has inline styles
        assert "<style>" in html_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
