"""Convenience helpers for running Code Cartographer quickly."""

from __future__ import annotations

import json
from pathlib import Path

from .core.analyzer import CodeAnalyzer


def quick_analyze(directory: str | Path, output: str | Path = "analysis_output/analysis.json") -> Path:
    """Run code analysis on *directory* and write results to *output*.

    Args:
        directory: Project directory to analyze.
        output: Path to write analysis results JSON.

    Returns:
        Path to the generated JSON file.
    """
    directory = Path(directory)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer = CodeAnalyzer(directory, output_path.parent)
    results = analyzer.analyze()
    output_path.write_text(json.dumps(results, indent=2, default=str))
    return output_path

