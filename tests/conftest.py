import pytest
from pathlib import Path

from code_cartographer.core.analyzer import CodeAnalyzer
from code_cartographer.core.variable_analyzer import VariableAnalyzer
from code_cartographer.core.dependency_analyzer import DependencyAnalyzer
from tests.test_functionality import create_sample_codebase

@pytest.fixture(scope="session")
def codebase_dir(tmp_path_factory) -> Path:
    return create_sample_codebase()

@pytest.fixture(scope="session")
def output_dir(codebase_dir: Path) -> Path:
    out = codebase_dir / "analysis_output"
    out.mkdir(exist_ok=True)
    return out

@pytest.fixture(scope="session")
def analysis_results(codebase_dir: Path, output_dir: Path):
    analyzer = CodeAnalyzer(codebase_dir, output_dir)
    return analyzer.analyze()


@pytest.fixture(scope="session")
def call_graph(analysis_results):
    return analysis_results.get("call_graph", {})

@pytest.fixture(scope="session")
def variable_results(codebase_dir: Path):
    analyzer = VariableAnalyzer(codebase_dir)
    return analyzer.analyze()

@pytest.fixture(scope="session")
def dependency_results(codebase_dir: Path, analysis_results, variable_results):
    analyzer = DependencyAnalyzer(codebase_dir)
    return analyzer.analyze(analysis_results.get("call_graph", {}), variable_results)

