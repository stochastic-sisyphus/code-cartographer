# Code Cartographer - GitHub Copilot Instructions

Code Cartographer is a Python-based static analysis tool for visualizing and analyzing complex Python codebases. It detects code variants, clones, and partial rewrites using AST-based analysis, SHA-256 hashing, and semantic similarity via sentence-transformers.

## Code Standards

### Required Before Each Commit
- Run `black code_cartographer/ tests/` to format code
- Run `ruff check code_cartographer/ tests/` to lint
- Type check with `mypy code_cartographer/ --ignore-missing-imports` (non-blocking)

### Development Flow
- **Build**: `pip install -r requirements.txt -r requirements-dev.txt`
- **Test**: `pytest tests/ -v --cov=code_cartographer`
- **Full CI check**: Format → Lint → Type Check → Test → CLI Validation

## Repository Structure
- `code_cartographer/`: Main package
  - `cli.py`: Command-line interface using Click
  - `core/`: Core analysis modules (analyzer, reporter, variant_analyzer, dependency_analyzer, variable_analyzer, visualizer)
- `tests/`: Test suite (analyzer, CLI, functionality, variant tests)
- `templates/`: Jinja2 HTML templates for dashboards
- `examples/mini_repo/`: Integration test data
- `pyproject.toml`: Hatchling build config, Black/Ruff/MyPy settings

## Key Guidelines

1. **Minimal Changes**: Make surgical changes to existing code. Don't refactor unnecessarily.
2. **Always Use pathlib.Path**: Cross-platform compatibility required for all file operations.
3. **AST Parsing**: Uses Python's `ast` module. Be aware of version-specific syntax compatibility (Python 3.8+).
4. **Error Handling**: Gracefully handle parsing errors. Continue analysis even if individual files fail. Log warnings for skipped files.
5. **Testing**: Always run `pytest tests/ -v` after changes. Use `examples/mini_repo/` for integration tests.
6. **Type Hints**: Add type hints following existing patterns. MyPy runs with `--ignore-missing-imports`.
7. **Backward Compatibility**: Maintain compatibility with existing JSON output schemas.
8. **Memory Usage**: Large projects with many variants consume significant memory. Consider this for new features.
9. **Dependencies**: Core analysis should work without Graphviz. Visualization features should fail gracefully if Graphviz is missing.
10. **Documentation**: Update README.md and guide.md for user-facing changes.

## Architecture Patterns

**Analyzer Pattern**: Core analyzers in `code_cartographer/core/` are stateful classes with:
- Initialization taking project paths as parameters
- `analyze()` methods returning structured results
- Support for exclusion patterns for filtering files

**Reporter Pattern**: `reporter.py` generates multiple output formats:
- Markdown summaries
- JSON data structures  
- HTML dashboards using Jinja2 templates

**Variant Detection**: Uses semantic analysis and SHA-256 hashing:
- Normalizes code for comparison
- Calculates similarity scores using sentence-transformers
- Groups similar code blocks
- Generates merge suggestions

## Common Pitfalls

1. **Import Cycles**: Be careful adding cross-module imports in `core/`
2. **Complexity Metrics**: Uses `radon` library for cyclomatic complexity and maintainability index
3. **NLP Models**: sentence-transformers downloads models on first use. This can be slow in CI.
4. **CLI Interface**: Uses `rich` library for formatted terminal output. Uses Click for command definitions.

## Testing Requirements

- **Unit Tests**: Test individual analyzers and components
- **Integration Tests**: Use `examples/mini_repo/` for full workflow validation
- **CLI Tests**: Verify command functionality via `python -m code_cartographer.cli`
- **Coverage**: Aim for high coverage on core analysis modules (CLI and visualization may have lower coverage)
