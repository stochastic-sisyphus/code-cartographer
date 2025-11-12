# Code Cartographer - GitHub Copilot Instructions

## Project Overview

Code Cartographer is a Python-based static analysis tool for visualizing and analyzing complex Python codebases. It helps developers understand code structure, detect variants, identify orphaned code, and visualize dependencies.

### Core Purpose
- Automated deep multilayer code analysis for large-scale codebases
- Detection of code variants, clones, and partial rewrites
- Complexity analysis and maintainability metrics
- Dependency graph generation
- Auto-generation of refactoring prompts

## Repository Structure

```
code-cartographer/
├── code_cartographer/       # Main package
│   ├── cli.py              # Command-line interface
│   └── core/               # Core analysis modules
│       ├── analyzer.py           # Main code analyzer
│       ├── dependency_analyzer.py # Dependency tracking
│       ├── reporter.py           # Report generation
│       ├── variable_analyzer.py  # Variable usage analysis
│       ├── variant_analyzer.py   # Code variant detection
│       └── visualizer.py         # Visualization generation
├── tests/                  # Test suite
├── templates/              # HTML templates (Jinja2)
├── examples/               # Example projects for testing
├── analyze_codebase.sh     # Shell script wrapper
└── pyproject.toml         # Project configuration
```

## Development Workflow

### Setting Up Development Environment

1. **Python Version**: Requires Python 3.8+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```
3. **System Dependencies**: Graphviz is required for visualization features
   ```bash
   sudo apt-get install graphviz graphviz-dev  # Ubuntu/Debian
   brew install graphviz                       # macOS
   ```

### Code Style and Quality

- **Formatter**: Black (line length: 88)
- **Linter**: Ruff
- **Type Checker**: MyPy (with `--ignore-missing-imports`)
- **Import Sorting**: isort (Black-compatible profile)

### Running Linters

```bash
# Format code
black code_cartographer/ tests/

# Lint
ruff check code_cartographer/ tests/

# Type check
mypy code_cartographer/ --ignore-missing-imports
```

### Testing

- **Framework**: pytest with coverage
- **Test Location**: `tests/` directory
- **Running Tests**:
  ```bash
  pytest tests/ -v --cov=code_cartographer
  ```
- **CLI Testing**: Test CLI functionality using `python -m code_cartographer.cli`

### Pre-commit Hooks

The project uses pre-commit hooks configured in `.pre-commit-config.yaml`:
- Trailing whitespace removal
- End-of-file fixer
- YAML validation
- Black formatting
- Ruff linting
- MyPy type checking

## Key Conventions and Patterns

### Code Analysis Architecture

1. **Analyzer Pattern**: Core analyzers in `code_cartographer/core/` are stateful classes that:
   - Take project paths as initialization parameters
   - Provide `analyze()` methods that return structured results
   - Support exclusion patterns for filtering files

2. **Reporter Pattern**: The `reporter.py` module generates multiple output formats:
   - Markdown summaries
   - JSON data structures
   - HTML dashboards (using Jinja2 templates)

3. **Variant Detection**: Uses semantic analysis and SHA-256 hashing:
   - Normalizes code for comparison
   - Calculates similarity scores
   - Groups similar code blocks
   - Generates merge suggestions

### Important Design Decisions

- **AST-based Analysis**: Uses Python's `ast` module for accurate code parsing
- **Hash-based Deduplication**: SHA-256 hashes for function/class signatures
- **Complexity Metrics**: Uses `radon` library for cyclomatic complexity and maintainability index
- **NLP for Variants**: Uses `sentence-transformers` for semantic similarity
- **Rich CLI**: Uses `rich` library for formatted terminal output

### Error Handling

- Gracefully handle parsing errors in source files
- Log warnings for skipped files
- Continue analysis even if individual files fail
- Provide meaningful error messages with file paths and line numbers

## Common Tasks

### Adding a New Analyzer

1. Create a new module in `code_cartographer/core/`
2. Implement an analyzer class with `analyze()` method
3. Integrate with main `CodeAnalyzer` in `analyzer.py`
4. Add tests in `tests/`
5. Update CLI in `cli.py` if needed

### Adding CLI Commands

1. Edit `code_cartographer/cli.py`
2. Use Click decorators for command definitions
3. Follow existing patterns for error handling and output
4. Add corresponding tests in `tests/test_cli_smoke.py`

### Modifying Reports

1. Update `code_cartographer/core/reporter.py`
2. Modify Jinja2 templates in `templates/` for HTML output
3. Ensure backward compatibility with existing JSON schemas
4. Update examples in README.md

## Testing Approach

### Test Categories

1. **Unit Tests**: Test individual analyzers and components
2. **Integration Tests**: Test full analysis workflows
3. **CLI Tests**: Verify command-line interface behavior
4. **Functionality Tests**: End-to-end feature validation

### Test Data

- Use `examples/mini_repo/` for integration tests
- Create minimal test cases for unit tests
- Mock file system operations where appropriate

### Coverage Requirements

- Aim for high coverage on core analysis modules
- CLI and visualization code may have lower coverage
- CI runs coverage reporting via pytest-cov

## Dependencies

### Core Dependencies
- `click`: CLI framework
- `rich`: Terminal formatting
- `pyyaml`: Configuration parsing
- `nltk`: Natural language processing
- `sentence-transformers`: Semantic similarity

### Development Dependencies
- `pytest`, `pytest-cov`: Testing
- `black`: Code formatting
- `ruff`: Linting
- `mypy`: Type checking
- `pre-commit`: Git hooks

### Optional Dependencies
- Graphviz: Required for dependency graph visualization

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

Runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

Test matrix:
- Python 3.8, 3.9, 3.10, 3.11

Steps:
1. Lint with ruff
2. Format check with black
3. Type check with mypy (non-blocking)
4. Run pytest with coverage
5. Test CLI functionality
6. Upload coverage to Codecov

### Publishing (`.github/workflows/publish.yml`)

- Triggered on version tags
- Builds package with hatchling
- Publishes to PyPI

## Important Notes for AI Assistants

1. **Minimal Changes**: Make surgical changes to existing code. Don't refactor unnecessarily.

2. **Testing**: Always run tests after changes:
   ```bash
   pytest tests/ -v
   ```

3. **Code Style**: Run formatters before committing:
   ```bash
   black code_cartographer/ tests/
   ruff check --fix code_cartographer/ tests/
   ```

4. **Type Hints**: Add type hints for new functions, following existing patterns.

5. **Documentation**: Update README.md and guide.md for user-facing changes.

6. **Backward Compatibility**: Maintain compatibility with existing JSON output schemas.

7. **Performance**: Code analysis can be resource-intensive. Consider memory usage for large codebases.

8. **Error Messages**: Provide clear, actionable error messages with file paths and context.

## Common Pitfalls

1. **Import Cycles**: Be careful when adding cross-module imports in `core/`
2. **File Path Handling**: Always use `pathlib.Path` for cross-platform compatibility
3. **AST Parsing**: Some Python syntax features may not parse on older Python versions
4. **Memory Usage**: Large projects with many variants can consume significant memory
5. **Graphviz Dependency**: Features requiring Graphviz should fail gracefully if not installed

## Project Philosophy

This tool exists to help developers manage chaotic, duplicated, or fragmented codebases. The author's perspective:
- Perfection is the goal, even if unattainable
- Analysis should be comprehensive and deep
- Output should be actionable and clear
- The tool should handle messy real-world code gracefully

When contributing, maintain this philosophy of thoroughness and practicality.
