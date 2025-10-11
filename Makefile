.PHONY: help install test lint format clean analyze

help:
	@echo "Code Cartographer - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linters (flake8)"
	@echo "  format     - Format code (black, isort)"
	@echo "  clean      - Clean generated files"
	@echo "  analyze    - Run code analysis on project itself"

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=code_cartographer

lint:
	flake8 code_cartographer/ tests/ --max-line-length=88 --extend-ignore=E203,E501
	mypy code_cartographer/ --ignore-missing-imports

format:
	black code_cartographer/ tests/
	isort code_cartographer/ tests/

clean:
	rm -rf artifacts/
	rm -rf analysis_output/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

analyze:
	python -m code_cartographer.cli analyze -d . -o artifacts/self_analysis.json
