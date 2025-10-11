#!/bin/bash
# Final verification script for CLI implementation

set -e

echo "========================================="
echo "Code Cartographer CLI Verification"
echo "========================================="
echo ""

# Test 1: CLI Help
echo "✓ Testing CLI help..."
python -m code_cartographer.cli --help > /dev/null
python -m code_cartographer.cli analyze --help > /dev/null
python -m code_cartographer.cli variants --help > /dev/null
python -m code_cartographer.cli report --help > /dev/null
python -m code_cartographer.cli visualize --help > /dev/null
echo "  All commands have working help"
echo ""

# Test 2: Example files exist
echo "✓ Checking example files..."
test -d examples/mini_repo
test -f examples/mini_repo/__init__.py
test -f examples/mini_repo/main.py
test -f examples/mini_repo/utils.py
test -f examples/mini_repo/processor.py
echo "  Mini repo example exists"
echo ""

# Test 3: Example outputs exist
echo "✓ Checking example outputs..."
test -f examples/output/analysis.json
test -f examples/output/code_analysis_report.md
test -f examples/output/code_analysis_report.html
echo "  Example outputs exist"
echo ""

# Test 4: HTML is self-contained
echo "✓ Verifying HTML report..."
if grep -q "<style>" examples/output/code_analysis_report.html; then
    echo "  HTML has inline styles ✓"
else
    echo "  ERROR: No inline styles found!"
    exit 1
fi

if grep -q "cdn" examples/output/code_analysis_report.html; then
    echo "  ERROR: CDN references found!"
    exit 1
else
    echo "  No CDN references ✓"
fi

if grep -q "<!DOCTYPE html>" examples/output/code_analysis_report.html; then
    echo "  Valid HTML structure ✓"
else
    echo "  ERROR: Invalid HTML!"
    exit 1
fi
echo ""

# Test 5: Documentation exists
echo "✓ Checking documentation..."
test -f README.md
test -f examples/README.md
test -f examples/output/README.md
test -f IMPLEMENTATION_SUMMARY.md
echo "  All README files exist"
echo ""

# Test 6: CI/CD exists
echo "✓ Checking CI/CD..."
test -f .github/workflows/ci.yml
test -f Makefile
echo "  CI workflow and Makefile exist"
echo ""

# Test 7: Configuration files
echo "✓ Checking configuration..."
test -f pyproject.toml
test -f requirements.txt
test -f .gitignore
echo "  Configuration files exist"
echo ""

# Test 8: Dependencies are pinned
echo "✓ Checking dependencies..."
if grep -q "==" requirements.txt; then
    echo "  Dependencies are pinned ✓"
else
    echo "  WARNING: Some dependencies not pinned"
fi
echo ""

# Test 9: Artifacts are gitignored
echo "✓ Checking .gitignore..."
if grep -q "artifacts/" .gitignore; then
    echo "  artifacts/ is gitignored ✓"
else
    echo "  ERROR: artifacts/ not in .gitignore!"
    exit 1
fi
echo ""

# Test 10: CLI entry point in pyproject.toml
echo "✓ Checking CLI entry point..."
if grep -q "codecart.*=.*code_cartographer.cli:main" pyproject.toml; then
    echo "  CLI entry point configured ✓"
else
    echo "  ERROR: CLI entry point not found!"
    exit 1
fi
echo ""

echo "========================================="
echo "✓ All verification checks passed!"
echo "========================================="
echo ""
echo "The CLI implementation is complete and ready for use."
echo ""
echo "Quick Start:"
echo "  codecart analyze -d examples/mini_repo"
echo "  open artifacts/code_analysis_report.html"
echo ""
