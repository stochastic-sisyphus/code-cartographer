#!/bin/bash

# Exit on error
set -e

# Cleanup on error
trap 'rm -rf "$TEMP_DIR"' ERR

# Default values
PROJECT_DIR="${1:-$(pwd)}"
OUTPUT_DIR="${PROJECT_DIR}/code_analysis"
DIFFS_DIR="${OUTPUT_DIR}/diffs"
SUMMARY_CSV="${OUTPUT_DIR}/summary.csv"
MERGED_DIR="${PROJECT_DIR}/merged_code"
TEMPLATE_DIR="${PROJECT_DIR}/templates"
DASHBOARD_OUT="${OUTPUT_DIR}/dashboard.html"
EXCLUDE_PATTERNS=("tests/.*" "build/.*" ".venv/.*" ".git/.*" ".cursor/.*" ".vscode/.*")

# Check if mise is available
if command -v mise &> /dev/null; then
    echo "Using mise for environment management..."
    mise install
    mise run install
else
    echo "mise not found, falling back to manual install..."
    if [ ! -f ".venv/bin/activate" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate
    pip install -e .
fi

# Create output directories
echo "Creating output directories..."
mkdir -p "$OUTPUT_DIR" "$DIFFS_DIR" "$MERGED_DIR" "$TEMPLATE_DIR"

# Copy dashboard template if it doesn't exist
if [ ! -f "${TEMPLATE_DIR}/dashboard.html.j2" ]; then
    echo "Copying dashboard template..."
    cp "$(dirname "$0")/templates/dashboard.html.j2" "${TEMPLATE_DIR}/"
fi

# Create temp directory for analyzer output
TEMP_DIR="${OUTPUT_DIR}/temp"
mkdir -p "$TEMP_DIR"

echo "Running code analysis pipeline..."

# Step 1: Deep code analysis
echo "Step 1: Running deep code analyzer..."
python "$(dirname "$0")/code_analyzer_engine.py" \
    -d "$PROJECT_DIR" \
    --output "${TEMP_DIR}/analysis.json" \
    --markdown "${TEMP_DIR}/analysis.md" \
    --graphviz "${TEMP_DIR}/dependencies.dot" \
    --exclude "${EXCLUDE_PATTERNS[@]}"

echo "Moving analysis files to final location..."
mv "${TEMP_DIR}"/* "${OUTPUT_DIR}/"
rm -rf "${TEMP_DIR}"

echo "Step 2: Analyzing code variants..."
python "$(dirname "$0")/code_variant_analyzer.py" compare \
    --summary-dir "$OUTPUT_DIR" \
    --diffs-dir "$DIFFS_DIR" \
    --summary-csv "$SUMMARY_CSV" \
    --profile balanced \
    --include-similar

echo "Step 3: Merging similar code..."
python "$(dirname "$0")/code_variant_analyzer.py" merge \
    --summary-dir "$OUTPUT_DIR" \
    --output-dir "$MERGED_DIR" \
    --format both \
    --preserve-structure \
    --profile balanced \
    --include-similar

echo "Step 4: Generating analysis dashboard..."
python "$(dirname "$0")/code_variant_analyzer.py" dashboard \
    --summary-csv "$SUMMARY_CSV" \
    --diffs-dir "$DIFFS_DIR" \
    --template-dir "$TEMPLATE_DIR" \
    --output "$DASHBOARD_OUT" \
    --show-all-variants

echo "Code analysis pipeline complete!"
echo
echo "Output locations:"
echo "- Analysis files: $OUTPUT_DIR"
echo "  - analysis.json: Detailed analysis data"
echo "  - analysis.md: Human-readable summary"
echo "  - dependencies.dot: Dependency graph (use Graphviz to visualize)"
echo "- Code variants: $DIFFS_DIR"
echo "- Merged implementations: $MERGED_DIR"
echo "- Analysis dashboard: $DASHBOARD_OUT"
