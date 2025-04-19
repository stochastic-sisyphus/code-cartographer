#!/bin/bash

# Exit on error
set -e

# Default values
PROJECT_DIR="$(pwd)"
OUTPUT_DIR="${PROJECT_DIR}/code_analysis"
DIFFS_DIR="${OUTPUT_DIR}/diffs"
SUMMARY_CSV="${OUTPUT_DIR}/summary.csv"
MERGED_DIR="${PROJECT_DIR}/merged_code"
TEMPLATE_DIR="${PROJECT_DIR}/templates"
DASHBOARD_OUT="${OUTPUT_DIR}/dashboard.html"
EXCLUDE_PATTERNS=("tests/.*" "build/.*" ".venv/.*" ".git/.*" ".cursor/.*" ".vscode/.*")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --exclude)
            # Convert comma-separated list to array
            IFS=',' read -ra EXCLUDE_PATTERNS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check Python environment
if [ ! -f ".venv/bin/activate" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    
    echo "Installing required packages..."
    pip install radon jinja2
else
    source .venv/bin/activate
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

# Only proceed if analyzer succeeded
if [ $? -eq 0 ]; then
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

    echo "✅ Code analysis pipeline complete!"
    echo
    echo "Output locations:"
    echo "- Analysis files: $OUTPUT_DIR"
    echo "  - analysis.json: Detailed analysis data"
    echo "  - analysis.md: Human-readable summary"
    echo "  - dependencies.dot: Dependency graph (use Graphviz to visualize)"
    echo "- Code variants: $DIFFS_DIR"
    echo "- Merged implementations: $MERGED_DIR"
    echo "- Analysis dashboard: $DASHBOARD_OUT"
else
    echo "❌ Analysis failed - cleaning up"
    rm -rf "$TEMP_DIR"
    exit 1
fi
