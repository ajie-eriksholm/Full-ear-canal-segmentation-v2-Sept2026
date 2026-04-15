#!/bin/bash
# ==============================================================================
# Inference and Postprocessing Pipeline
# ==============================================================================
# This script runs model inference and postprocessing in sequence
# 
# Usage:
#   ./run_inference_and_postprocessing.sh
#
# Prerequisites:
#   - Preprocessing pipeline (P1-P4) must be completed
#   - Inference scans should be available in OUTPUT_DIR/Preprocessing/P4_Normalized_Ears/
#
# Configuration:
#   Before running, edit the configuration paths in:
#   - model/test_tissue_air.py (inference configuration)
#   - postprocessing/generate_results.py (postprocessing configuration)
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Edit these variables as needed
# ==============================================================================

# Python environment (used for both inference and postprocessing)
LANDMARK_ENV="/home/sbeo/Full-ear-canal-segmentation/landmark_env"

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Script locations
INFERENCE_SCRIPT="$PROJECT_ROOT/model/test_tissue_air.py"
POSTPROCESSING_SCRIPT="$PROJECT_ROOT/postprocessing/generate_results.py"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

check_file_exists() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
}

check_env_exists() {
    if [ ! -d "$1" ]; then
        print_error "Python environment not found: $1"
        echo "Please create the environment first using setup_environments.sh"
        exit 1
    fi
}

# ==============================================================================
# Pre-flight Checks
# ==============================================================================

print_header "Pre-flight Checks"

# Check if Python environment exists
check_env_exists "$LANDMARK_ENV"
print_success "Python environment found: $LANDMARK_ENV"

# Check if scripts exist
check_file_exists "$INFERENCE_SCRIPT"
print_success "Inference script found: $INFERENCE_SCRIPT"

check_file_exists "$POSTPROCESSING_SCRIPT"
print_success "Postprocessing script found: $POSTPROCESSING_SCRIPT"

# ==============================================================================
# STEP 1: Run Inference (test_tissue_air.py)
# ==============================================================================

print_header "STEP 1: Running Model Inference"
echo "Script: $INFERENCE_SCRIPT"
echo "This will generate segmentation masks and landmark predictions..."
echo ""

# Activate environment and run inference
source "$LANDMARK_ENV/bin/activate"

if ! python "$INFERENCE_SCRIPT"; then
    print_error "Inference failed!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check that the configuration paths in $INFERENCE_SCRIPT are correct"
    echo "2. Verify that the test data exists in the specified TEST_DIR"
    echo "3. Ensure the trained model exists at the specified MODEL_PATH"
    echo "4. Check the logs above for specific error messages"
    exit 1
fi

print_success "Inference completed successfully!"

# ==============================================================================
# STEP 2: Run Postprocessing (generate_results.py)
# ==============================================================================

print_header "STEP 2: Running Postprocessing"
echo "Script: $POSTPROCESSING_SCRIPT"
echo "This will generate markup JSONs, NIfTI masks, and STL files..."
echo ""

if ! python "$POSTPROCESSING_SCRIPT"; then
    print_error "Postprocessing failed!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check that the configuration paths in $POSTPROCESSING_SCRIPT are correct"
    echo "2. Verify that inference outputs exist (predicted_landmark_coordinates.csv and masks)"
    echo "3. Check if the FH plane landmarks CSV exists (optional but recommended)"
    echo "4. Check the logs above for specific error messages"
    exit 1
fi

print_success "Postprocessing completed successfully!"

# ==============================================================================
# Summary
# ==============================================================================

print_header "Pipeline Complete!"

echo "[OK] Inference: Segmentation masks and landmark predictions generated"
echo "[OK] Postprocessing: Markup JSONs, NIfTI masks, and STL files created"
echo ""
echo "Next steps:"
echo "  - Review the generated outputs in the configured output directories"
echo "  - Check markup JSON files for landmark coordinates"
echo "  - Visualize STL files in 3D software (e.g., 3D Slicer, MeshLab)"
echo ""

print_success "All steps completed successfully!"
