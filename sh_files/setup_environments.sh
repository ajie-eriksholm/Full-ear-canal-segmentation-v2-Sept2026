#!/bin/bash
# ==============================================================================
# Environment Setup Script
# ==============================================================================
# This script automatically creates all required environments:
# - seg_env      (venv)  - for P1 preprocessing
# - landmark_env (venv)  - for P2-P4 preprocessing, inference, postprocessing
# - metric_env   (conda) - for metric extraction (vmtk centerline analysis)
#
# Usage:
#   ./setup_environments.sh
#
# Requirements:
#   - Python 3.x installed
#   - pip installed
#   - conda installed (for metric_env / vmtk)
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Environment paths (at project root)
SEG_ENV="$PROJECT_ROOT/seg_env"
LANDMARK_ENV="$PROJECT_ROOT/landmark_env"

# Metric environment (conda)
METRIC_ENV_NAME="metric_env"
METRIC_PYTHON_VERSION="3.11"

# Requirement files (in env_req/ folder)
SEG_REQ="$PROJECT_ROOT/env_req/seg_env_req.txt"
LANDMARK_REQ="$PROJECT_ROOT/env_req/landmark_env_req.txt"
METRIC_REQ="$PROJECT_ROOT/env_req/metric_env_req.txt"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}=================================================="
    echo -e "$1"
    echo -e "==================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ==============================================================================
# Main Setup
# ==============================================================================

print_header "ENVIRONMENT SETUP"

echo "This script will create the following environments:"
echo "  1. seg_env      (venv)  - for P1 preprocessing (TotalSegmentator, SimpleITK)"
echo "  2. landmark_env (venv)  - for P2-P4, inference, postprocessing (PyTorch, MONAI, PyVista)"
echo "  3. metric_env   (conda) - for metric extraction (vmtk, centerline analysis)"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
print_info "Found: $PYTHON_VERSION"
echo ""

# Check if requirement files exist
if [ ! -f "$SEG_REQ" ]; then
    print_error "Requirement file not found: $SEG_REQ"
    exit 1
fi

if [ ! -f "$LANDMARK_REQ" ]; then
    print_error "Requirement file not found: $LANDMARK_REQ"
    exit 1
fi

if [ ! -f "$METRIC_REQ" ]; then
    print_error "Requirement file not found: $METRIC_REQ"
    exit 1
fi

print_success "Found all requirement files"
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Setup cancelled by user"
    exit 0
fi

# ==============================================================================
# Create seg_env
# ==============================================================================

print_header "STEP 1/6: Creating seg_env"

if [ -d "$SEG_ENV" ]; then
    print_warning "seg_env already exists at: $SEG_ENV"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing seg_env..."
        rm -rf "$SEG_ENV"
    else
        print_warning "Skipping seg_env creation"
        SKIP_SEG_ENV=true
    fi
fi

if [ "$SKIP_SEG_ENV" != true ]; then
    print_info "Creating virtual environment at: $SEG_ENV"
    python3 -m venv "$SEG_ENV"
    
    if [ $? -eq 0 ]; then
        print_success "seg_env created successfully"
    else
        print_error "Failed to create seg_env"
        exit 1
    fi
fi

# ==============================================================================
# Install seg_env requirements
# ==============================================================================

print_header "STEP 2/6: Installing seg_env requirements"

if [ "$SKIP_SEG_ENV" != true ]; then
    print_info "Activating seg_env..."
    source "$SEG_ENV/bin/activate"
    
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_info "Installing requirements from: $SEG_REQ"
    echo "This may take several minutes..."
    pip install -r "$SEG_REQ"
    
    if [ $? -eq 0 ]; then
        print_success "seg_env requirements installed successfully"
    else
        print_error "Failed to install seg_env requirements"
        exit 1
    fi
    
    deactivate
else
    print_warning "Skipped seg_env installation"
fi

# ==============================================================================
# Create landmark_env
# ==============================================================================

print_header "STEP 3/6: Creating landmark_env"

if [ -d "$LANDMARK_ENV" ]; then
    print_warning "landmark_env already exists at: $LANDMARK_ENV"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing landmark_env..."
        rm -rf "$LANDMARK_ENV"
    else
        print_warning "Skipping landmark_env creation"
        SKIP_LANDMARK_ENV=true
    fi
fi

if [ "$SKIP_LANDMARK_ENV" != true ]; then
    print_info "Creating virtual environment at: $LANDMARK_ENV"
    python3 -m venv "$LANDMARK_ENV"
    
    if [ $? -eq 0 ]; then
        print_success "landmark_env created successfully"
    else
        print_error "Failed to create landmark_env"
        exit 1
    fi
fi

# ==============================================================================
# Install landmark_env requirements
# ==============================================================================

print_header "STEP 4/6: Installing landmark_env requirements"

if [ "$SKIP_LANDMARK_ENV" != true ]; then
    print_info "Activating landmark_env..."
    source "$LANDMARK_ENV/bin/activate"
    
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    print_info "Installing requirements from: $LANDMARK_REQ"
    echo "This may take several minutes..."
    pip install -r "$LANDMARK_REQ"
    
    if [ $? -eq 0 ]; then
        print_success "landmark_env requirements installed successfully"
    else
        print_error "Failed to install landmark_env requirements"
        exit 1
    fi
    
    deactivate
else
    print_warning "Skipped landmark_env installation"
fi

# ==============================================================================
# Create metric_env (conda)
# ==============================================================================

print_header "STEP 5/6: Creating metric_env (conda)"

# Check conda is available
if ! command -v conda &> /dev/null; then
    print_error "conda not found. Please install Anaconda or Miniconda first."
    print_info "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_info "Found: $(conda --version)"

if conda env list | grep -qw "$METRIC_ENV_NAME"; then
    print_warning "metric_env already exists"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing metric_env..."
        conda env remove -n "$METRIC_ENV_NAME" -y
    else
        print_warning "Skipping metric_env creation"
        SKIP_METRIC_ENV=true
    fi
fi

if [ "$SKIP_METRIC_ENV" != true ]; then
    print_info "Creating conda environment with Python $METRIC_PYTHON_VERSION and vmtk..."
    echo "This may take several minutes as conda resolves dependencies..."
    conda create -n "$METRIC_ENV_NAME" -c conda-forge --override-channels \
        python="$METRIC_PYTHON_VERSION" vmtk -y

    if [ $? -eq 0 ]; then
        print_success "metric_env created successfully"
    else
        print_error "Failed to create metric_env"
        exit 1
    fi
fi

# ==============================================================================
# Install metric_env pip requirements + ITK symlink fix
# ==============================================================================

print_header "STEP 6/6: Installing metric_env pip requirements"

if [ "$SKIP_METRIC_ENV" != true ]; then
    print_info "Activating metric_env..."
    eval "$(conda shell.bash hook)"
    conda activate "$METRIC_ENV_NAME"

    print_info "Installing pip requirements from: $METRIC_REQ"
    pip install -r "$METRIC_REQ"

    if [ $? -eq 0 ]; then
        print_success "metric_env pip requirements installed successfully"
    else
        print_error "Failed to install metric_env pip requirements"
        exit 1
    fi

    # Fix ITK version symlinks for vmtk compatibility
    # vmtk was built against ITK 5.3 but conda-forge provides ITK 5.4
    print_info "Fixing ITK version symlinks for vmtk compatibility..."
    cd "$CONDA_PREFIX/lib"
    for f in *-5.4.so.1; do
        link="${f/-5.4.so.1/-5.3.so.1}"
        [ ! -e "$link" ] && ln -s "$f" "$link"
    done
    for f in *-5.4.so; do
        link="${f/-5.4.so/-5.3.so}"
        [ ! -e "$link" ] && ln -s "$f" "$link"
    done
    cd "$PROJECT_ROOT"
    print_success "ITK symlinks created"

    # Verify imports
    print_info "Verifying metric_env imports..."
    python -c "from vmtk import vmtkscripts; print('  vmtk OK')" && \
    python -c "import SimpleITK; print('  SimpleITK OK')" && \
    python -c "import pyvista; print('  pyvista OK')" && \
    print_success "metric_env verification passed" || \
    print_warning "Some metric_env imports failed - check the environment"

    conda deactivate
else
    print_warning "Skipped metric_env installation"
fi

# ==============================================================================
# Summary
# ==============================================================================

print_header "SETUP COMPLETE!"

echo "Environments created:"
echo ""
if [ "$SKIP_SEG_ENV" != true ]; then
    print_success "seg_env:      $SEG_ENV"
    echo "  → Activate with: source seg_env/bin/activate"
    echo "  → Used for: P1 preprocessing"
else
    print_warning "seg_env:      Skipped (already exists)"
fi
echo ""
if [ "$SKIP_LANDMARK_ENV" != true ]; then
    print_success "landmark_env: $LANDMARK_ENV"
    echo "  → Activate with: source landmark_env/bin/activate"
    echo "  → Used for: P2-P4, inference, postprocessing"
else
    print_warning "landmark_env: Skipped (already exists)"
fi
echo ""
if [ "$SKIP_METRIC_ENV" != true ]; then
    print_success "metric_env:   (conda environment)"
    echo "  → Activate with: conda activate metric_env"
    echo "  → Used for: metric extraction (vmtk centerline analysis)"
else
    print_warning "metric_env:   Skipped (already exists)"
fi
echo ""
print_info "You can now run the full pipeline using:"
echo "  ./sh_files/run_full_pipeline.sh"
echo ""
print_header "READY TO GO!"
