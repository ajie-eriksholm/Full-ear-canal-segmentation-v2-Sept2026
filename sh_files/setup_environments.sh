#!/bin/bash
# ==============================================================================
# Environment Setup Script
# ==============================================================================
# This script automatically creates both required virtual environments:
# - seg_env (for P1 preprocessing)
# - landmark_env (for P2, P3, P4 preprocessing)
#
# Usage:
#   ./setup_environments.sh
#
# Requirements:
#   - Python 3.x installed
#   - pip installed
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

# Environment paths
SEG_ENV="$SCRIPT_DIR/seg_env"
LANDMARK_ENV="$SCRIPT_DIR/landmark_env"

# Requirement files
SEG_REQ="$SCRIPT_DIR/seg_env_req.txt"
LANDMARK_REQ="$SCRIPT_DIR/landmark_env_req.txt"

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

print_header "PREPROCESSING ENVIRONMENT SETUP"

echo "This script will create two virtual environments:"
echo "  1. seg_env      - for P1 preprocessing (TotalSegmentator, SimpleITK)"
echo "  2. landmark_env - for P2-P4 preprocessing (PyTorch, MONAI, PyVista)"
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

print_success "Found both requirement files"
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

print_header "STEP 1/4: Creating seg_env"

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

print_header "STEP 2/4: Installing seg_env requirements"

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

print_header "STEP 3/4: Creating landmark_env"

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

print_header "STEP 4/4: Installing landmark_env requirements"

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
# Summary
# ==============================================================================

print_header "SETUP COMPLETE!"

echo "Virtual environments created:"
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
    echo "  → Used for: P2, P3, P4 preprocessing"
else
    print_warning "landmark_env: Skipped (already exists)"
fi
echo ""
print_info "You can now run the preprocessing pipeline using:"
echo "  ./sh_files/run_full_preprocessing.sh"
echo ""
print_header "READY TO PREPROCESS!"
