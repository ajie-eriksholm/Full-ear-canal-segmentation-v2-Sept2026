#!/bin/bash
# ============================================================================
# Setup metric_env conda environment for metric_extraction pipeline.
#
# vmtk is only available via conda (not pip), so we need a conda environment.
# vmtk 1.5.0 on conda-forge supports Python 3.9, 3.10, 3.11.
#
# Usage:
#   bash sh_files/setup_metric_env.sh
# ============================================================================

set -e

ENV_NAME="metric_env"
PYTHON_VERSION="3.11"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ_FILE="$PROJECT_ROOT/metric_extraction/requirements.txt"

echo "============================================"
echo "Setting up conda environment: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

# Check conda is available
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -qw "$ENV_NAME"; then
    echo ""
    echo "WARNING: Environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Keeping existing environment. Skipping creation."
        echo "To install remaining pip packages, activate the env and run:"
        echo "  conda activate $ENV_NAME"
        echo "  pip install numpy pandas pyvista scikit-image scipy trimesh SimpleITK"
        exit 0
    fi
fi

echo ""
echo "Step 1: Creating conda environment with vmtk..."
echo "  (This may take several minutes as conda resolves dependencies)"
conda create -n "$ENV_NAME" -c conda-forge python="$PYTHON_VERSION" vmtk -y

echo ""
echo "Step 2: Installing remaining pip packages..."
# Activate the conda env and install pip packages
# (numpy, vtk, itk are already installed by vmtk's conda dependencies)
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

pip install pandas pyvista scikit-image scipy trimesh SimpleITK

echo ""
echo "Step 3: Fixing ITK version symlinks for vmtk compatibility..."
# vmtk was built against ITK 5.3 but conda-forge provides ITK 5.4.
# Create compatibility symlinks so vmtk can find the libraries.
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
echo "  Done."

echo ""
echo "Step 4: Verifying imports..."
python -c "from metric_extraction.pipeline import main; print('  All imports OK')" 2>/dev/null \
    || (cd "$PROJECT_ROOT/metric_extraction" && python -c "from pipeline import main; print('  All imports OK')")

echo ""
echo "============================================"
echo "Environment '$ENV_NAME' created successfully!"
echo ""
echo "To activate:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the metric extraction pipeline:"
echo "  conda activate $ENV_NAME"
echo "  cd $PROJECT_ROOT/metric_extraction"
echo "  python run_pipeline.py --input /path/to/Results --output /path/to/metric_output"
echo "============================================"
