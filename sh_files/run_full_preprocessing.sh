#!/bin/bash
# ==============================================================================
# Full Preprocessing Pipeline
# ==============================================================================
# This script runs all preprocessing steps (P1-P4) in sequence
# 
# Usage:
#   ./sh_files/run_full_preprocessing.sh
#
# Configuration:
#   Edit the variables below to customize paths and settings
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Edit these variables as needed
# ==============================================================================

# Directory paths
RAW_SCANS_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Raw"
PROCESSED_SCANS_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Processed-Data"
OUTPUT_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Output"


# Processing options
NO_EYES="False"              # Set to "True" if scans don't include eyes
SKIP_ALIGNMENT="False"       # Set to "True" to skip alignment step

# Python environments
SEG_ENV="/home/sbeo/Full-ear-canal-segmentation/seg_env"           # For P1
LANDMARK_ENV="/home/sbeo/Full-ear-canal-segmentation/landmark_env" # For P2, P3, P4

# Landmark detection model (for P2)
LANDMARK_MODEL="/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/best_model_2025-12-05_13-16-35.pth"

# Quality assessment filter (for P1)
ACCEPTABLE_PATIENTID_CSV="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/acceptable_patientid.csv"
PRE_QUALITY_ASSESSED="False"  # Set to "True" to only process scans in acceptable_patientid_csv

# Excluded scans (for P3)
EXCLUDED_SCANS_CSV="/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Excluded_scans_cropping.csv"

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Preprocessing scripts location
PREPROCESSING_DIR="$PROJECT_ROOT/preprocessing"

# ==============================================================================
# P1: Initial Preprocessing (Clipping, Segmentation, Cropping, Resampling)
# ==============================================================================
echo "=================================================="
echo "STEP 1/4: Running P1 - Initial Preprocessing"
echo "=================================================="
echo "Activating seg_env for P1..."
source "$SEG_ENV/bin/activate"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate seg_env at $SEG_ENV"
    exit 1
fi

echo "Python environment: $(which python)"
python --version
echo "Raw scans directory: $RAW_SCANS_DIR"
echo "Processed scans directory: $PROCESSED_SCANS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Pre-quality assessed: $PRE_QUALITY_ASSESSED"
echo "=================================================="

python "$PREPROCESSING_DIR/p1_preprocessing.py" \
    --raw_scans_dir "$RAW_SCANS_DIR" \
    --processed_scans_dir "$PROCESSED_SCANS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --acceptable_patientid_csv "$ACCEPTABLE_PATIENTID_CSV" \
    --pre_quality_assessed "$PRE_QUALITY_ASSESSED"

if [ $? -ne 0 ]; then
    echo "ERROR: P1 preprocessing failed!"
    exit 1
fi

echo ""
echo "[OK] P1 preprocessing completed successfully"
echo ""

# ==============================================================================
# P2: Landmark Detection and Alignment
# ==============================================================================
echo "=================================================="
echo "STEP 2/4: Running P2 - Landmark Detection & Alignment"
echo "=================================================="
echo "Switching to landmark_env for P2-P4..."
source "$LANDMARK_ENV/bin/activate"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate landmark_env at $LANDMARK_ENV"
    exit 1
fi

echo "Python environment: $(which python)"
python --version
echo "Processed scans directory: $PROCESSED_SCANS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Landmark model: $LANDMARK_MODEL"
echo "No eyes mode: $NO_EYES"
echo "Skip alignment: $SKIP_ALIGNMENT"
echo "=================================================="

python "$PREPROCESSING_DIR/p2_preprocessing.py" \
    --processed_scans_dir "$PROCESSED_SCANS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --landmark_model "$LANDMARK_MODEL" \
    --no_eyes "$NO_EYES" \
    --skip_alignment "$SKIP_ALIGNMENT"

if [ $? -ne 0 ]; then
    echo "ERROR: P2 preprocessing failed!"
    exit 1
fi

echo ""
echo "[OK] P2 preprocessing completed successfully"
echo ""

# ==============================================================================
# P3: ROI Cropping Around Ear Landmarks
# ==============================================================================
echo "=================================================="
echo "STEP 3/4: Running P3 - ROI Cropping"
echo "=================================================="
echo "Processed scans directory: $PROCESSED_SCANS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Excluded scans CSV: $EXCLUDED_SCANS_CSV"
echo "No eyes mode: $NO_EYES"
echo "Skip alignment: $SKIP_ALIGNMENT"
echo "=================================================="

python "$PREPROCESSING_DIR/p3_preprocessing.py" \
    --processed_scans_dir "$PROCESSED_SCANS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --excluded_scans_csv "$EXCLUDED_SCANS_CSV" \
    --no_eyes "$NO_EYES" \
    --skip_alignment "$SKIP_ALIGNMENT"

if [ $? -ne 0 ]; then
    echo "ERROR: P3 preprocessing failed!"
    exit 1
fi

echo ""
echo "[OK] P3 preprocessing completed successfully"
echo ""

# ==============================================================================
# P4: Upsampling and Normalization for Inference
# ==============================================================================
echo "=================================================="
echo "STEP 4/4: Running P4 - Upsampling & Normalization"
echo "=================================================="
echo "Processed scans directory: $PROCESSED_SCANS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="

python "$PREPROCESSING_DIR/p4_preprocessing.py" \
    --processed_scans_dir "$PROCESSED_SCANS_DIR" \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: P4 preprocessing failed!"
    exit 1
fi

echo ""
echo "[OK] P4 preprocessing completed successfully"
echo ""

# ==============================================================================
# COMPLETION
# ==============================================================================
echo "=================================================="
echo "ALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY!"
echo "=================================================="
echo "Raw scans directory:       $RAW_SCANS_DIR"
echo "Processed scans directory: $PROCESSED_SCANS_DIR"
echo "Output directory:          $OUTPUT_DIR"
echo "=================================================="
echo "Final outputs can be found in:"
echo "  - Processed data:  $PROCESSED_SCANS_DIR"
echo "  - Logs:            $OUTPUT_DIR/Logs"
echo "  - P2 Landmarks:    $OUTPUT_DIR/Preprocessing/P2_Landmarks"
echo "  - P3 Cropped ears: $OUTPUT_DIR/Preprocessing/P3_Cropped_Ears"
echo "  - P4 Normalized:   $OUTPUT_DIR/Preprocessing/P4_Normalized_Ears"
echo "=================================================="
