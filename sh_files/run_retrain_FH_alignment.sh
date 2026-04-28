#!/bin/bash

# Usage:
# ./run_retrain_FH_alignment.sh <CT_DIR> <JSON_DIR> <HEATMAP_DIR> <LOG_DIR>
# bash sh_files/run_retrain_FH_alignment.sh 


# Directory paths
RETRAIN_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/Retrain_FH_Alignment"
RAW_SCANS_DIR="$RETRAIN_DIR/CTs_Raw"
PROCESSED_SCANS_DIR="$RETRAIN_DIR/CTs_Processed"
OUTPUT_DIR="$RETRAIN_DIR/Output"
HEATMAP_DIR="$RETRAIN_DIR/Heatmaps"
LOG_DIR="$RETRAIN_DIR/Logs"
JSON_DIR="/projects/oticon/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/fh_alignment_landmarks_train"

# Python environments
SEG_ENV="./seg_env"
LANDMARK_ENV="./landmark_env"

# Preprocessing scripts location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PREPROCESSING_DIR="$PROJECT_ROOT/preprocessing"

# Ensure output directories exist
mkdir -p "$HEATMAP_DIR" "$LOG_DIR"

set -e

# ============================================================================== 
# P1: Initial Preprocessing
# ==============================================================================
echo "=================================================="
echo "STEP 1/4: Running P1 - Initial Preprocessing"
echo "=================================================="
echo "Activating seg_env for P1..."
source "$SEG_ENV/bin/activate"

python "$PREPROCESSING_DIR/p1_preprocessing.py" \
  --raw_scans_dir "$RAW_SCANS_DIR" \
  --processed_scans_dir "$PROCESSED_SCANS_DIR" \
  --output_dir "$OUTPUT_DIR"

deactivate
echo "[OK] P1 preprocessing completed successfully"
echo ""

# ==============================================================================
# Heatmap Generation
# ==============================================================================
echo "Generating heatmaps..."
source "$LANDMARK_ENV/bin/activate"
python "$PROJECT_ROOT/utils/heatmap_creation.py" --nii_dir "$PROCESSED_SCANS_DIR" --json_dir "$JSON_DIR" --heatmap_dir "$HEATMAP_DIR" --fh_alignment
echo "[OK] Heatmap generation completed"

# ============================================================================== 
# Training
# ==============================================================================
echo "Starting FH alignment training..."
python "$PROJECT_ROOT/model/train_FH_alignment.py" \
  --ct_dir "$PROCESSED_SCANS_DIR" \
  --json_dir "$JSON_DIR" \
  --heatmap_dir "$HEATMAP_DIR" \
  --log_dir "$LOG_DIR" \
  --fh_alignment


