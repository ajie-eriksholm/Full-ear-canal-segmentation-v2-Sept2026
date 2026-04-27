#!/bin/bash

# Usage:
# ./run_retrain_FH_alignment.sh <CT_DIR> <JSON_DIR> <HEATMAP_DIR> <LOG_DIR> <MODEL_DIR>

# Central retraining directory
RETRAIN_DIR="/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain_FH_Alignment"

# Default input/output directories
CT_DIR=${1:-/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain_FH_Alignment/CTs_Raw}
JSON_DIR=${2:-/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/fh_alignment_landmarks_train}
HEATMAP_DIR=${3:-$RETRAIN_DIR/Heatmaps}
LOG_DIR=${4:-$RETRAIN_DIR/Logs}
MODEL_DIR=${5:-$RETRAIN_DIR/Models}

set -e

echo "Running P1 preprocessing..."
python preprocessing/p1_preprocessing.py --raw_scans_dir "$CT_DIR" --processed_scans_dir "$RETRAIN_DIR/CTs_P1" --output_dir "$LOG_DIR"

# Use P1 output as input for heatmap creation
P1_OUT="$RETRAIN_DIR/CTs_P1"

echo "Generating heatmaps..."
python utils/heatmap_creation.py --nii_dir "$P1_OUT" --json_dir "$JSON_DIR" --heatmap_dir "$HEATMAP_DIR"

echo "Starting FH alignment training..."
python model/train_FH_alignment.py \
  --ct_dir "$P1_OUT" \
  --json_dir "$JSON_DIR" \
  --heatmap_dir "$HEATMAP_DIR" \
  --log_dir "$LOG_DIR" \
  --model_dir "$MODEL_DIR"
