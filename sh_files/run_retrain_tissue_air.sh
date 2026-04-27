#!/bin/bash

# Usage:
# ./run_retrain_tissue_air.sh <CT_DIR> <JSON_DIR> <SEG_DIR> <HEATMAP_DIR> <LOG_DIR> <MODEL_DIR>

# Central retraining directory
RETRAIN_DIR="/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain"

# Default input/output directories
CT_DIR=${1:-/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain/CTs}
JSON_DIR=${2:-/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/ear_anatomical_landmarks_train}
SEG_DIR=${3:-/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/segmentations/ear_masks_train}
HEATMAP_DIR=${4:-$RETRAIN_DIR/Heatmaps}
LOG_DIR=${5:-$RETRAIN_DIR/Logs}
MODEL_DIR=${6:-$RETRAIN_DIR/Models}

set -e

echo "Generating heatmaps..."
python utils/heatmap_creation.py --nii_dir "$CT_DIR" --json_dir "$JSON_DIR" --heatmap_dir "$HEATMAP_DIR"

echo "Starting training..."
python model/train_tissue_air.py \
  --ct_dir "$CT_DIR" \
  --seg_dir "$SEG_DIR" \
  --heatmap_dir "$HEATMAP_DIR" \
  --log_dir "$LOG_DIR" \
  --model_dir "$MODEL_DIR"
