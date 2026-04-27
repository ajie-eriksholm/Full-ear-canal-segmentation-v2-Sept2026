#!/bin/bash

# Usage:
# ./run_retrain_tissue_air.sh <CT_DIR> <JSON_DIR> <SEG_DIR> <HEATMAP_DIR> <LOG_DIR> <MODEL_DIR>

# Central retraining directory
RETRAIN_DIR="/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain_Tissue_Air"


# Default input/output directories
CT_DIR=${1:-/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain_Tissue_Air/CTs_Raw}
JSON_DIR=${2:-/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/ear_anatomical_landmarks_train}
SEG_DIR=${3:-/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/segmentations/ear_masks_train}
HEATMAP_DIR=${4:-$RETRAIN_DIR/Heatmaps}
LOG_DIR=${5:-$RETRAIN_DIR/Logs}
MODEL_DIR=${6:-$RETRAIN_DIR/Models}

# Preprocessing output directories
P1_OUT=${RETRAIN_DIR}/CTs_P1
P2_OUT=${RETRAIN_DIR}/CTs_P2
P3_OUT=${RETRAIN_DIR}/CTs_P3
P4_OUT=${RETRAIN_DIR}/CTs_P4

set -e

echo "Starting training..."

echo "Running P1 preprocessing..."
python preprocessing/p1_preprocessing.py --raw_scans_dir "$CT_DIR" --processed_scans_dir "$P1_OUT" --output_dir "$LOG_DIR"

echo "Running P2 preprocessing..."
python preprocessing/p2_preprocessing.py --input_dir "$P1_OUT" --output_dir "$P2_OUT"

echo "Running P3 preprocessing..."
python preprocessing/p3_preprocessing.py --input_dir "$P2_OUT" --output_dir "$P3_OUT"

echo "Running P4 preprocessing..."
python preprocessing/p4_preprocessing.py --input_dir "$P3_OUT" --output_dir "$P4_OUT"

echo "Generating heatmaps..."
python utils/heatmap_creation.py --nii_dir "$P4_OUT" --json_dir "$JSON_DIR" --heatmap_dir "$HEATMAP_DIR"

echo "Starting training..."
python model/train_tissue_air.py \
  --ct_dir "$P4_OUT" \
  --seg_dir "$SEG_DIR" \
  --heatmap_dir "$HEATMAP_DIR" \
  --log_dir "$LOG_DIR" \
  --model_dir "$MODEL_DIR"
