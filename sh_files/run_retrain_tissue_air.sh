
#!/bin/bash
# ============================================================================== 
# Retraining Pipeline: Full Preprocessing + Training
# ============================================================================== 
# This script runs all preprocessing steps (P1-P4) in sequence, then starts training.
# Edit the variables below to customize paths and settings
# ==============================================================================

# Directory paths
RETRAIN_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/Retrain_Tissue_Air"
RAW_SCANS_DIR="$RETRAIN_DIR/CTs_Raw"
PROCESSED_SCANS_DIR="$RETRAIN_DIR/CTs_Processed"
OUTPUT_DIR="$RETRAIN_DIR/Output"
HEATMAP_DIR="$RETRAIN_DIR/Heatmaps"
LOG_DIR="$RETRAIN_DIR/Logs"
JSON_DIR="/projects/oticon/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/ear_anatomical_landmarks_train"
SEG_DIR="/projects/oticon/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/segmentations/ear_masks_train"

# Processing options
NO_EYES="False"
SKIP_ALIGNMENT="False"
LANDMARK_MODEL="/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/best_model_2025-12-05_13-16-35.pth"
ACCEPTABLE_PATIENTID_CSV="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/acceptable_patientid.csv"
PRE_QUALITY_ASSESSED="False"
EXCLUDED_SCANS_CSV="/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Excluded_scans_cropping.csv"

# Python environments
SEG_ENV="/home/sbeo/Full-ear-canal-segmentation/seg_env"
LANDMARK_ENV="/home/sbeo/Full-ear-canal-segmentation/landmark_env"


# Preprocessing scripts location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PREPROCESSING_DIR="$PROJECT_ROOT/preprocessing"

# Number of epochs for training
EPOCHS=10

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
  --output_dir "$OUTPUT_DIR" \
  --acceptable_patientid_csv "$ACCEPTABLE_PATIENTID_CSV" \
  --pre_quality_assessed "$PRE_QUALITY_ASSESSED"

deactivate
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

python "$PREPROCESSING_DIR/p2_preprocessing.py" \
  --processed_scans_dir "$PROCESSED_SCANS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --landmark_model "$LANDMARK_MODEL" \
  --no_eyes "$NO_EYES" \
  --skip_alignment "$SKIP_ALIGNMENT"

echo "[OK] P2 preprocessing completed successfully"
echo ""

# ============================================================================== 
# P3: ROI Cropping Around Ear Landmarks
# ==============================================================================
echo "=================================================="
echo "STEP 3/4: Running P3 - ROI Cropping"
echo "=================================================="

python "$PREPROCESSING_DIR/p3_preprocessing.py" \
  --processed_scans_dir "$PROCESSED_SCANS_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --excluded_scans_csv "$EXCLUDED_SCANS_CSV" \
  --no_eyes "$NO_EYES" \
  --skip_alignment "$SKIP_ALIGNMENT"

echo "[OK] P3 preprocessing completed successfully"
echo ""

# ============================================================================== 
# P4: Upsampling and Normalization for Inference
# ==============================================================================
echo "=================================================="
echo "STEP 4/4: Running P4 - Upsampling & Normalization"
echo "=================================================="

python "$PREPROCESSING_DIR/p4_preprocessing.py" \
  --processed_scans_dir "$PROCESSED_SCANS_DIR" \
  --output_dir "$OUTPUT_DIR"

echo "[OK] P4 preprocessing completed successfully"
echo ""

# ============================================================================== 
# Heatmap Generation
# ==============================================================================
echo "Generating heatmaps..."
python "$PROJECT_ROOT/utils/heatmap_creation.py" --nii_dir "$OUTPUT_DIR/Preprocessing/P4_Normalized_Ears" --json_dir "$JSON_DIR" --heatmap_dir "$HEATMAP_DIR"

# ============================================================================== 
# Training
# ==============================================================================
echo "Starting training for $EPOCHS epochs..."
python "$PROJECT_ROOT/model/train_tissue_air.py" \
  --ct_dir "$OUTPUT_DIR/Preprocessing/P4_Normalized_Ears" \
  --seg_dir "$SEG_DIR" \
  --heatmap_dir "$HEATMAP_DIR" \
  --log_dir "$LOG_DIR" \
  --epochs "$EPOCHS"

