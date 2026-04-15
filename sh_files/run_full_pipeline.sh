#!/bin/bash
# ==============================================================================
# Full Pipeline: Preprocessing -> Inference -> Postprocessing
# ==============================================================================
# This script runs the entire ear canal segmentation pipeline end-to-end.
# All paths are configured here — no need to edit any Python files.
#
# Usage:
#   ./sh_files/run_full_pipeline.sh
#
# Pipeline steps:
#   P1 (seg_env)      - Intensity clipping, segmentation, cropping, resampling
#   P2 (landmark_env) - Landmark detection and Frankfort plane alignment
#   P3 (landmark_env) - ROI cropping around ear landmarks
#   P4 (landmark_env) - Upsampling and normalization for inference
#   Inference Tissue   - 3D U-Net tissue vs air segmentation and landmark prediction
#   Inference Bone     - nnU-Net bone segmentation
#   Postprocessing     - Markup JSONs, NIfTI masks, and STL generation
# ==============================================================================

# ==============================================================================
# CONFIGURATION - Edit these variables as needed
# ==============================================================================

# --- Directory paths ---
RAW_SCANS_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Raw"
PROCESSED_SCANS_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Processed-Data"
OUTPUT_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Output"

# --- Processing options ---
NO_EYES="False"              # Set to "True" if scans don't include eyes
SKIP_ALIGNMENT="False"       # Set to "True" to skip alignment step

# --- Python environments ---
SEG_ENV="/home/sbeo/Full-ear-canal-segmentation/seg_env"           # For P1
LANDMARK_ENV="/home/sbeo/Full-ear-canal-segmentation/landmark_env" # For P2, P3, P4, inference, postprocessing

# --- Landmark detection model (for P2) ---
LANDMARK_MODEL="/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/best_model_2025-12-05_13-16-35.pth"

# --- Quality assessment filter (for P1) ---
ACCEPTABLE_PATIENTID_CSV="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/acceptable_patientid.csv"
PRE_QUALITY_ASSESSED="False"  # Set to "True" to only process scans in acceptable_patientid_csv

# --- Excluded scans (for P3) ---
EXCLUDED_SCANS_CSV="/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Excluded_scans_cropping.csv"

# --- Inference configuration ---
# Directory containing trained model logs (each run has a best_model.pth inside)
INFERENCE_LOGS_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Logs"

# Which model run to use for postprocessing (the predictions from this run
# will be passed to the postprocessing step)
POSTPROCESSING_RUN="run_20260210_105409"

# --- nnU-Net configuration (for bone segmentation) ---
NNUNET_RAW="/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_raw"
NNUNET_PREPROCESSED="/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_preprocessed"
NNUNET_RESULTS="/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results"
NNUNET_DATASET_ID=1

# ==============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Script locations
PREPROCESSING_DIR="$PROJECT_ROOT/preprocessing"
INFERENCE_SCRIPT="$PROJECT_ROOT/model/test_tissue_air.py"
BONE_INFERENCE_SCRIPT="$PROJECT_ROOT/model/test_bone.py"
POSTPROCESSING_SCRIPT="$PROJECT_ROOT/postprocessing/generate_results.py"

# Derived paths
INFERENCE_SCANS_DIR="$OUTPUT_DIR/Preprocessing/P4_Normalized_Ears"
BONE_INFERENCE_OUTPUT_DIR="$OUTPUT_DIR/Inference/nnUNet"
FH_PLANE_LM="$OUTPUT_DIR/Preprocessing/P2_Landmarks/landmark_positions_after_cropping.csv"
INFERENCE_OUTPUT_DIR="$OUTPUT_DIR/Inference"
PREDICTIONS_DIR="$INFERENCE_OUTPUT_DIR/$POSTPROCESSING_RUN/test_predictions"
PREDICTED_LANDMARKS_CSV="$PREDICTIONS_DIR/predicted_landmark_coordinates.csv"
RESULTS_OUTPUT_DIR="$OUTPUT_DIR/Results"

# Inference path templates (using {} as placeholder for run name)
# Models are loaded from INFERENCE_LOGS_DIR, but predictions are saved under OUTPUT_DIR
MODEL_PATH_TEMPLATE="$INFERENCE_LOGS_DIR/{}/best_model.pth"
OUTPUT_PREDICTIONS_DIR_TEMPLATE="$INFERENCE_OUTPUT_DIR/{}/test_predictions"

# ==============================================================================
# P1: Initial Preprocessing (Clipping, Segmentation, Cropping, Resampling)
# ==============================================================================
echo "=================================================="
echo "STEP 1/7: Running P1 - Initial Preprocessing"
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
echo "STEP 2/7: Running P2 - Landmark Detection & Alignment"
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
echo "STEP 3/7: Running P3 - ROI Cropping"
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
echo "STEP 4/7: Running P4 - Upsampling & Normalization"
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
# STEP 5: Tissue vs Air Inference
# ==============================================================================
echo "=================================================="
echo "STEP 5/7: Running Tissue vs Air Inference"
echo "=================================================="
echo "Test scans directory: $INFERENCE_SCANS_DIR"
echo "Model logs directory: $INFERENCE_LOGS_DIR"
echo "Model path template: $MODEL_PATH_TEMPLATE"
echo "Output predictions template: $OUTPUT_PREDICTIONS_DIR_TEMPLATE"
echo "=================================================="

python "$INFERENCE_SCRIPT" \
    --test_dir "$INFERENCE_SCANS_DIR" \
    --model_path_template "$MODEL_PATH_TEMPLATE" \
    --output_predictions_dir_template "$OUTPUT_PREDICTIONS_DIR_TEMPLATE" \
    --run_names "$POSTPROCESSING_RUN"

if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed!"
    exit 1
fi

echo ""
echo "[OK] Tissue vs air inference completed successfully"
echo ""

# ==============================================================================
# STEP 6: Bone Segmentation Inference (nnU-Net)
# ==============================================================================
echo "=================================================="
echo "STEP 6/7: Running Bone Segmentation Inference (nnU-Net)"
echo "=================================================="
echo "Input scans directory: $INFERENCE_SCANS_DIR"
echo "Output directory: $BONE_INFERENCE_OUTPUT_DIR"
echo "=================================================="

python "$BONE_INFERENCE_SCRIPT" \
    --input_dir "$INFERENCE_SCANS_DIR" \
    --output_dir "$BONE_INFERENCE_OUTPUT_DIR" \
    --dataset_id "$NNUNET_DATASET_ID" \
    --nnunet_raw "$NNUNET_RAW" \
    --nnunet_preprocessed "$NNUNET_PREPROCESSED" \
    --nnunet_results "$NNUNET_RESULTS"

if [ $? -ne 0 ]; then
    echo "ERROR: Bone segmentation inference failed!"
    exit 1
fi

echo ""
echo "[OK] Bone segmentation inference completed successfully"
echo ""

# ==============================================================================
# STEP 7: Postprocessing
# ==============================================================================
echo "=================================================="
echo "STEP 7/7: Running Postprocessing"
echo "=================================================="
echo "FH plane landmarks: $FH_PLANE_LM"
echo "Predicted landmarks: $PREDICTED_LANDMARKS_CSV"
echo "Masks directory: $PREDICTIONS_DIR"
echo "Bone masks directory: $BONE_INFERENCE_OUTPUT_DIR"
echo "Output directory: $RESULTS_OUTPUT_DIR"
echo "=================================================="

python "$POSTPROCESSING_SCRIPT" \
    --fh_plane_lm "$FH_PLANE_LM" \
    --predicted_landmarks "$PREDICTED_LANDMARKS_CSV" \
    --masks_dir "$PREDICTIONS_DIR" \
    --output_dir_markups "$RESULTS_OUTPUT_DIR/markups" \
    --output_dir_markups_no_fh "$RESULTS_OUTPUT_DIR/markups" \
    --output_dir_masks "$RESULTS_OUTPUT_DIR/masks" \
    --output_dir_stl "$RESULTS_OUTPUT_DIR/stl" \
    --bone_masks_dir "$BONE_INFERENCE_OUTPUT_DIR" \
    --output_dir_masks_bone "$RESULTS_OUTPUT_DIR/masks_bone" \
    --output_dir_stl_bone "$RESULTS_OUTPUT_DIR/stl_bone"

if [ $? -ne 0 ]; then
    echo "ERROR: Postprocessing failed!"
    exit 1
fi

echo ""
echo "[OK] Postprocessing completed successfully"
echo ""

# ==============================================================================
# COMPLETION
# ==============================================================================
echo "=================================================="
echo "ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Raw scans:               $RAW_SCANS_DIR"
echo "  Processed scans:         $PROCESSED_SCANS_DIR"
echo "  Output directory:        $OUTPUT_DIR"
echo ""
echo "Outputs:"
echo "  Processed data:          $PROCESSED_SCANS_DIR"
echo "  Logs:                    $OUTPUT_DIR/Logs"
echo "  P2 Landmarks:            $OUTPUT_DIR/Preprocessing/P2_Landmarks"
echo "  P3 Cropped ears:         $OUTPUT_DIR/Preprocessing/P3_Cropped_Ears"
echo "  P4 Normalized ears:      $OUTPUT_DIR/Preprocessing/P4_Normalized_Ears"
echo "  Tissue/air predictions:  $PREDICTIONS_DIR"
echo "  Bone predictions:        $BONE_INFERENCE_OUTPUT_DIR"
echo "  Markup JSONs:            $RESULTS_OUTPUT_DIR/markups"
echo "  NIfTI masks:             $RESULTS_OUTPUT_DIR/masks"
echo "  STL meshes:              $RESULTS_OUTPUT_DIR/stl"
echo "  Bone NIfTI masks:        $RESULTS_OUTPUT_DIR/masks_bone"
echo "  Bone STL meshes:         $RESULTS_OUTPUT_DIR/stl_bone"
echo "=================================================="
