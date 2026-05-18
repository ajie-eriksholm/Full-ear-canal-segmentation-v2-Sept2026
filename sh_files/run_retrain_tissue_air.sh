
#!/bin/bash
#$ -N Retrain_Tissue_Air               # Job name
#$ -cwd                                # Run in current working directory
#$ -l nvgpu=1                          # Request 1 NVIDIA GPU
#$ -l gputype=rtx*                     # Request RTX series GPU
#$ -l cores=8                         # Request 16 CPU cores
#$ -l mem_free=64G                     # Request 64 GB of RAM
#$ -o retrain_tissue_air_3.log           # Standard output log
#$ -e retrain_tissue_air_3_err.log       # Standard error log
#$ -dl 203501010000                    # Hard deadline by which the job must finish (GPU branch here).

# ==============================================================================
# Retraining Pipeline: Tissue/Air model
# ==============================================================================
# Inputs in $RAW_SCANS_DIR are assumed to already be cropped, resampled to
# 128x128x128, and intensity-normalized to [0, 1] (i.e. they are the equivalent
# of P4_Normalized_Ears output). Files must be named:
#     <patient>_left_ear.nii.gz
#     <patient>_right_ear.nii.gz
# Therefore P1-P4 are skipped. The script:
#   1. Stages CTs into P4_Normalized_Ears/ with the nnUNet "_0000" suffix.
#   2. Generates landmark heatmaps from the JSON markups.
#   3. Trains the tissue/air model.
#
# Submit to cluster: qsub sh_files/run_retrain_tissue_air.sh
# Run locally:      bash sh_files/run_retrain_tissue_air.sh
# ==============================================================================

# Anchor to the repo root so paths work both locally and under qsub
# (under SGE, BASH_SOURCE points to the spool copy of this script).
PROJECT_ROOT="$HOME/Full-ear-canal-segmentation"
cd "$PROJECT_ROOT"

# Directory paths
RETRAIN_DIR="/projects/oticon/erhdata/Processed-Data/SBEO/Retrain_Tissue_Air"
RAW_SCANS_DIR="$RETRAIN_DIR/CTs_Raw"           # Already-cropped, 128^3, normalized ear CTs
OUTPUT_DIR="$RETRAIN_DIR/Output"
HEATMAP_DIR="$RETRAIN_DIR/Heatmaps"
LOG_DIR="$RETRAIN_DIR/Logs"
JSON_DIR="$RETRAIN_DIR/GT/Markups"             # *.mrk.json landmark files
SEG_DIR="$RETRAIN_DIR/GT/Masks"                # *.nrrd segmentation masks

# Staged inputs for the trainer (nnUNet naming convention with _0000 suffix)
STAGED_CT_DIR="$OUTPUT_DIR/Preprocessing/P4_Normalized_Ears"

# Python environment (P2-P4 envs no longer needed since preprocessing is skipped)
LANDMARK_ENV="./landmark_env"

# Number of epochs for training
EPOCHS=300

mkdir -p "$STAGED_CT_DIR" "$HEATMAP_DIR" "$LOG_DIR"

set -e

# ==============================================================================
# Stage inputs: symlink *_left_ear.nii.gz / *_right_ear.nii.gz into the trainer
# input directory with the "_0000" suffix expected by the nnUNet-style pipeline.
# Symlinks (rather than copies) keep disk usage minimal and preserve a single
# source of truth in CTs_Raw.
# ==============================================================================
echo "=================================================="
echo "STEP 1/3: Staging CTs into $STAGED_CT_DIR"
echo "=================================================="

if [ ! -d "$RAW_SCANS_DIR" ]; then
  echo "[ERROR] Raw scans dir does not exist: $RAW_SCANS_DIR" >&2
  exit 1
fi

shopt -s nullglob
staged=0
for src in "$RAW_SCANS_DIR"/*_ear.nii.gz; do
  base=$(basename "$src" .nii.gz)         # e.g. CHUM-013_right_ear
  dst="$STAGED_CT_DIR/${base}_0000.nii.gz" # e.g. CHUM-013_right_ear_0000.nii.gz
  if [ ! -e "$dst" ]; then
    ln -sf "$src" "$dst"
  fi
  staged=$((staged + 1))
done
shopt -u nullglob

echo "Staged $staged ear scans."
if [ "$staged" -eq 0 ]; then
  echo "[ERROR] No *_ear.nii.gz files found in $RAW_SCANS_DIR" >&2
  exit 1
fi
echo ""

# ==============================================================================
# Heatmap Generation (uses landmark_env which has torch/nibabel/scipy)
# ==============================================================================
echo "=================================================="
echo "STEP 2/3: Generating heatmaps"
echo "=================================================="
source "$LANDMARK_ENV/bin/activate"

python "$PROJECT_ROOT/utils/heatmap_creation.py" \
  --nii_dir "$STAGED_CT_DIR" \
  --json_dir "$JSON_DIR" \
  --heatmap_dir "$HEATMAP_DIR" \
  --nnunet

echo ""

# ==============================================================================
# Training
# ==============================================================================
echo "=================================================="
echo "STEP 3/3: Training tissue/air model ($EPOCHS epochs)"
echo "=================================================="
python "$PROJECT_ROOT/model/train_tissue_air.py" \
  --ct_dir "$STAGED_CT_DIR" \
  --seg_dir "$SEG_DIR" \
  --heatmap_dir "$HEATMAP_DIR" \
  --log_dir "$LOG_DIR" \
  --epochs "$EPOCHS"

