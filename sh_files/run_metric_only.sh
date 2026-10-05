#!/bin/bash
#$ -N MetricExtractionOnly              # Job name
#$ -cwd                                 # Run in current working directory
#$ -l cores=4                           # Metric extraction is CPU-only (vtk/vmtk), no GPU needed
#$ -l mem_free=32G                      # Request 32 GB of RAM
#$ -o metric_only.$JOB_ID.log           # Standard output log (unique per submission)
#$ -e metric_only.$JOB_ID.err.log       # Standard error log (unique per submission)

# ==============================================================================
# Metric Extraction ONLY (re-run/resume STEP 8/8 without redoing P1-P7)
# ==============================================================================
# Useful to fill in ears that were skipped after a previous run crashed
# partway through metric extraction (process_single_sample() already skips
# any ear whose markup JSON already exists, so already-processed ears are
# not redone).
#
# Usage:
#   qsub sh_files/run_metric_only.sh HNSC
#   qsub sh_files/run_metric_only.sh HECKTOR
#   qsub sh_files/run_metric_only.sh CTMR
# ==============================================================================

DATASET="$1"
if [ -z "$DATASET" ]; then
    echo "ERROR: Usage: qsub sh_files/run_metric_only.sh <HNSC|HECKTOR|CTMR>"
    exit 1
fi

OUTPUT_DIR="/projects/oticon/erhdata/Processed-Data/AJIE/Pre-processed_CT_annotations/$DATASET/Output"
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: OUTPUT_DIR does not exist: $OUTPUT_DIR"
    exit 1
fi

METRIC_ENV="/home/ajie/.conda/envs/metric_env"   # Activate by path: avoids ambiguity with other users' same-named envs

PROJECT_ROOT="$HOME/code/Full-ear-canal-segmentation"
cd "$PROJECT_ROOT"

METRIC_EXTRACTION_SCRIPT="$PROJECT_ROOT/metric_extraction/run_pipeline.py"
RESULTS_OUTPUT_DIR="$OUTPUT_DIR/Results"
METRICS_OUTPUT_DIR="$OUTPUT_DIR/Metrics"

echo "=================================================="
echo "Metric Extraction ONLY - dataset: $DATASET"
echo "=================================================="
echo "Switching to metric_env for metric extraction..."
eval "$(conda shell.bash hook)"
conda activate "$METRIC_ENV"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment at $METRIC_ENV"
    exit 1
fi

echo "Python environment: $(which python)"
python --version
echo "Input directory: $RESULTS_OUTPUT_DIR"
echo "Output directory: $METRICS_OUTPUT_DIR"
echo "=================================================="

python "$METRIC_EXTRACTION_SCRIPT" \
    --input "$RESULTS_OUTPUT_DIR" \
    --output "$METRICS_OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Metric extraction failed!"
    exit 1
fi

conda deactivate

echo ""
echo "[OK] Metric extraction completed successfully"
