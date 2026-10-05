
# Full Ear Canal Segmentation

## Table of Contents

- [Introduction](#introduction)
  - [Input Requirements](#input-requirements)
- [Environment Setup](#environment-setup)
  - [Automatic Setup (Recommended)](#automatic-setup-recommended)
  - [Manual Setup](#manual-setup)
- [Running the Full Pipeline](#running-the-full-pipeline)
  - [Configuration Options](#configuration-options)
- [Output Folder Organization](#output-folder-organization)
- [Pipeline Details](#pipeline-details)
  - [Preprocessing (P1 → P2 → P3 → P4)](#preprocessing-p1--p2--p3--p4)
    - [P1: Initial Preprocessing](#p1-initial-preprocessing-p1_preprocessingpy)
    - [P2: Landmark Detection & Alignment](#p2-landmark-detection--alignment-p2_preprocessingpy)
    - [P3: Ear ROI Cropping](#p3-ear-roi-cropping-p3_preprocessingpy)
    - [P4: Upsampling & Normalization](#p4-upsampling--normalization-p4_preprocessingpy)
  - [Eardrum Preprocessing](#eardrum-preprocessing-auxiliary-track)
  - [Inference](#inference-modeltest_tissue_airpy)
  - [Bone Segmentation Inference](#bone-segmentation-inference-modeltest_bonepy)
  - [Postprocessing](#postprocessing-postprocessinggenerate_resultspy)
  - [Map to Original Space](#map-to-original-space-postprocessingmap_results_to_originalpy)
  - [Metric Extraction](#metric-extraction-metric_extractionpipelinepy)
- [Bend Artifact Removal (Optional)](#bend-artifact-removal-optional-postprocessingbend_artifact_removalpy)
- [Retraining Scripts](#retraining-scripts)
  - [Retraining FH Alignment Model](#retraining-fh-alignment-model)
  - [Retraining Tissue vs Air Segmentation and Landmark Placement](#retraining-tissue-vs-air-segmentation-and-landmark-placement)
- [Webpage Feature Preparation](#webpage-feature-preparation-webpage_features_setupall_features_extractionpy)
- [References](#references)
- [Changes from Previous Version](#changes-from-previous-version)

---

## Introduction

This repository contains a complete end-to-end pipeline for **ear canal segmentation** and **anatomical landmark detection** from CT scans. Starting from raw NIfTI CT scans of the head, it produces 3D surface meshes (STL) of the ear canal along with anatomical landmark coordinates.

The pipeline consists of seven main stages:

1. **Preprocessing (P1-P4)** — Transforms raw CT scans into ear-cropped, orientation-standardized, normalized sub-volumes ready for inference.
2. **Eardrum Preprocessing** — Auxiliary track: crops high-resolution volumes around the eardrum region for detailed eardrum analysis.
3. **Tissue vs Air Inference** — Applies trained 3D U-Net models with dual task-specific heads to simultaneously predict ear canal segmentations (tissue vs air) and anatomical landmarks.
4. **Bone Segmentation Inference** — Applies a trained nnU-Net model to predict bone structures (skull, mandible) and additional landmarks.
5. **Postprocessing** — Combines predictions into visualization-ready outputs: JSON markup files (compatible with 3D Slicer), NIfTI segmentation masks, and STL surface meshes.
6. **Map to Original Space** — Maps the results (masks, STL, markups) from the ear-local processing space back onto the original scan grid using the P1/P2/P3 transform logs.
7. **Metric Extraction** — Computes ear canal centerlines (via vmtk) and anatomical measurements (lengths, diameters, cross-sections, angles) from the segmentation results.

### Input Requirements

Raw CT scans must be in **NIfTI format** (`.nii.gz`) and placed in a single input folder. Each file must follow the naming convention:

```
{PatientID}__CT.nii.gz
```

For example: `Patient001__CT.nii.gz`, `S12345__CT.nii.gz`. The `PatientID` (everything before `__CT.nii.gz`) is used throughout the pipeline to name all intermediate and output files.

---

## Environment Setup

Before running any part of the pipeline, you must create the required Python environments. The requirement files are located in the `env_req/` folder.

| Environment | Type | Used by | Key packages |
|---|---|---|---|
| `seg_env` | venv | P1 | TotalSegmentator, SimpleITK |
| `landmark_env` | venv | P2, P3, P4, Inference, Postprocessing | PyTorch, MONAI, PyVista |
| `metric_env` | conda | Metric Extraction | vmtk, SimpleITK, pyvista |

### Automatic Setup (Recommended)

**Option 1: Bash script (Linux/Mac)**
```bash
./sh_files/setup_environments.sh
```

**Option 2: Python script (Cross-platform)**
```bash
python utils/setup_environments.py
```

> **Important:** Run the setup script with your **system Python** (not from within a virtual environment). If you get "No such file or directory" errors, deactivate any active environment first with `deactivate`. **conda** must be installed for the metric extraction environment (vmtk is only available via conda-forge).

Both scripts will:
- Create both virtual environments (`seg_env` and `landmark_env`) at the project root
- Create the conda environment `metric_env` with vmtk and pip dependencies
- Install all required packages from `env_req/seg_env_req.txt`, `env_req/landmark_env_req.txt`, and `env_req/metric_env_req.txt`
- Fix ITK version symlinks for vmtk compatibility
- Handle any existing environments (asks before overwriting)
- Automatically resolve common VTK/pyvista conflicts if they occur

#### Troubleshooting: Permission Denied

If you get `Permission denied` when running the setup script:

```bash
chmod +x sh_files/setup_environments.sh
./sh_files/setup_environments.sh
```

Or run it directly with bash:
```bash
bash sh_files/setup_environments.sh
```

> **Note:** If you downloaded the repository as a ZIP file instead of cloning it with git, executable permissions are not preserved. Always use `git clone` to get the repository.

### Manual Setup

```bash
# Create seg_env for P1
python3 -m venv seg_env
source seg_env/bin/activate
pip install -r env_req/seg_env_req.txt
deactivate

# Create landmark_env for P2-P4, inference, postprocessing
python3 -m venv landmark_env
source landmark_env/bin/activate
pip install -r env_req/landmark_env_req.txt
deactivate

# Create metric_env for metric extraction (requires conda)
# NOTE: use python=3.10 - the conda-forge vmtk 1.5.0 build for python 3.11
# (py311hdced90c_14) ships broken Python bindings (vtkvmtk classes don't
# inherit vtkAlgorithm, so SetInputData/GetOutput are missing at runtime).
conda create -n metric_env -c conda-forge --override-channels python=3.10 vmtk -y
conda activate metric_env
pip install -r env_req/metric_env_req.txt
# Install pyvista separately, pinned + --no-deps, so pip doesn't pull in a
# newer vtk (pyvista>=0.45 requires vtk>=9.3.1, but vmtk needs the bundled
# vtk 9.2.6 - a pip-installed vtk breaks vmtk's compiled Python bindings).
pip install --no-deps "pyvista==0.44.2" pyvista-validation cyclopts rich-rst matplotlib pooch scooby "typing_extensions>=4.10"
# Fix ITK symlinks (vmtk built against ITK 5.3, conda provides 5.4)
cd "$CONDA_PREFIX/lib"
for f in *-5.4.so.1; do link="${f/-5.4.so.1/-5.3.so.1}"; [ ! -e "$link" ] && ln -s "$f" "$link"; done
for f in *-5.4.so; do link="${f/-5.4.so/-5.3.so}"; [ ! -e "$link" ] && ln -s "$f" "$link"; done
conda deactivate
```

---

## Running the Full Pipeline

The simplest way to run everything (preprocessing + inference + postprocessing) is with the unified pipeline script. All configuration is done in the script header — no need to edit any Python files.

```bash
./sh_files/run_full_pipeline.sh
```

> **Note:** If you get `Permission denied`, run `chmod +x sh_files/run_full_pipeline.sh` first, or use `bash sh_files/run_full_pipeline.sh`.

**Before running**, edit the configuration section at the top of `sh_files/run_full_pipeline.sh`:

```bash
# Directory paths
RAW_SCANS_DIR="/path/to/raw/scans"
PROCESSED_SCANS_DIR="/path/to/processed/scans"
OUTPUT_DIR="/path/to/output"

# Processing options
NO_EYES="False"              # Set to "True" if scans don't include eyes
SKIP_ALIGNMENT="False"       # Set to "True" to skip alignment step
MAP_TO_ORIGINAL="True"       # Set to "False" to skip mapping results back to original space
EXPORT_SLICER_MARKUPS="True" # Set to "True" to export landmarks as 3D Slicer markups

# Python environments
SEG_ENV="/path/to/seg_env"
LANDMARK_ENV="/path/to/landmark_env"
METRIC_ENV="/path/to/metric_env"    # Conda environment for metric extraction (vmtk)

# Models
LANDMARK_MODEL="/path/to/landmark_model.pth"
INFERENCE_LOGS_DIR="/path/to/model/logs"
POSTPROCESSING_RUN="run_20260210_105409"   # Which model run to use

# Optional filters
ACCEPTABLE_PATIENTID_CSV="/path/to/acceptable_patientid.csv"
PRE_QUALITY_ASSESSED="False"
EXCLUDED_SCANS_CSV="/path/to/excluded_scans.csv"
```

The script runs all 10 steps automatically (**P1 → P2 → P3 → P4 → Eardrum Preprocessing → Tissue/Air Inference → Bone Inference → Postprocessing → Map to Original Space → Metric Extraction**), activates the correct environment for each step, and stops with a clear error if any step fails.

### Configuration Options

| Option | Default | Description |
|---|---|---|
| `NO_EYES` | `"False"` | Set to `"True"` if scans don't include eye structures. Uses mandible top points instead of eye landmarks for Frankfort plane fitting. Must be consistent across P2 and P3. |
| `SKIP_ALIGNMENT` | `"False"` | Set to `"True"` to skip Frankfort plane alignment. Only performs landmark detection without rotation. Must be consistent across P2 and P3. |
| `MAP_TO_ORIGINAL` | `"True"` | Set to `"False"` to skip mapping results back to the original scan space. When `"True"`, step 9 resamples masks, STL meshes, and landmarks onto the original CT grid. |
| `EXPORT_SLICER_MARKUPS` | `"True"` | Set to `"True"` to also export landmarks as 3D Slicer-compatible markup files (`.mrk.json`). Useful for clinical visualization and validation. |
| `PRE_QUALITY_ASSESSED` | `"False"` | Set to `"True"` to only process scans listed in `ACCEPTABLE_PATIENTID_CSV`. |
| `POSTPROCESSING_RUN` | — | The model run name to use for inference and postprocessing (e.g., `"run_20260210_105409"`). |

---

## Output Folder Organization

After running the full pipeline, the output directory structure will be:

```
OUTPUT_DIR/
├── Preprocessing/
│   ├── P2_Landmarks/                          # All P2 landmark outputs
│   │   ├── heatmaps/                          # Per-patient heatmaps & predictions
│   │   │   └── {patient_id}/
│   │   │       ├── predicted_landmarks.txt
│   │   │       ├── predicted_landmarks.json
│   │   │       └── pred_heatmap_landmark*.nii.gz
│   │   ├── aligned_npy/                       # Aligned landmark coordinates (.npy)
│   │   │   └── {patient_id}_lm_aligned.npy
│   │   ├── all_landmark_predictions.csv       # All detected landmarks
│   │   ├── all_aligned_landmarks.csv          # Post-alignment positions
│   │   └── landmark_positions_after_cropping.csv
│   ├── P3_Cropped_Ears/                       # Cropped ear ROIs from P3
│   │   ├── {patient_id}_left_ear.nii.gz
│   │   └── {patient_id}_right_ear.nii.gz
│   └── P4_Normalized_Ears/                    # Final normalized scans (nnU-Net naming)
│       ├── {patient_id}_left_0000.nii.gz
│       └── {patient_id}_right_0000.nii.gz
│
├── Inference/                                 # Model predictions
│   ├── {run_name}/                            # Tissue vs air predictions
│   │   └── test_predictions/
│   │       ├── predicted_landmark_coordinates.csv
│   │       └── {patient_id}_{ear_side}_pred_mask.nii.gz
│   └── nnUNet/                                # Bone segmentation predictions
│       └── {patient_id}_{ear_side}.nii.gz
│
├── Results/                                   # Final user-facing outputs
│   ├── markups/                               # JSON landmark files (all landmarks combined)
│   │   └── {patient_id}_{ear_side}.json
│   ├── masks/                                 # Tissue vs air NIfTI segmentation masks
│   │   └── {patient_id}_{ear_side}.nii.gz
│   ├── stl/                                   # Tissue vs air STL surface meshes
│   │   └── {patient_id}_{ear_side}.stl
│   ├── masks_bone/                            # Bone NIfTI masks (skull+mandible)
│   │   └── {patient_id}_{ear_side}.nii.gz
│   └── stl_bone/                              # Bone STL surface meshes
│       └── {patient_id}_{ear_side}.stl
│
├── Results_original_space/                    # Results mapped back onto the original scan grid
│   ├── markups/                               # Landmark JSONs in original scan coordinates
│   ├── masks/                                 # Tissue masks resampled to the original CT grid
│   ├── stl/                                   # Tissue STL meshes in original scan coordinates
│   ├── masks_bone/                            # Bone masks resampled to the original CT grid
│   └── stl_bone/                              # Bone STL meshes in original scan coordinates
│
├── Metrics/                                   # Metric extraction outputs
│   ├── processing_results.csv                 # All measurements aggregated
│   ├── markups/                               # Updated landmark JSONs
│   │   └── {patient_id}_{ear_side}.json
│   ├── masks/                                 # Inverted masks used for processing
│   │   └── {patient_id}_{ear_side}_inverted.nii.gz
│   ├── stl/                                   # Derived surface meshes
│   │   ├── {id}_clean_one_blob.stl            # Cleaned canal surface
│   │   ├── {id}_eardrum.stl                   # Eardrum cap
│   │   ├── {id}_isthmus.stl                   # Isthmus cross-section
│   │   ├── {id}_isthmus_eardrum_segment.stl   # Canal segment: isthmus to eardrum
│   │   ├── {id}_hard_tissue.stl               # Hard tissue portion
│   │   ├── {id}_soft_tissue.stl               # Soft tissue portion
│   │   ├── {id}_open_surface_cut_eardrum.stl  # Open surface at eardrum
│   │   └── {id}_outside.stl                   # Outer canal surface
│   └── vtk/                                   # Centerline & geodesic data
│       ├── {id}_centerline.vtk                # Final centerline
│       ├── {id}_centerline_features.vtk       # Centerline with cross-section features
│       ├── {id}_centerline_landmarks.json     # Landmark positions on centerline
│       ├── {id}_centerline_raw.vtk            # Raw vmtk centerline
│       ├── {id}_centerline_refined.vtk        # Refined centerline
│       ├── {id}_centerline_refined_raw.vtk    # Raw refined centerline
│       ├── {id}_geodesic.vtk                  # Geodesic path
│       └── {id}_geodesic_raw.vtk              # Raw geodesic path
│
└── Logs/                                      # Diagnostic & debug output
    ├── transform_logs/                        # JSON transformation history (P1, P2, P3)
    ├── P1_head_cropping_visualizations/        # Comparison images from P1
    └── P2_FH_landmark_visualizations/          # FH-plane landmark overlay images (P2)
        └── flagged_large_rotations.csv

PROCESSED_SCANS_DIR/
└── {patient_id}/
    ├── {patient_id}_CT_intensity_clipped.nii.gz
    ├── {patient_id}_CT_cavities_segmented.nii.gz
    ├── {patient_id}_CT_cropped.nii.gz
    ├── {patient_id}_CT_resampled.nii.gz
    ├── {patient_id}_CT_shifted_to_origin.nii.gz
    ├── {patient_id}_CT_padded.nii.gz
    ├── {patient_id}_CT_resampled_256.nii.gz
    ├── {patient_id}_aligned.nii.gz (if alignment enabled)
    ├── {patient_id}_lm_aligned.npy (if alignment enabled)
    ├── {patient_id}_centroids.csv
    ├── {patient_id}_left_ear_cropped.nii.gz
    ├── {patient_id}_right_ear_cropped.nii.gz
    ├── {patient_id}_left_ear_cropped_mirrored.nii.gz
    ├── {patient_id}_left_ear_cropped_mirrored_origin_reset.nii.gz
    └── {patient_id}_right_ear_cropped_origin_reset.nii.gz

Processed_eardrum/                                # Eardrum-centered high-resolution crops (auxiliary track)
└── {patient_id}/
    ├── {patient_id}_right_ear_raw_hu.nii.gz    # Right ear crop, raw Hounsfield units
    ├── {patient_id}_right_ear_0000.nii.gz      # Right ear crop, normalized to [0, 1]
    ├── {patient_id}_left_ear_raw_hu.nii.gz     # Left ear crop, raw HU (mirrored)
    ├── {patient_id}_left_ear_0000.nii.gz       # Left ear crop, normalized (mirrored)
    ├── {patient_id}_transform_log.json         # Full record of this script's processing steps
    └── visualizations/                         # Optional QA screenshots
```

The eardrum preprocessing step is an optional auxiliary track and does not affect the main pipeline results. If it fails, the pipeline continues. See [Eardrum Crop Extraction (Detailed)](#eardrum-crop-extraction-detailed) for exactly how these files are produced.

---

## Pipeline Details

### Preprocessing (P1 → P2 → P3 → P4)

The preprocessing code is in `preprocessing/` and consists of four sequential steps that transform raw CT scans into normalized, ear-cropped sub-volumes.

#### Running Preprocessing Separately

You can run just the preprocessing steps using the dedicated script:

```bash
./sh_files/run_full_preprocessing.sh
```

Or run individual steps directly:

```bash
# P1 (activate seg_env first)
source seg_env/bin/activate
python preprocessing/p1_preprocessing.py \
    --raw_scans_dir "/path/to/raw" \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output"

# P2-P4 (activate landmark_env first)
source landmark_env/bin/activate

python preprocessing/p2_preprocessing.py \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output" \
    --landmark_model "/path/to/model.pth" \
    --no_eyes "False" \
    --skip_alignment "False"

python preprocessing/p3_preprocessing.py \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output" \
    --no_eyes "False" \
    --skip_alignment "False"

python preprocessing/p4_preprocessing.py \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output"
```

Individual shell scripts are also available: `sh_files/run_p1.sh` and `sh_files/run_p2_p3.sh`.

---

#### P1: Initial Preprocessing (`p1_preprocessing.py`)

**Environment:** `seg_env`

Takes raw CT scans (`*__CT.nii.gz`) and produces resampled 256³ volumes centered on the ear region.

**Steps:**

| Step | Description |
|---|---|
| 1. Quality Filtering | Checks voxel spacing against thresholds (1.5mm, 1.5mm, 3.5mm). Skips scans exceeding these thresholds. Optional: process only pre-approved patient IDs. |
| 2. Intensity Clipping | Clips CT values to **[-1000, 2007 HU]**. |
| 3. Anatomical Segmentation | Uses **TotalSegmentator** (`head_glands_cavities` task) to identify left/right eyes (labels 1-2) and left/right auditory canals (labels 16-17). |
| 4. Centroid Calculation | Computes 3D centroids for segmented structures and the midpoint between ear canals. Saves to CSV. Skips patients if both ear centroids cannot be found. |
| 5. ROI Cropping | Crops a **200×270×250 mm** ROI centered on the ear midpoint. Pads with -1000 HU where ROI extends beyond scan boundaries. |
| 6. Isotropic Resampling | Resamples to **0.5×0.5×0.5 mm** spacing using B-spline interpolation. |
| 7. Origin Reset | Sets image origin to (0, 0, 0). |
| 8. Padding | Pads to **540×540×540 voxels** using -1000 HU. |
| 9. Final Resampling | Resamples to **256×256×256 voxels** using linear interpolation. |

**Output:** `{PatientID}_CT_resampled_256.nii.gz`

---

#### P2: Landmark Detection & Alignment (`p2_preprocessing.py`)

**Environment:** `landmark_env`

Detects 6 anatomical landmarks using a 3D U-Net and aligns scans to the Frankfort horizontal plane.

**Landmark Detection:**
- Loads a pre-trained **3D U-Net** (instance normalization, 16 base features, 4-level encoder, 6-channel output)
- Generates probability heatmaps for each landmark, applies sigmoid activation and 0.5 threshold
- Computes weighted centroids for sub-voxel precision
- Converts from prediction space `(x,y,z)` to anatomical space `(z,-y,-x)`
- **Landmarks detected:** IDs 8, 9, 10, 11, 12, 13

**Frankfort Plane Alignment** (two-step rotation):

1. **Make Frankfort Plane Horizontal** — Computes Frankfort plane normal using landmarks 10-13 (or 8-11 + mandible points in no-eyes mode) and rotates to align with horizontal.
2. **Correct Left-Right Orientation** — Projects left-right vector (landmark 12→13) onto XY plane and rotates around Z-axis to align with X-axis.

**No-Eyes Mode:** When `no_eyes=True`, uses TotalSegmentator to segment craniofacial structures and extracts mandible top points as replacements for eye landmarks.

**Quality Control:** Flags scans with Z-axis rotation > 10° and creates 3-view visualizations for manual review.

**Output:** `{PatientID}_aligned.nii.gz`, `{PatientID}_lm_aligned.npy`, aggregated CSVs

---

#### P3: Ear ROI Cropping (`p3_preprocessing.py`)

**Environment:** `landmark_env`

Extracts individual ear volumes from aligned scans and ensures both ears have the same anatomical orientation.

**Steps:**

| Step | Description |
|---|---|
| 11. Right Ear Cropping | Centers a **90×90×90 voxel** ROI at Landmark 10 with offset **[+20, +10, -5] mm**. Pads with -1000 HU. |
| 12. Left Ear Cropping | Centers a **90×90×90 voxel** ROI at Landmark 11 with offset **[-20, +10, -5] mm**. Pads with -1000 HU. |
| 13. Left Ear Mirroring | Flips left ear along axis 0 (x-axis) so both ears share the same anatomical orientation. |
| 14. Origin Reset | Sets origin to (0, 0, 0) for both ears. Saves to patient directory and `Preprocessing/P3_Cropped_Ears/`. |

**Output:** `{PatientID}_right_ear_cropped_origin_reset.nii.gz`, `{PatientID}_left_ear_cropped_mirrored_origin_reset.nii.gz`

---

#### P4: Upsampling & Normalization (`p4_preprocessing.py`)

**Environment:** `landmark_env`

Prepares ear volumes for model inference by resampling and normalizing.

**Steps:**

| Step | Description |
|---|---|
| 15. Upsampling | Resamples to **128×128×128 voxels** using trilinear interpolation. |
| 16. Intensity Normalization | Clips to [-1000, 2007 HU] and normalizes to **[0, 1]** range. Output: float32. |

**Output:**
- `{PatientID}_left_0000.nii.gz`, `{PatientID}_right_0000.nii.gz` in `Output/Preprocessing/P4_Normalized_Ears/`
- Uses nnU-Net `_0000` naming convention so both tissue/air and bone inference read from the same folder

---

### Eardrum Preprocessing (Auxiliary Track)

**Environment:** `landmark_env`  
**Script:** `pre-processing-eardrum.py` (located in parent directory)

Optional auxiliary preprocessing step that crops high-resolution volumes centered around the eardrum region for detailed eardrum analysis. Runs independently alongside the main inference pipeline and does not affect the final segmentation/landmark results. It duplicates and adapts the logic of P1–P4 rather than chaining them, so it can output a much higher-resolution crop.

#### Why a separate script instead of reusing P1–P4

P1–P4 downsample every scan onto a coarse ~1.05 mm / 256³ grid very early, because the trained landmark/segmentation models expect that exact scale, then crop a 90³-voxel ear ROI from that coarse grid. That resolution is too low to resolve fine eardrum structures. `pre-processing-eardrum.py` keeps the coarse grid only as a disposable "detection copy" used to find the landmarks and the Frankfort-plane rotation, but crops the eardrum ROI from the **native-resolution scan resampled directly to a finer isotropic spacing (0.2 mm by default)** — with no intermediate downsampling step.

#### Processing steps

| Step | Description |
|---|---|
| 1. Head localization | Same as P1: clips intensity to [-1000, 2007 HU], runs TotalSegmentator to find eyes/auditory canals, and crops a 200×270×200 mm ROI (+50 mm at the top) centered on the ear midpoint — all at native resolution. |
| 2. Reference frames | Builds two volumes from the native-resolution crop: (a) a disposable low-res volume (0.5 mm spacing, padded to 540³, resampled to 256³) matching the grid the landmark model was trained on, and (b) — since the pipeline calls this script with `--fast False` — the **full high-resolution volume** resampled directly to 0.2 mm isotropic spacing. |
| 3. Landmark detection | Runs the same 3D U-Net landmark model used by P2 on the disposable low-res volume to locate landmarks 8–13. |
| 4. Frankfort-plane alignment | Computes the same two-step rotation as P2 (horizontal + left-right correction), including the no-eyes/mandible fallback and QA rotation-angle flagging. Because `--fast False`, the entire high-resolution volume is physically rotated to match (not just the landmark coordinates). |
| 5. Ear crop + mirror | Crops a **256³ voxel cube (51.2 mm at 0.2 mm spacing)** centered on the aligned right-ear landmark (10) / left-ear landmark (11), offset 20 mm along X (sign mirrored for the left ear) to better enclose the cochlea. The crop's output affine is computed so it geometrically overlaps the same local coordinate frame P3/P4 would have produced — it replicates P3's 90-voxel ROI box at ~1.05 mm and P4's origin reset internally, so no transform logs from the main pipeline are needed. The left ear is flipped along axis 0 to match P3's mirroring convention. |
| 6. Normalization | Clips to [-1000, 2007 HU] and scales to [0, 1] float32, identical to P4's final step. Both the raw-HU and normalized crops are saved. |
| 7. Transform log | Saves a per-patient JSON recording every step (centroids, landmark coordinates, rotation matrix, crop geometry) — the information needed to later map an eardrum segmentation back onto the original scan. |

#### How it differs from the main P1→P4 pipeline

| | Main pipeline (P1–P4) | Eardrum track |
|---|---|---|
| Final spacing | ~1.05 mm isotropic | 0.2 mm isotropic (configurable) |
| Final FOV | 90³ voxels (≈ 94.5 mm cube) | 256³ voxels (51.2 mm cube) |
| Downsampling before crop | Yes (onto the 1.05 mm grid) | No — cropped directly from native resolution |
| Landmark model | Same 3D U-Net (landmarks 8–13) | Same 3D U-Net, reused only for detection |
| Dependency on transform logs | N/A (is the source of the logs) | None — recomputes its own alignment independently |
| Failure handling in `run_full_pipeline.sh` | Stops the pipeline | Logged as a warning; pipeline continues (auxiliary step) |

**Output** (per patient, in `Processed_eardrum/{patient_id}/`):
- `{patient_id}_right_ear_raw_hu.nii.gz` / `{patient_id}_left_ear_raw_hu.nii.gz` — raw Hounsfield-unit crops
- `{patient_id}_right_ear_0000.nii.gz` / `{patient_id}_left_ear_0000.nii.gz` — normalized to [0, 1] (left mirrored)
- `{patient_id}_transform_log.json` — full record of every processing step
- `visualizations/` — optional QA screenshots
- If this step fails, the pipeline logs a warning and continues (non-critical auxiliary step)

---

### Inference (`model/test_tissue_air.py`)

**Environment:** `landmark_env`

Runs trained 3D U-Net models to generate ear canal segmentation masks and anatomical landmark predictions from the preprocessed 128³ volumes (already normalized to [0, 1] by P4).

#### Running Inference Separately

```bash
source landmark_env/bin/activate
python model/test_tissue_air.py \
    --test_dir "/path/to/Output/Preprocessing/P4_Normalized_Ears" \
    --model_path_template "/path/to/Logs/{}/best_model.pth" \
    --output_predictions_dir_template "/path/to/Output/Inference/{}/test_predictions" \
    --run_names "run_20260210_105409"
```

Use `--run_names` to specify which model(s) to evaluate. Multiple runs can be passed: `--run_names run_A run_B`.

#### Model Architecture

**3D U-Net** with two task-specific heads:
- **Segmentation head:** Binary ear canal mask prediction
- **Landmark head:** 7-channel heatmap for anatomical points
- Supports base features of 32, 48, or 64
- Two architecture variants: "late divergence" and "early divergence"

#### Inference Process

1. Loads preprocessed 128³ ear volumes (already normalized by P4)
2. Runs forward pass through the model
3. Applies sigmoid activation and thresholding for segmentation
4. Extracts landmark coordinates from heatmap centroids
5. Converts predictions from voxel space to world coordinates (mm)
6. Saves segmentation masks as NIfTI files
7. Aggregates all landmark predictions into CSV

#### Predicted Landmarks (IDs 1-7)

| ID | Landmark |
|---|---|
| 1 | Cartilaginous Canal Point |
| 2 | Bony Canal Point |
| 3 | Eardrum |
| 4 | Top Retroauricular Sulcus |
| 5 | Bottom Retroauricular Sulcus |
| 6 | 1st Bend |
| 7 | 2nd Bend |

**Output:** `*_pred_mask.nii.gz` segmentation masks, `predicted_landmark_coordinates.csv`

---

### Bone Segmentation Inference (`model/test_bone.py`)

**Environment:** `landmark_env`

Runs a trained nnU-Net model to predict bone structures (skull, mandible) and 4 additional landmarks from the preprocessed ear volumes. Uses the nnU-Net-formatted scans from P4 (`P4_Normalized_Ears_nnUNet/`).

#### Running Bone Inference Separately

```bash
source landmark_env/bin/activate
python model/test_bone.py \
    --input_dir "/path/to/Output/Preprocessing/P4_Normalized_Ears_nnUNet" \
    --output_dir "/path/to/Output/Inference/nnUNet"
```

Additional options:

| Option | Default | Description |
|---|---|---|
| `--device` | auto-detect | `cpu` or `cuda`. Auto-detects GPU compatibility (requires CUDA \u2265 7.0). |
| `--dataset_id` | `1` | nnU-Net dataset ID |
| `--folds` | `0 1 2 3 4` | Folds to use for ensemble prediction |
| `--checkpoint` | `checkpoint_best.pth` | Checkpoint file to use |
| `--trainer` | — | nnU-Net trainer class (e.g., `nnUNetTrainerNoMirroring`) |
| `--plans` | — | nnU-Net plans name (e.g., `nnUNetResEncUNetLPlans`) |

#### nnU-Net Labels

| ID | Structure |
|---|---|
| 0 | Background |
| 1 | Skull |
| 2 | Mandible |
| 3-6 | Landmarks |

#### nnU-Net Data Preparation (`preprocessing/pre_nnunet_heatmaps.py`)

For **training** the nnU-Net model (not needed for inference), use this script to prepare the dataset:

```bash
python preprocessing/pre_nnunet_heatmaps.py \
    --cts_dir /path/to/cropped_ears \
    --masks_dir /path/to/bone_masks \
    --landmarks_dir /path/to/landmarks \
    --out_dir /path/to/nnUNet_raw/DatasetXXX_Ear
```

This creates the `imagesTr/`, `labelsTr/`, and `dataset.json` structure required by nnU-Net, fusing bone segmentation masks with landmark spheres (classes 3-6).

**Output:** `{patient_id}_{side}.nii.gz` bone segmentation predictions in `Output/Inference/nnUNet/`

---

### Postprocessing (`postprocessing/generate_results.py`)

**Environment:** `landmark_env`

Combines inference results with preprocessing landmarks to create visualization-ready outputs.

#### Running Postprocessing Separately

```bash
source landmark_env/bin/activate
python postprocessing/generate_results.py \
    --fh_plane_lm "/path/to/Output/Preprocessing/P2_Landmarks/landmark_positions_after_cropping.csv" \
    --predicted_landmarks "/path/to/predictions/predicted_landmark_coordinates.csv" \
    --masks_dir "/path/to/predictions" \
    --output_dir_markups "/path/to/Output/Results/markups" \
    --output_dir_markups_no_fh "/path/to/Output/Results/markups" \
    --output_dir_masks "/path/to/Output/Results/masks" \
    --output_dir_stl "/path/to/Output/Results/stl"
```

#### What It Does

**Landmark Combination:**
- Loads canal landmarks (1-7) from inference and FH plane landmarks (8-13) from preprocessing
- Maps landmarks to correct output IDs based on ear side
- Creates JSON markup files compatible with 3D Slicer

**Mask Processing:**
- Loads predicted segmentation masks and saves cleaned NIfTI masks with proper spatial metadata

**STL Generation:**
- Extracts 3D surface from binary segmentation mask using **marching cubes**
- Generates binary STL files viewable in 3D Slicer, MeshLab, Blender, etc.

**Output Organization:**
- Markup JSONs (all landmarks combined) → `Results/markups/`
- Tissue vs air NIfTI masks → `Results/masks/`
- Tissue vs air STL meshes → `Results/stl/`
- Bone NIfTI masks (skull + mandible) → `Results/masks_bone/`
- Bone STL meshes → `Results/stl_bone/`

---

### Map to Original Space (`postprocessing/map_results_to_original.py`)

**Environment:** `landmark_env` (venv)

The pipeline outputs live in the ear-local processing space (P1 standardizes
orientation, P2 aligns to the Frankfort plane, and P3 crops/mirrors each ear and
resets the origin). This step maps the results back onto the **original scan
grid** so masks, STL meshes, and markups can be overlaid on the raw CT.

**How it works:** the whole result→original chain is a composition of
voxel-space affine operations, so per ear it collapses to a single 4×4 matrix
built from the `P1`, `P2`, and `P3` transform logs (`Logs/transform_logs/`) plus
the intermediate NIfTIs in `Processed-Data/{patient_id}/`. Masks are resampled
onto the original CT grid (nearest neighbour, so they share the original
affine/shape); STL vertices and markup positions are mapped point-by-point. It
is independent of the input affine convention (`x,y,z` or `-x,-y,z`).

#### Running It Separately

```bash
python postprocessing/map_results_to_original.py \
    --logs_dir      "/path/to/Output/Logs/transform_logs" \
    --processed_dir "/path/to/Processed-Data" \
    --masks_dir      "/path/to/Output/Results/masks" \
    --masks_bone_dir "/path/to/Output/Results/masks_bone" \
    --stl_dir        "/path/to/Output/Results/stl" \
    --stl_bone_dir   "/path/to/Output/Results/stl_bone" \
    --markups_dir    "/path/to/Output/Results/markups" \
    --output_dir     "/path/to/Output/Results_original_space"
```

Add `--validate` (instead of `--output_dir`) to sanity-check the transform: it
maps each canal-mask centroid to original coordinates and prints the distance to
the ear centroid recorded by P1. Use `--patients ID1 ID2 ...` to limit the run.

**Output** (`Results_original_space/`): `masks/`, `masks_bone/`, `stl/`,
`stl_bone/`, and `markups/`, all in original scan coordinates.

---

### Metric Extraction (`metric_extraction/pipeline.py`)

**Environment:** `metric_env` (conda)

Computes ear canal centerlines and anatomical measurements from the segmentation results using vmtk for centerline extraction.

#### Running Metric Extraction Separately

```bash
conda activate metric_env
python metric_extraction/run_pipeline.py \
    --input "/path/to/Output/Results" \
    --output "/path/to/Output/Metrics"
```

#### Processing Steps

| Step | Description |
|---|---|
| 1. Mask inversion | Loads binary segmentation mask and inverts it for surface extraction |
| 2. Mesh creation | Creates a 3D surface mesh from the inverted mask |
| 3. Landmark loading | Reads landmarks from JSON markups (canal landmarks 1-7, FH landmarks 8-13, CBJ landmarks) |
| 4. Geodesic path | Computes geodesic path between Top RS and Bottom RS landmarks |
| 5. Centerline extraction | Extracts canal centerline using vmtk with plane-based trimming |
| 6. Centerline validation | Validates and refines the centerline |
| 7. Cross-section features | Extracts cross-section measurements along the centerline (area, perimeter, radii, aspect ratio) |
| 8. Metric computation | Computes lengths, tortuosity indices, volumes, and landmark-specific measurements |
| 9. CBJ analysis | Fits plane to CBJ landmarks, splits canal into soft/hard tissue portions |
| 10. Output saving | Saves VTK centerlines, STL meshes, landmark JSONs, and aggregated CSV |

#### Computed Metrics

The `processing_results.csv` contains per-sample measurements including:

| Category | Metrics |
|---|---|
| **Centerline** | length, tortuosity index |
| **Geodesic path** | length, tortuosity index |
| **Isthmus-to-eardrum segment** | length, tortuosity, volume |
| **Cross-sections (mean ± std)** | min/max radius, area, perimeter, aspect ratio |
| **Landmark cross-sections** | area, perimeter, radii, aspect ratio at 1st bend, 2nd bend, eardrum, isthmus |
| **CBJ (cartilage-bone junction)** | length from start, length to end, proportion, plane distance |
| **Tissue portions** | soft tissue volume, hard tissue volume |

**Output:** `processing_results.csv`, per-sample VTK/STL/JSON files in `Metrics/` subdirectories

---

## Bend Artifact Removal (Optional) (`postprocessing/bend_artifact_removal.py`)

**Environment:** `landmark_env`

Standalone post-hoc utility that removes the flat "bend" artefact from the tissue/air STL meshes produced by the main pipeline. This artefact is the soft-tissue sheet that appears where the 90×90×90 voxel ear ROI (cropped in P3) cuts through the head — it shows up as a flat plane in the STL and can clutter 3D visualisations.

This script is **not** part of the main pipeline and is **not invoked** by `run_full_pipeline.sh`. Run it on its own, after `generate_results.py` has produced the STL files in `Output/Results/stl/`.

#### How It Works

1. Reads the P2 transform log (`Output/Logs/transform_logs/{patient_id}_transform_log_P2.json`) to recover the Frankfort-plane rotation matrix.
2. Computes the bend plane normal by applying the inverse P2 rotation to the original-CT +Z axis (and mirroring the X component for the left ear, to match P3's left-side mirroring).
3. Sweeps a plane with that fixed normal upward along world +Z and locks onto the offset with the most surface inliers (the flat sheet).
4. Clips the mesh, keeping the +normal side (the anatomy above the bend), then retains the largest connected component.
5. If no strong flat sheet is detected (low inlier fraction or weak peak), the mesh is saved unmodified.
6. Saves a cleaned STL per ear and a side-by-side before/after PNG visualisation.

#### Running It

The script is configured via constants at the top of the file (no CLI arguments). Edit these before running:

```python
# postprocessing/bend_artifact_removal.py
STL_DIR    = "/path/to/Output/Results/stl"
LOG_DIR    = "/path/to/Output/Logs/transform_logs"
OUTPUT_DIR = "/path/to/bend_removal_output"

NUM_SAMPLES = None   # None = process every patient; or set an int to sample N at random
RANDOM_SEED = 42
```

Then:

```bash
source landmark_env/bin/activate
python postprocessing/bend_artifact_removal.py
```

The script auto-discovers patients that have both `{patient_id}_left.stl` and `{patient_id}_right.stl` in `STL_DIR` and a matching `{patient_id}_transform_log_P2.json` in `LOG_DIR`.

#### Tunable Parameters

Most defaults work out of the box; adjust at the top of the file only if a particular dataset misbehaves:

| Constant | Default | Purpose |
|---|---|---|
| `FACE_INLIER_TOL` | `1.5` mm | Plane thickness used when counting inliers during the sweep. |
| `OFFSET_STEP_MM` | `0.25` mm | Plane sweep step along world +Z. |
| `CLIP_MARGIN_MM` | `1.5` mm | Extra offset added to the clip plane so the cut sits just inside the bend. |
| `MIN_BEND_INLIER_FRAC` | `0.05` | Skip clipping if the best plane contains fewer than 5% of the mesh points. |
| `MIN_BEND_PEAK_RATIO` | `4.0` | Skip clipping if the best plane is not at least 4× denser than the average sweep slice. |
| `KEEP_SIDE` | `"+"` | Side of the plane to keep along the bend normal (`"+"`, `"-"`, `"auto"`, `"auto_inv"`). |
| `MANUAL_BEND_OFFSET` | `None` | Force a fixed bend offset (mm along the normal); bypasses the sweep. |

#### Output

For each processed patient:

```
OUTPUT_DIR/
├── {patient_id}_left_no_bend.stl
├── {patient_id}_right_no_bend.stl
└── {patient_id}_bend_removal.png   # 4-row before/after comparison (iso + side views)
```

STLs where no bend was detected are still saved (suffixed `_no_bend.stl`) but contain the original geometry unchanged.

---

## Retraining Scripts

This repository provides two retraining workflows:

### Retraining FH Alignment Model

If you want to retrain the FH alignment model (for example, to add more data or improve performance), follow these steps:

#### Prepare Your Data

- **CT Scans:** Place your raw NIfTI CT volumes in a directory (default: `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain_FH_Alignment/CTs_Raw`).
- **Landmark JSONs:** Place your FH alignment landmark JSON files in `/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/fh_alignment_landmarks_train` (or specify a different directory).

#### Run the Retraining Script

Use the provided shell script to run P1 preprocessing, generate heatmaps, and start training:

```bash
bash sh_files/run_retrain_FH_Alignment.sh
```

This will:
- Run P1 preprocessing on your CTs (outputting to `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain_FH_Alignment/CTs_P1` by default)
- Generate heatmaps from the P1-processed CTs and your FH alignment landmark JSONs
- Train the FH alignment model using the P1-processed CTs and heatmaps
- Save all outputs (processed CTs, heatmaps, logs, models) under `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain_FH_Alignment` by default

**Customizing Input/Output Directories:**
You can override the default directories by passing arguments to the script:

```bash
bash run_retrain_FH_alignment.sh <CT_DIR> <JSON_DIR> <HEATMAP_DIR> <LOG_DIR> <MODEL_DIR>
```
For example:
```bash
bash run_retrain_FH_alignment.sh /path/to/CTs /path/to/fh_jsons /path/to/heatmaps /path/to/logs /path/to/models
```

**Note:** The CT directory you provide (`<CT_DIR>`) should contain your raw CTs. The script will automatically create the P1 output directory under the retraining folder.

#### Adding More Data

To improve training, simply add more CTs and FH alignment landmark JSONs to the respective directories before running the script. The pipeline will automatically use all available data in the specified folders.

---

### Retraining Tissue vs Air Segmentation and Landmark Placement

If you want to retrain the tissue/air segmentation and landmark placement model (for example, to add more data or improve performance), follow these steps:

#### Prepare Your Data

- **CT Scans:** Place your raw NIfTI CT volumes in a directory (default: `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain/CTs`).
- **Segmentation Masks:** Place your ear mask segmentations (NRRD format) in `/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/segmentations/ear_masks_train` (or specify a different directory).
- **Landmark JSONs:** Place your anatomical landmark JSON files in `/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/ear_anatomical_landmarks_train` (or specify a different directory).

#### Run the Retraining Script

Use the provided shell script to run the **full preprocessing pipeline (P1–P4)**, generate heatmaps, and start training:

```bash
bash sh_files/run_retrain_tissue_air.sh
```

This will:
- Run P1–P4 preprocessing on your CTs (outputting to `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain/CTs_P4` by default)
- Generate heatmaps from the P4-processed CTs and your landmark JSONs
- Train the tissue/air segmentation and landmark placement model using the P4-processed CTs, segmentation masks, and heatmaps
- Save all outputs (processed CTs, heatmaps, logs, models) under `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain` by default

**Customizing Input/Output Directories:**
You can override the default directories by passing arguments to the script:

```bash
bash run_retrain_tissue_air.sh <CT_DIR> <JSON_DIR> <SEG_DIR> <HEATMAP_DIR> <LOG_DIR>
```
For example:
```bash
bash run_retrain_tissue_air.sh /path/to/CTs /path/to/landmarks /path/to/masks /path/to/heatmaps /path/to/logs /path/to/models
```

**Note:** The CT directory you provide (`<CT_DIR>`) should contain your raw CTs. The script will automatically create intermediate directories for each preprocessing stage (P1–P4) under the retraining folder.

#### Adding More Data

To improve training, simply add more CTs, segmentation masks, and landmark JSONs to the respective directories before running the script. The pipeline will automatically use all available data in the specified folders.

---

## Webpage Feature Preparation (`webpage_features_setup/all_features_extraction.py`)

**Environment:** `metric_env` (conda)

This script bundles every per-ear artefact required by the visualisation webpage into a
single, **pseudonymised** package: a feature CSV, smoothed STL meshes (canal + outer
block), centerline VTKs, and landmark JSONs. It combines the centerline metrics
produced by `metric_extraction/` with demographics from a metadata CSV and computes
additional geometric features (bend angles, sagittal arcs, clipped sub-volumes,
radius ratios, etc.) directly from the meshes and markups.

### What It Produces

For an `output_dir`:

```
output_dir/
├── all_ear_canal_features.csv        # one row per ear, fully pseudonymised
├── stl_webpage/                       # canal STL meshes (renamed by pseudonym)
├── stl_webpage_block/                 # smoothed solid-block STL from inverted mask
├── vtk_webpage/                       # centerline_refined.vtk + centerline_features.vtk
├── markup_webpage/                    # landmark JSONs (renamed by pseudonym)
├── excluded_participants.txt          # ears dropped (missing CSV / output values)
└── volume_diagnostics.txt             # per-ear notes for any failed volume metrics
```

A central pseudonymisation map is maintained at
`pseudonym_dir/patient_id_mapping.csv` (default:
`/projects/oticon/erhdata/Output/SBEO/pseudonymization`). Pseudonyms are assigned
**per patient** (not per ear) as `patient_00001`, `patient_00002`, … and are
suffixed with `_left` / `_right` for the ear-level outputs (e.g.
`patient_00042_left`). The mapping is incrementally extended across runs so that
the same patient always receives the same pseudonym across datasets.

### Required Inputs

The script expects an input directory laid out exactly like the metric-extraction
output:

```
input_dir/
├── processing_results.csv             # centerline features (from metric_extraction)
├── vtk/                               # *_centerline_refined.vtk, *_centerline_features.vtk
├── markups/                           # *.json (landmarks + cross-section indices, incl. CBJ)
├── stl/                               # *_open_surface_cut_eardrum.stl
└── masks/                             # *_inverted.nii.gz (used to build the block STL)
```

A separate metadata CSV (or list of CSVs) supplies `age` and `sex`. The patient
ID column and age/sex columns are auto-detected, but you can override them via
the `META_PATIENT_ID_COL`, `META_AGE_COL`, and `META_SEX_COL` constants at the
top of the script. Age strings such as `048Y` (TCIA format) and sex strings such
as `M`/`F`/`Male`/`Female` are normalised automatically.

### Configuration

All paths are configured at the top of
[webpage_features_setup/all_features_extraction.py](webpage_features_setup/all_features_extraction.py):

```python
input_dir     = "/path/to/Metrics"                       # metric_extraction output
metadata_csv  = ["/path/to/metadata.csv"]                # one or more files
output_dir    = "/path/to/Features_csv/<dataset_name>"
pseudonym_dir = "/projects/oticon/erhdata/Output/SBEO/pseudonymization"
```

> Keep `pseudonym_dir` pointing at the **shared** location so pseudonyms remain
> consistent across all datasets and projects.

### Running

```bash
conda activate metric_env
python webpage_features_setup/all_features_extraction.py
```

The script will:

1. Load (or create) the central pseudonym mapping and extend it with any new patients.
2. Read `processing_results.csv` and discover all ears from the `markups/` folder.
3. Skip ears with missing values in the centerline CSV (logged in `excluded_participants.txt`).
4. For each remaining ear, compute the full feature set (lengths, distances,
   radius ratios, bend/sagittal/entrance angles, CBJ position and plane angle,
   and five clipped sub-volumes from the canal STL).
5. Drop any ear whose required output columns (everything except `age`/`sex`)
   are still missing — these are also recorded in `excluded_participants.txt`.
6. Copy the canal STL, generate a smoothed block STL from the inverted mask,
   and copy the two centerline VTKs and the markup JSON, all renamed with the
   ear-level pseudonym (`patient_XXXXX_left|right`).
7. Write `all_ear_canal_features.csv` with pseudonymised `patient_id` and `EarID`
   columns and the exact column order expected by the webpage.

### Notes

- Only `age` and `sex` are allowed to be empty in the output CSV. Any other
  missing field causes the ear to be excluded.
- Block STLs are scaled from millimetres to metres (ANSYS convention) and
  smoothed with a windowed-sinc + Laplacian pass (sigma ≈ 1.5).
- Volume features rely on the cross-section planes stored in the markup JSON
  (`isthmus`, `1st_bend`, `2nd_bend`, `eardrum`, and optionally `cbj`). Failures
  for individual volume metrics are reported in `volume_diagnostics.txt`.
- Re-running the script is safe: existing pseudonyms are preserved and only new
  patients are added to the mapping.

---

## References

- **TotalSegmentator:** Wasserthal et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.230024

---

## Changes from Previous Version

`sh_files/run_full_pipeline.sh` was updated with the following changes (not yet reflected elsewhere in this document):

- **Step count increased from 9 to 10** — a new auxiliary step, **Eardrum Preprocessing**, now runs after P4 and before Tissue/Air Inference. It crops high-resolution volumes around the eardrum (via `pre-processing-eardrum.py`) for separate eardrum analysis. Failures in this step are logged as warnings but do not stop the pipeline.
- **New `METRIC_ENV` variable** — the metric extraction conda environment is now activated by absolute path (`/home/ajie/.conda/envs/metric_env`) instead of by name, avoiding ambiguity with other users' identically named environments.
- **New `MAP_TO_ORIGINAL` option** (default `"True"`) — lets you skip the "map results back to original scan space" step entirely.
- **New `EXPORT_SLICER_MARKUPS` option** (default `"True"`) — additionally exports landmarks as 3D Slicer-compatible markup files (`.mrk.json`) into `Results/markups_slicer/`.
- **New output directories**: `Processed_eardrum/` (eardrum crops) and `Results/markups_slicer/` (Slicer markup exports).

