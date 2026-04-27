
# Full Ear Canal Segmentation

## Introduction

This repository contains a complete end-to-end pipeline for **ear canal segmentation** and **anatomical landmark detection** from CT scans. Starting from raw NIfTI CT scans of the head, it produces 3D surface meshes (STL) of the ear canal along with anatomical landmark coordinates.

The pipeline consists of five main stages:

1. **Preprocessing (P1-P4)** — Transforms raw CT scans into ear-cropped, orientation-standardized, normalized sub-volumes ready for inference.
2. **Tissue vs Air Inference** — Applies trained 3D U-Net models with dual task-specific heads to simultaneously predict ear canal segmentations (tissue vs air) and anatomical landmarks.
3. **Bone Segmentation Inference** — Applies a trained nnU-Net model to predict bone structures (skull, mandible) and additional landmarks.
4. **Postprocessing** — Combines predictions into visualization-ready outputs: JSON markup files (compatible with 3D Slicer), NIfTI segmentation masks, and STL surface meshes.
5. **Metric Extraction** — Computes ear canal centerlines (via vmtk) and anatomical measurements (lengths, diameters, cross-sections, angles) from the segmentation results.

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
conda create -n metric_env -c conda-forge --override-channels python=3.11 vmtk -y
conda activate metric_env
pip install -r env_req/metric_env_req.txt
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

# Python environments
SEG_ENV="/path/to/seg_env"
LANDMARK_ENV="/path/to/landmark_env"

# Models
LANDMARK_MODEL="/path/to/landmark_model.pth"
INFERENCE_LOGS_DIR="/path/to/model/logs"
POSTPROCESSING_RUN="run_20260210_105409"   # Which model run to use

# Optional filters
ACCEPTABLE_PATIENTID_CSV="/path/to/acceptable_patientid.csv"
PRE_QUALITY_ASSESSED="False"
EXCLUDED_SCANS_CSV="/path/to/excluded_scans.csv"
```

The script runs all 8 steps automatically (**P1 → P2 → P3 → P4 → Tissue/Air Inference → Bone Inference → Postprocessing → Metric Extraction**), activates the correct environment for each step, and stops with a clear error if any step fails.

### Configuration Options

| Option | Default | Description |
|---|---|---|
| `NO_EYES` | `"False"` | Set to `"True"` if scans don't include eye structures. Uses mandible top points instead of eye landmarks for Frankfort plane fitting. Must be consistent across P2 and P3. |
| `SKIP_ALIGNMENT` | `"False"` | Set to `"True"` to skip Frankfort plane alignment. Only performs landmark detection without rotation. Must be consistent across P2 and P3. |
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
```

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


## Retraining Tissue/Air Segmentation and Landmark Placement


If you want to retrain the model (for example, to add more data or improve performance), follow these steps:

### 1. Prepare Your Data

- **CT Scans:** Place your raw NIfTI CT volumes in a directory (default: `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain/CTs`).
- **Segmentation Masks:** Place your ear mask segmentations (NRRD format) in `/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/segmentations/ear_masks_train` (or specify a different directory).
- **Landmark JSONs:** Place your anatomical landmark JSON files in `/kbnnfsserver/erhdata/Raw/EarScans/Images/HECKTOR 2025 Training Data/Annotations/landmarks/ear_anatomical_landmarks_train` (or specify a different directory).

### 2. Run the Retraining Script

Use the provided shell script to run the **full preprocessing pipeline (P1–P4)**, generate heatmaps, and start training:

```bash
cd sh_files
bash run_retrain_tissue_air.sh
```

This will:
- Run P1–P4 preprocessing on your CTs (outputting to `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain/CTs_P4` by default)
- Generate heatmaps from the P4-processed CTs and your landmark JSONs
- Train the tissue/air segmentation and landmark placement model using the P4-processed CTs, segmentation masks, and heatmaps
- Save all outputs (processed CTs, heatmaps, logs, models) under `/kbnnfsserver/erhdata/Processed-Data/SBEO/Retrain` by default

#### Customizing Input/Output Directories
You can override the default directories by passing arguments to the script:

```bash
bash run_retrain_tissue_air.sh <CT_DIR> <JSON_DIR> <SEG_DIR> <HEATMAP_DIR> <LOG_DIR> <MODEL_DIR>
```
For example:
```bash
bash run_retrain_tissue_air.sh /path/to/CTs /path/to/landmarks /path/to/masks /path/to/heatmaps /path/to/logs /path/to/models
```

**Note:** The CT directory you provide (`<CT_DIR>`) should contain your raw CTs. The script will automatically create intermediate directories for each preprocessing stage (P1–P4) under the retraining folder.

### 3. Adding More Data

To improve training, simply add more CTs, segmentation masks, and landmark JSONs to the respective directories before running the script. The pipeline will automatically use all available data in the specified folders.

---

## References

- **TotalSegmentator:** Wasserthal et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.230024

