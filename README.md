# Full Ear Segmentation

## Complete Pipeline — From Raw CT Scans to 3D Ear Canal Models

This repository contains a complete pipeline for ear canal segmentation and landmark detection from CT scans. The pipeline consists of three main stages:

1. **Preprocessing (P1-P4)** — Transform raw CT scans into ear-cropped, standardized sub-volumes
2. **Inference** — Apply trained deep learning models to generate segmentations and landmarks
3. **Postprocessing** — Create visualization-ready outputs (JSON markups, NIfTI masks, STL meshes)

---

## Quick Start — Run the Full Pipeline

The simplest way to run everything (preprocessing + inference + postprocessing) is with the **unified pipeline script**. All configuration is done in the script header — no need to edit any Python files.

```bash
./sh_files/run_full_pipeline.sh
```

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
POSTPROCESSING_RUN="run_20260210_105409"   # Which model run to use for postprocessing

# Optional filters
ACCEPTABLE_PATIENTID_CSV="/path/to/acceptable_patientid.csv"
PRE_QUALITY_ASSESSED="False"
EXCLUDED_SCANS_CSV="/path/to/excluded_scans.csv"
```

The script runs all 6 steps automatically (P1 → P2 → P3 → P4 → Inference → Postprocessing), activates the correct environment for each step, and stops with a clear error if any step fails.

---

## Preprocessing Pipeline — From Full CT Scan to Oriented Ear Volumes

The preprocessing code is located in the folder `preprocessing/` and consists of **four sequential substeps**:

- **P1 → P2 → P3 → P4** (must be run sequentially)

More details for each substep can be found later in this section.

---

### Environment Setup

The repository provides **two requirements files** to create **two virtual environments**:

- `seg_env` — required for **P1** (preprocessing)
- `landmark_env` — required for **P2, P3, P4** (preprocessing), **inference**, and **postprocessing**

#### Automatic Setup (Recommended)

Use the provided setup script to automatically create both environments:

**Option 1: Bash script (Linux/Mac)**
```bash
./setup_environments.sh
```

**Option 2: Python script (Cross-platform)**
```bash
python setup_environments.py
# or
python3 setup_environments.py
```

> **Important:** Run the setup script with your **system Python** (not from within a virtual environment). The script will create the virtual environments for you. If you get "No such file or directory" errors, make sure you're not already in a virtual environment - deactivate it first with `deactivate`.

Both scripts will:
- Create both virtual environments (`seg_env` and `landmark_env`)
- Install all required packages from `seg_env_req.txt` and `landmark_env_req.txt`
- Handle any existing environments (asks before overwriting)
- Provide clear progress updates

**After setup completes**, you can activate the environments:
```bash
source seg_env/bin/activate      # For P1
source landmark_env/bin/activate # For P2-P4
```

#### Manual Setup

Alternatively, you can create the environments manually:

```bash
# Create seg_env for P1
python3 -m venv seg_env
source seg_env/bin/activate
pip install -r seg_env_req.txt
deactivate

# Create landmark_env for P2-P4
python3 -m venv landmark_env
source landmark_env/bin/activate
pip install -r landmark_env_req.txt
deactivate
```

Make sure both environments are configured before starting preprocessing.

---

### Quick Start - Unified Preprocessing Script

The easiest way to run the complete preprocessing pipeline is using the **unified shell script** that runs all four steps automatically:

```bash
cd sh_files
./run_full_preprocessing.sh
```

**Before running**, edit the configuration section at the top of `sh_files/run_full_preprocessing.sh`:

```bash
# Directory paths (REQUIRED - edit these)
RAW_SCANS_DIR="/path/to/your/raw/scans"
PROCESSED_SCANS_DIR="/path/to/save/processed/scans"
OUTPUT_DIR="/path/to/save/outputs"

# Processing options
NO_EYES="False"              # Set to "True" if scans don't include eyes
SKIP_ALIGNMENT="False"       # Set to "True" to skip alignment step

# Python environments
SEG_ENV="/path/to/seg_env"           # For P1
LANDMARK_ENV="/path/to/landmark_env" # For P2, P3, P4

# Landmark detection model (for P2)
LANDMARK_MODEL="/path/to/your/landmark_model.pth"
```

The script will automatically:
- Activate `seg_env` for P1
- Switch to `landmark_env` for P2-P4
- Run P1 (intensity clipping, segmentation, cropping, resampling)
- Run P2 (landmark detection and alignment)
- Run P3 (ROI cropping around ears)
- Run P4 (upsampling and normalization for inference)
- Stop with clear error messages if any step fails

---

### Running Individual Steps

If you need to run individual preprocessing steps or run jobs on a virtual machine, you can either:

**Option 1: Use individual shell scripts** (requires editing paths in scripts):
- `run_p1.sh` — runs P1 with the seg_env environment
- `run_p2_p3.sh` — runs P2 and P3 with the landmark_env environment

**Option 2: Run Python scripts directly with command-line arguments**:

```bash
# P1
python preprocessing/p1_preprocessing.py \
    --raw_scans_dir "/path/to/raw" \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output"

# P2
python preprocessing/p2_preprocessing.py \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output" \
    --landmark_model "/path/to/model.pth" \
    --no_eyes "False" \
    --skip_alignment "False"

# P3
python preprocessing/p3_preprocessing.py \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output" \
    --no_eyes "False" \
    --skip_alignment "False"

# P4
python preprocessing/p4_preprocessing.py \
    --processed_scans_dir "/path/to/processed" \
    --output_dir "/path/to/output"
```

---

### P1: `p1_preprocessing.py`

This is the **first stage (P1)** of the preprocessing pipeline for CT scans of the head region.

#### Inputs
- Raw CT scans located in a specified folder
- Files ending with `__CT.nii.gz`

#### Outputs
- **Final processed CT scan:**
  - `{PatientID}_CT_resampled_256.nii.gz`
- **Intermediate files:**
  - `{PatientID}_CT_intensity_clipped.nii.gz`
  - `{PatientID}_CT_cavities_segmented.nii.gz`
  - `{PatientID}_CT_cropped.nii.gz`
  - `{PatientID}_CT_resampled.nii.gz`
  - `{PatientID}_CT_shifted_to_origin.nii.gz`
  - `{PatientID}_CT_padded.nii.gz`
- **Metadata files:**
  - `{PatientID}_centroids.csv` — anatomical landmark coordinates
  - `{PatientID}_transform_log.json` — complete transformation history
  - `{PatientID}_skip.txt` — created if processing cannot complete
- **Visualization:**
  - Comparison images showing cropped vs resampled volumes



### Steps Performed in P1

#### **Step 1: Quality Filtering**
- Checks voxel spacing against thresholds (1.5mm, 1.5mm, 3.5mm)
- Skips scans exceeding these thresholds
- Optional: process only pre-approved patient IDs

#### **Step 2: Intensity Clipping**
- Clips CT intensity values to range **[-1000, 2007 HU]**

#### **Step 3: Anatomical Segmentation**
- Uses **TotalSegmentator** (`head_glands_cavities` task) to identify:
  - Left Eye (label 1)
  - Right Eye (label 2)
  - Right Auditory Canal (label 16)
  - Left Auditory Canal (label 17)

#### **Step 4: Centroid Calculation**
- Computes 3D centroids for each segmented structure
- Calculates midpoint between left and right ear canals
- Saves centroid coordinates to CSV
- Skips patients if both ear centroids cannot be found

#### **Step 5: ROI Cropping**
- Crops fixed-size ROI around ear midpoint
- ROI dimensions: **200mm × 270mm × 250mm**
- Centers volume on ear midpoint
- Pads with -1000 HU where ROI extends beyond scan boundaries

#### **Step 6: Resampling to Isotropic Spacing**
- Resamples to **0.5mm × 0.5mm × 0.5mm** spacing
- B-spline interpolation

#### **Step 7: Origin Reset**
- Sets image origin to (0, 0, 0) for standardized coordinates

#### **Step 8: Padding to Fixed Size**
- Pads volume to **540 × 540 × 540 voxels**
- Uses minimum value (-1000 HU) for padding values

#### **Step 9: Final Resampling**
- Resamples to **256 × 256 × 256 voxels**
- Linear interpolation

---

### P2: `p2_preprocessing.py`

This is the **second stage (P2)** of the preprocessing pipeline. It performs **anatomical landmark detection** using deep learning and **standardizes scan orientation** by aligning them to the Frankfort horizontal plane.

#### Inputs
- Preprocessed CT scans from P1 output
- Files: `{PatientID}_CT_resampled_256.nii.gz` (256×256×256 voxels)
- Pre-trained 3D U-Net model for landmark detection

#### Outputs
- **Aligned scan:**
  - `{PatientID}_aligned.nii.gz` — orientation-corrected CT scan
- **Landmark detection results:**
  - `predicted_landmarks.txt` — human-readable coordinates
  - `predicted_landmarks.json` — machine-readable format
  - `pred_heatmap_landmark{ID}.nii.gz` — individual probability maps (6 files)
- **Aligned landmarks:**
  - `{PatientID}_lm_aligned.npy` — aligned landmark coordinates (NumPy array)
- **Visualizations** (for flagged scans only):
  - `{PatientID}_large_rotation_vis.png` — 3-view display with landmarks
- **Transformation logs:**
  - `{PatientID}_transform_log_P2.json` — complete processing record
- **Aggregated CSVs:**
  - `all_landmark_predictions.csv` — all detected landmarks (world + voxel coordinates)
  - `all_aligned_landmarks.csv` — post-alignment landmark positions
  - `flagged_large_rotations.csv` — scans requiring manual review


### Steps Performed in P2

#### **Step 9: Landmark Detection**
- Loads pre-trained **3D U-Net model** with instance normalization
- Generates **probability heatmaps** for each landmark (6 channels output)
- Applies sigmoid activation and **0.5 threshold**
- Computes **weighted centroids** from heatmaps for sub-voxel precision
- Converts voxel coordinates to world coordinates (mm)
- **Coordinate correction:** transforms from prediction space `(x,y,z)` to anatomical space `(z,-y,-x)`
- Saves individual heatmaps as NIfTI files for each landmark
- **Landmarks detected:** IDs 8, 9, 10, 11, 12, 13

#### **Step 10: Frankfort Plane Alignment**
A **two-step rotation** process to standardize orientation:

**Rotation 1: Make Frankfort Plane Horizontal**
- Computes normal vector of Frankfort plane using **landmarks 10-13** (or 8-11 + mandible points if `no_eyes=True`)
- Rotates plane to align with horizontal (Z-axis points down)
- Ensures consistent superior-inferior orientation

**Rotation 2: Correct Left-Right Orientation**
- Projects left-right vector (**landmark 12 → 13**, or mandible top points if `no_eyes=True`) onto XY plane
- Rotates around Z-axis to align with X-axis (right → left direction)
- Ensures consistent lateral orientation
- Validates rotation angle and flags large deviations

**No-Eyes Mode:**
- When `no_eyes=True`, uses **TotalSegmentator** to segment craniofacial structures
- Extracts mandible top points on left and right sides as replacements for eye landmarks (12-13)
- Useful for scans that don't include eye structures
- Reference: Wasserthal et al. (2023) TotalSegmentator

**Skip Alignment Mode:**
- When `skip_alignment=True`, only performs landmark detection without alignment
- Useful for testing landmark detection or when alignment is not needed

**Quality Control:**
- Measures Z-axis rotation angle from step 2
- **Flags scans** with rotation **> 10°** as potentially problematic
- Creates **3-view visualizations** for flagged scans (axial, sagittal, coronal)
- Large rotations likely indicate landmark detection errors

**Volume Transformation:**
- Applies combined rotation matrix to entire CT volume
- Uses **affine transformation** with linear interpolation
- Fills extrapolated regions with **-1000 HU** (air density)
- Validates handedness (fixes reflections if detected)
- **Aligns all 6 landmarks** using the same rotation transformation



### Model Architecture

**3D U-Net** with:
- **Encoder:** 4 levels with double convolutions, max pooling
- **Decoder:** 3 levels with upsampling and skip connections
- **Normalization:** Instance normalization 
- **Output:** 6-channel heatmap (one per landmark)
- **Base features:** 16 (lightweight for 3D volumes)

---

### P3: `p3_preprocessing.py`

This is the **third stage (P3)** of the preprocessing pipeline. It extracts **ear-specific ROIs** from aligned scans and ensures both ears are in the **same anatomical orientation** for downstream processing.

#### Inputs
- Aligned CT scans from P2: `{PatientID}_aligned.nii.gz`
- Aligned landmarks from P2: `{PatientID}_lm_aligned.npy`
- Optional: CSV file with excluded patient IDs

#### Outputs
- **Final cropped ear volumes (per patient):**
  - `{PatientID}_right_ear.nii.gz` — final right ear volume
  - `{PatientID}_left_ear.nii.gz` — final left ear volume (mirrored)
- **Intermediate files (per patient):**
  - `{PatientID}_right_ear_cropped.nii.gz` — right ear ROI
  - `{PatientID}_left_ear_cropped.nii.gz` — left ear ROI
  - `{PatientID}_left_ear_cropped_mirrored.nii.gz` — mirrored left ear
  - `{PatientID}_right_ear_cropped_origin_reset.nii.gz` — right ear with origin reset
  - `{PatientID}_left_ear_cropped_mirrored_origin_reset.nii.gz` — left ear final
- **Transformation logs:**
  - `{PatientID}_transform_log_P3.json` — complete processing record
- **Centralized output:**
  - All final ears saved to `Final_Cropped_Ears_256/` directory


### Steps Performed in P3

#### **Step 11: Right Ear ROI Cropping**
- Centers ROI at **Landmark 10** (right ear canal)
- Applies offset: **[+20, +10, -5] mm** from landmark position
- Crops **90 × 90 × 90 voxel** cube
- Pads with **-1000 HU** where ROI extends beyond scan boundaries

#### **Step 12: Left Ear ROI Cropping**
- Centers ROI at **Landmark 11** (left ear canal)
- Applies offset: **[-20, +10, -5] mm** from landmark position (x-axis flipped)
- Crops **90 × 90 × 90 voxel** cube
- Pads with **-1000 HU** where ROI extends beyond scan boundaries

#### **Step 13: Left Ear Mirroring**
- **Left ear only:** Flips volume along **axis 0** (x-axis)
- Ensures both ears have the **same anatomical orientation**
- Maintains affine matrix (changes anatomical interpretation)
- Critical for bilateral symmetry in downstream models

#### **Step 14: Origin Reset**
- Sets image origin to **(0, 0, 0)** for both ears
- Applied to both right ear and mirrored left ear
- Ensures standardized coordinate system
- Saves final volumes to both patient directory and centralized folder

---

### P4: `p4_preprocessing.py`

This is the **fourth and final stage (P4)** of the preprocessing pipeline. It upsamples cropped ear volumes to a consistent resolution and normalizes intensity values for inference.

#### Inputs
- Cropped ear volumes from P3:
  - `{PatientID}_left_ear_cropped_mirrored_origin_reset.nii.gz`
  - `{PatientID}_right_ear_cropped_origin_reset.nii.gz`

#### Outputs
- **Final normalized scans for inference:**
  - `{PatientID}_left_ear.nii.gz` — 128³ normalized left ear
  - `{PatientID}_right_ear.nii.gz` — 128³ normalized right ear
- All outputs saved to `Output/Inference_Scans/` directory

### Steps Performed in P4

#### **Step 15: Upsampling to 128³**
- Resamples ear volumes to **128 × 128 × 128 voxels**
- Uses **trilinear interpolation** (order=1)
- Adjusts affine matrix to maintain physical dimensions
- Updates voxel spacing accordingly

#### **Step 16: Intensity Normalization**
- Clips CT values to range **[-1000, 2007 HU]**
- Normalizes to **[0, 1]** range: `(clipped - min_HU) / (max_HU - min_HU)`
- Output data type: **float32**
- Prepares data for deep learning model inference

---

## Inference and Postprocessing Pipeline

After completing the preprocessing pipeline (P1-P4), you can run **model inference** to generate ear canal segmentations and landmark predictions, followed by **postprocessing** to create visualization-ready outputs.

---

### Quick Start - Unified Inference & Postprocessing Script

The easiest way to run inference and postprocessing is using the **unified shell script**:

```bash
cd sh_files
./run_inference_and_postprocessing.sh
```

**Before running**, you must edit the configuration paths in two files:

1. **Inference Configuration** — Edit paths in [`model/test_multiclass.py`](model/test_multiclass.py):
   ```python
   # Directory containing preprocessed inference scans (from P4)
   TEST_DIR = "/path/to/Output/Inference_Scans"
   
   # Path to trained model checkpoint
   MODEL_PATH_TEMPLATE = "/path/to/Logs/{run_name}/best_model.pth"
   
   # Directory to save predictions
   OUTPUT_PREDICTIONS_DIR_TEMPLATE = "/path/to/Logs/{run_name}/test_predictions"
   
   # Model run names (trained models to evaluate)
   BF_32_RUN = "run_20260210_102624"  # Example run name
   BF_48_RUN = "run_20260210_103433"
   # ... etc
   ```

2. **Postprocessing Configuration** — Edit paths in [`postprocessing/markup_comb_stl_generator.py`](postprocessing/markup_comb_stl_generator.py):
   ```python
   # Path to FH plane landmarks (from preprocessing P2)
   FH_plane_lm = "/path/to/Output/Aligned_Landmarks/landmark_positions_after_cropping.csv"
   
   # Path to predicted canal landmarks (from inference)
   predicted_landmarks = "/path/to/Logs/{run_name}/test_predictions/predicted_landmark_coordinates.csv"
   
   # Directory containing predicted segmentation masks (from inference)
   masks_dir = "/path/to/Logs/{run_name}/test_predictions"
   
   # Output directories for postprocessing results
   output_dir_markups = "/path/to/Output/Test/all_markups"
   output_dir_markups_no_fh = "/path/to/Output/Test/markups_no_FH"
   output_dir_masks = "/path/to/Output/Test/all_masks"
   output_dir_stl = "/path/to/Output/Test/all_stl"
   ```

The script will automatically:
- Activate the `landmark_env` environment
- Run inference to generate segmentation masks and landmark predictions
- Run postprocessing to create markup files, cleaned masks, and STL meshes
- Display clear error messages if any step fails

---

### Running Scripts Individually

You can also run the scripts separately:

```bash
# Activate environment
source landmark_env/bin/activate

# Run inference (edit configuration in file first)
python model/test_multiclass.py

# Run postprocessing (edit configuration in file first)
python postprocessing/markup_comb_stl_generator.py
```

---

### Inference: `test_multiclass.py`

Runs trained 3D U-Net models to generate ear canal segmentation masks and anatomical landmark predictions.

#### Inputs
- **Preprocessed ear volumes** (from P4):
  - `{PatientID}_left_ear.nii.gz` (128³ normalized volumes)
  - `{PatientID}_right_ear.nii.gz`
- **Trained model checkpoints** (`.pth` files)

#### Outputs (per model configuration)
- **Segmentation masks:**
  - `{PatientID}_{ear_side}_pred_mask.nii.gz` — binary ear canal segmentation
- **Landmark predictions:**
  - `predicted_landmark_coordinates.csv` — landmark positions (world + voxel coords)
  - Individual landmark heatmaps (optional)
- **Evaluation metrics** (if ground truth available):
  - Dice scores, precision, recall per patient
  - Aggregated performance statistics

#### What it does

**Model Architecture:**
- **3D U-Net** with two task-specific heads:
  - **Segmentation head:** Binary ear canal mask prediction
  - **Landmark head:** 7-channel heatmap for anatomical points
- Supports multiple model configurations with different base features (32, 48, 64)
- Two architecture variants: "late divergence" and "early divergence"

**Inference Process:**
1. Loads preprocessed 128³ ear volumes
2. Runs forward pass through trained model(s)
3. Applies sigmoid activation and thresholding for segmentation
4. Extracts landmark coordinates from heatmap centroids
5. Converts predictions from voxel space to world coordinates (mm)
6. Saves segmentation masks as NIfTI files
7. Aggregates all predictions into CSV files

**Predicted Landmarks (IDs 1-7):**
- **1:** Cartilaginous Canal Point
- **2:** Bony Canal Point
- **3:** Eardrum
- **4:** Top Retroauricular Sulcus
- **5:** Bottom Retroauricular Sulcus
- **6:** 1st Bend
- **7:** 2nd Bend

---

### Postprocessing: `markup_comb_stl_generator.py`

Combines inference results with preprocessing landmarks to create visualization-ready outputs.

#### Inputs
- **Canal landmarks** (from inference):
  - `predicted_landmark_coordinates.csv` — landmarks 1-7
- **Segmentation masks** (from inference):
  - `{PatientID}_{ear_side}_pred_mask.nii.gz`
- **FH plane landmarks** (from preprocessing P2) — *optional but recommended*:
  - `landmark_positions_after_cropping.csv` — landmarks 8-13

#### Outputs
- **Markup JSON files** (per patient ear):
  - `{PatientID}_{ear_side}.json` — combined landmark coordinates
  - Compatible with 3D Slicer and other visualization tools
  - Includes landmarks 1-10 (canal + FH plane)
- **NIfTI masks** (per patient ear):
  - `{PatientID}_{ear_side}_mask.nii.gz` — cleaned segmentation mask
- **STL mesh files** (per patient ear):
  - `{PatientID}_{ear_side}_canal.stl` — 3D surface mesh of ear canal
  - Binary STL format, ready for 3D visualization

#### What it does

**Landmark Combination:**
1. Loads canal landmarks (1-7) from inference results
2. Loads FH plane landmarks (8-13) from preprocessing (if available)
3. Maps landmarks to correct output IDs based on ear side
4. Handles missing landmarks gracefully (e.g., when `no_eyes=True`)
5. Creates JSON markups with all available landmarks

**Mask Processing:**
1. Loads predicted segmentation masks from inference
2. Converts to NIfTI format with proper spatial metadata
3. Saves cleaned masks with anatomically correct orientation

**STL Generation:**
1. Extracts 3D surface from binary segmentation mask
2. Uses **marching cubes** algorithm for mesh extraction
3. Generates binary STL files for efficient storage
4. Meshes can be viewed in 3D Slicer, MeshLab, Blender, etc.

**Output Organization:**
- If FH plane landmarks available → saves to `all_markups/`
- If FH plane landmarks missing → saves to `markups_no_FH/`
- All masks saved to `all_masks/`
- All STL files saved to `all_stl/`

---

### Complete Output Structure

After running the full pipeline (preprocessing + inference + postprocessing), the output directory structure will be:

```
OUTPUT_DIR/
├── transform_logs/                    # JSON logs of all transformations (P1, P2, P3)
├── visualization_images/              # Visualization images from P1
├── Landmarks/                         # Predicted landmarks from P2
│   └── {patient_id}/
│       ├── predicted_landmarks.txt
│       ├── predicted_landmarks.json
│       └── pred_heatmap_landmark*.nii.gz
├── Aligned_Landmarks/                 # Aligned landmarks (if alignment enabled)
│   ├── all_landmark_predictions.csv
│   ├── all_aligned_landmarks.csv
│   ├── flagged_large_rotations.csv
│   └── landmark_positions_after_cropping.csv
├── Final_Cropped_Ears_256/           # Cropped ear ROIs from P3
│   ├── {patient_id}_left_ear_cropped_mirrored_origin_reset.nii.gz
│   └── {patient_id}_right_ear_cropped_origin_reset.nii.gz
├── Inference_Scans/                  # Final normalized scans from P4
│   ├── {patient_id}_left_ear.nii.gz
│   └── {patient_id}_right_ear.nii.gz
└── Test/                             # Postprocessing outputs
    ├── all_markups/                  # JSON landmark files (with FH plane)
    │   └── {patient_id}_{ear_side}.json
    ├── markups_no_FH/                # JSON landmark files (canal only)
    │   └── {patient_id}_{ear_side}.json
    ├── all_masks/                    # NIfTI segmentation masks
    │   └── {patient_id}_{ear_side}_mask.nii.gz
    └── all_stl/                      # STL surface meshes
        └── {patient_id}_{ear_side}_canal.stl

MODEL_LOGS_DIR/                       # Inference results (separate from OUTPUT_DIR)
└── {run_name}/
    └── test_predictions/
        ├── predicted_landmark_coordinates.csv  # All canal landmarks
        └── {patient_id}_{ear_side}_pred_mask.nii.gz  # Segmentation masks

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

### Key Configuration Options

#### `no_eyes` (P2 and P3)
- **Default:** `False`
- **When to use:** Set to `True` if scans don't include eye structures
- **Effect:** Uses mandible top points instead of eye landmarks (12-13) for plane fitting
- **Note:** Must be set consistently across P2 and P3

#### `skip_alignment` (P2 and P3)
- **Default:** `False`
- **When to use:** Set to `True` to skip the Frankfort plane alignment step
- **Effect:** Only performs landmark detection without rotation alignment
- **Note:** Must be set consistently across P2 and P3

#### `pre_quality_assessed` (P1)
- **Default:** `False`
- **When to use:** Set to `True` to only process scans listed in `acceptable_patientid.csv`
- **Effect:** Filters scans before processing

---

### References

- **TotalSegmentator:** Wasserthal et al. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.230024

---

