# Full Ear Segmentation

## Preprocessing pipeline — From Full CT Scan to Oriented Ear Volumes

This repository contains the preprocessing steps of the full ear segmentation pipeline. The goal of this step is to transform raw CT scans into **ear-cropped, standardized sub-volumes** aligned to the Frankfort plane and oriented consistently.

The preprocessing code is located in the folder `Final_Preprocessing` and consists of **three substeps**:

- **P1 → P2 → P3** (must be run sequentially)

More details for each substep can be found later in this section.

---

### ⚙️ Environment Setup

The repository provides **two requirements files** to create **two virtual environments**:

- `seg_env` — required for **P1**
- `landmark_env` — required for **P2 and P3**

Make sure to configure both environments before starting preprocessing.

If you plan to run jobs on a virtual machine, two `.sh` scripts are provided:

- `run_p1.sh` — runs P1 with the correct environment
- `run_p2_p3.sh` — runs P2 and P3 with the correct environment

---

### 📂 P1: `P1_Final_Preprocessing.py`

This is the **first stage (P1)** of the preprocessing pipeline for CT scans of the head region.

#### ✅ Inputs
- Raw CT scans located in a specified folder
- Files ending with `__CT.nii.gz`

#### ✅ Outputs
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



### 🔍 Steps Performed in P1

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

### 📂 P2: `P2_Final_Preprocessing.py`

This is the **second stage (P2)** of the preprocessing pipeline. It performs **anatomical landmark detection** using deep learning and **standardizes scan orientation** by aligning them to the Frankfort horizontal plane.

#### ✅ Inputs
- Preprocessed CT scans from P1 output
- Files: `{PatientID}_CT_resampled_256.nii.gz` (256×256×256 voxels)
- Pre-trained 3D U-Net model for landmark detection

#### ✅ Outputs
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


### 🔍 Steps Performed in P2

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
- Computes normal vector of Frankfort plane using **landmarks 10-13**
- Rotates plane to align with horizontal (Z-axis points down)
- Ensures consistent superior-inferior orientation

**Rotation 2: Correct Left-Right Orientation**
- Projects left-right vector (**landmark 12 → 13**) onto XY plane
- Rotates around Z-axis to align with X-axis (right → left direction)
- Ensures consistent lateral orientation
- Validates rotation angle and flags large deviations

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



### 🧠 Model Architecture

**3D U-Net** with:
- **Encoder:** 4 levels with double convolutions, max pooling
- **Decoder:** 3 levels with upsampling and skip connections
- **Normalization:** Instance normalization (better for medical imaging)
- **Output:** 6-channel heatmap (one per landmark)
- **Base features:** 16 (lightweight for 3D volumes)

---

### 📂 P3: `P3_Final_Preprocessing.py`

This is the **third and final stage (P3)** of the preprocessing pipeline. It extracts **ear-specific ROIs** from aligned scans and ensures both ears are in the **same anatomical orientation** for downstream processing.

#### ✅ Inputs
- Aligned CT scans from P2: `{PatientID}_aligned.nii.gz`
- Aligned landmarks from P2: `{PatientID}_lm_aligned.npy`
- Optional: CSV file with excluded patient IDs

#### ✅ Outputs
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

---

### 🔍 Steps Performed in P3

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


