# Full Ear Segmentation

## Step 1: Preprocessing — From Full CT Scan to Oriented Ear Volumes

This repository contains **Step 1: Preprocessing** of the full ear segmentation pipeline. The goal of this step is to transform raw CT scans into **ear-cropped, standardized sub-volumes** aligned to the Frankfort plane and oriented consistently.

The preprocessing code is located in the folder `Final_Preprocessing` and consists of **three substeps**:

- **P1 → P2 → P3** (must be run sequentially)

More details for each substep can be found later in this section.

---

## ⚙️ Environment Setup

The repository provides **two requirements files** to create **two virtual environments**:

- `seg_env` — required for **P1**
- `landmark_env` — required for **P2 and P3**

Make sure to configure both environments before starting preprocessing.

If you plan to run jobs on a virtual machine, two `.sh` scripts are provided:

- `run_p1.sh` — runs P1 with the correct environment
- `run_p2_p3.sh` — runs P2 and P3 with the correct environment

---

## 📂 P1: `P1_Final_Preprocessing.py`

This is the **first stage (P1)** of the preprocessing pipeline for CT scans of the head region.

### ✅ Inputs
- Raw CT scans located in a specified folder
- Files ending with `__CT.nii.gz`

### ✅ Outputs
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

---

## 🔍 Steps Performed in P1

### **Step 1: Quality Filtering**
- Checks voxel spacing against thresholds (1.5mm, 1.5mm, 3.5mm)
- Skips scans exceeding these thresholds
- Optional: process only pre-approved patient IDs

### **Step 2: Intensity Clipping**
- Clips CT intensity values to range **[-1000, 2007 HU]**
- Removes extreme outliers while preserving diagnostic info

### **Step 3: Anatomical Segmentation**
- Uses **TotalSegmentator** (`head_glands_cavities` task) to identify:
  - Left Eye (label 1)
  - Right Eye (label 2)
  - Right Auditory Canal (label 16)
  - Left Auditory Canal (label 17)

### **Step 4: Centroid Calculation**
- Computes 3D centroids for each segmented structure
- Calculates midpoint between left and right ear canals
- Saves centroid coordinates to CSV
- Skips patients if both ear centroids cannot be found

### **Step 5: ROI Cropping**
- Crops fixed-size ROI around ear midpoint
- ROI dimensions: **200mm × 270mm × 250mm**
- Centers volume on ear midpoint
- Pads with -1000 HU where ROI extends beyond scan boundaries

### **Step 6: Resampling to Isotropic Spacing**
- Resamples to **0.5mm × 0.5mm × 0.5mm** spacing
- Uses B-spline interpolation for smooth results

### **Step 7: Origin Reset**
- Sets image origin to (0, 0, 0) for standardized coordinates

### **Step 8: Padding to Fixed Size**
- Pads volume to **540 × 540 × 540 voxels**
- Uses -1000 HU for padding values

### **Step 9: Final Resampling**
- Resamples to **256 × 256 × 256 voxels**
- Uses linear interpolation for efficiency

