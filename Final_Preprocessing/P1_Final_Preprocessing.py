import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import csv
import json
from datetime import datetime
from scipy.ndimage import zoom
import torch
from totalsegmentator.python_api import totalsegmentator
import pandas as pd

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  

Raw_scans_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Raw"
Processed_scans_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Processed-Data"
output_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Output"
output_img_dir = os.path.join(output_dir, "visualization_images")
output_transform_dir = os.path.join(output_dir, "transform_logs")
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_transform_dir, exist_ok=True)

acceptable_patientid_csv= r"/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/acceptable_patientid.csv"

# Quality assessment filter
pre_quality_assessed = False  # Set to True to only process scans in acceptable_patientid_csv

# Debug mode - process only one specific scan
debug_mode = False  # Set to False to process all scans
debug_scan_name = "sub01_pituitary__CT.nii.gz"  # Specific scan to process in debug mode

voxel_threshold = (1.5, 1.5, 3.5)
intensity_clip_range = (-1000, 2007)
new_spacing = [0.5, 0.5, 0.5]
interpolator = sitk.sitkBSpline

labels_info = {
    1: "Left Eye",
    2: "Right Eye",
    16: "Right Auditory Canal",
    17: "Left Auditory Canal"
}

# === Utility Functions ===
def initialize_transform_json(scan_path, img, header, affine, voxel_size):
    """Initialize the transform JSON with original scan metadata."""
    patient_id = os.path.basename(scan_path).replace('__CT.nii.gz', '')
    
    transform_data = {
        "patient_id": patient_id,
        "original_scan_path": scan_path,
        "processing_timestamp": datetime.now().isoformat(),
        "original_metadata": {
            "dimensions": [int(x) for x in img.shape],
            "voxel_spacing_mm": [float(x) for x in voxel_size],
            "affine_matrix": [[float(x) for x in row] for row in affine],
            "origin_mm": [float(x) for x in affine[:3, 3]],
            "data_type": str(header.get_data_dtype()),
            "qform_code": int(header['qform_code']),
            "sform_code": int(header['sform_code'])
        },
        "transformations": []
    }
    
    return transform_data, patient_id

def save_transform_json(transform_data, patient_id, output_dir):
    """Save the transform JSON to file."""
    json_path = os.path.join(output_dir, f"{patient_id}_transform_log.json")
    with open(json_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    return json_path

def clip_intensity(data, clip_range):
    return np.clip(data, clip_range[0], clip_range[1])

def run_segmentation(input_path, patient_output_dir, patient_id):
    output_img = totalsegmentator(input_path, task='head_glands_cavities')
    if output_img is None:
        raise ValueError("Segmentation failed or returned no output.")
    output_path = os.path.join(patient_output_dir, f"{patient_id}_CT_cavities_segmented.nii.gz")
    nib.save(output_img, output_path)
    return output_path

def compute_centroids(segmented_data, affine, input_path):
    centroids = []
    for label_id, label_name in labels_info.items():
        coords = np.argwhere(segmented_data == label_id)
        if coords.size > 0:
            centroid_voxel = coords.mean(axis=0)
            centroid_mm = nib.affines.apply_affine(affine, centroid_voxel)
            centroids.append((label_id, label_name, centroid_mm))
    return centroids

def crop_to_roi_fixed(image_data, affine, voxel_spacing, midpoint_mm, roi_size_mm=np.array([200, 270, 200]), extra_top_mm=50):
    inv_affine = np.linalg.inv(affine)
    midpoint_voxel = np.round(nib.affines.apply_affine(inv_affine, midpoint_mm)).astype(int)
    roi_size_voxels = np.round(roi_size_mm / voxel_spacing).astype(int)
    half_size = roi_size_voxels // 2
    roi_start = midpoint_voxel - half_size
    roi_end = midpoint_voxel + half_size
    extra_voxels = int(np.round(extra_top_mm / voxel_spacing[2]))
    roi_end[2] += extra_voxels
    roi_size_voxels[2] += extra_voxels
    roi = np.full(roi_size_voxels, -1000, dtype=image_data.dtype)
    data_shape = np.array(image_data.shape)
    data_start = np.maximum(roi_start, 0)
    data_end = np.minimum(roi_end, data_shape)
    roi_start_in_roi = data_start - roi_start
    roi_end_in_roi = roi_start_in_roi + (data_end - data_start)
    data_slices = tuple(slice(data_start[i], data_end[i]) for i in range(3))
    roi_slices = tuple(slice(roi_start_in_roi[i], roi_end_in_roi[i]) for i in range(3))
    roi[roi_slices] = image_data[data_slices]
    new_origin = nib.affines.apply_affine(affine, roi_start)
    new_affine = affine.copy()
    new_affine[:3, 3] = new_origin
    return roi, new_affine

def resample_volume(volume_path, new_spacing, interpolator):
    volume = sitk.ReadImage(volume_path, sitk.sitkFloat32)
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(
        volume,
        new_size,
        sitk.Transform(),
        interpolator,
        volume.GetOrigin(),
        new_spacing,
        volume.GetDirection(), 
        -1000,  # Use -1000 as default value to match CT intensity range
        volume.GetPixelID()
    )

def visualize_comparison(cropped_img, resampled_img, cropped_spacing, resampled_spacing, output_path):
    cropped_array = np.transpose(sitk.GetArrayFromImage(cropped_img), (2, 1, 0))
    resampled_array = np.transpose(sitk.GetArrayFromImage(resampled_img), (2, 1, 0))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    z = cropped_array.shape[2] // 2
    axes[0, 0].imshow(cropped_array[:, :, z], cmap='gray', aspect=cropped_spacing[1]/cropped_spacing[0])
    axes[0, 0].set_title(f'Cropped Axial (Z={z})')
    axes[0, 0].axis('off')
    x = cropped_array.shape[0] // 2
    axes[0, 1].imshow(cropped_array[x, :, :].T, cmap='gray', origin='lower', aspect=cropped_spacing[2]/cropped_spacing[1])
    axes[0, 1].set_title(f'Cropped Sagittal (X={x})')
    axes[0, 1].axis('off')
    y = cropped_array.shape[1] // 2
    axes[0, 2].imshow(cropped_array[:, y, :].T, cmap='gray', origin='lower', aspect=cropped_spacing[2]/cropped_spacing[0])
    axes[0, 2].set_title(f'Cropped Coronal (Y={y})')
    axes[0, 2].axis('off')
    z = resampled_array.shape[2] // 2
    axes[1, 0].imshow(resampled_array[:, :, z], cmap='gray', aspect=resampled_spacing[1]/resampled_spacing[0])
    axes[1, 0].set_title(f'Resampled Axial (Z={z})')
    axes[1, 0].axis('off')
    x = resampled_array.shape[0] // 2
    axes[1, 1].imshow(resampled_array[x, :, :].T, cmap='gray', origin='lower', aspect=resampled_spacing[2]/resampled_spacing[1])
    axes[1, 1].set_title(f'Resampled Sagittal (X={x})')
    axes[1, 1].axis('off')
    y = resampled_array.shape[1] // 2
    axes[1, 2].imshow(resampled_array[:, y, :].T, cmap='gray', origin='lower', aspect=resampled_spacing[2]/resampled_spacing[0])
    axes[1, 2].set_title(f'Resampled Coronal (Y={y})')
    axes[1, 2].axis('off')
    plt.suptitle("Cropped vs Resampled ROI Views")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# === First Processing Function ===
def process_single_scan(scan_path):
    try:
        print(f"\nProcessing: {scan_path}")
        img = nib.load(scan_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine
        voxel_size = header.get_zooms()
        
        # Initialize transform JSON with original metadata
        transform_data, patient_id = initialize_transform_json(scan_path, img, header, affine, voxel_size)
        
        # Create patient-specific output directory
        patient_output_dir = os.path.join(Processed_scans_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        print(f"Output directory: {patient_output_dir}")
        
        if any(voxel_size[i] > voxel_threshold[i] for i in range(3)):
            print(f"Skipping scan due to voxel size {voxel_size}")
            return
        
        print("Clipping intensity values...")
        clipped_data = clip_intensity(data, intensity_clip_range)
        clipped_img = nib.Nifti1Image(clipped_data, affine, header)
        clipped_path = os.path.join(patient_output_dir, f"{patient_id}_CT_intensity_clipped.nii.gz")
        nib.save(clipped_img, clipped_path)
        
        # Record intensity clipping in transform JSON
        transform_data["transformations"].append({
            "step": 1,
            "operation": "intensity_clipping",
            "parameters": {
                "clip_range": list(intensity_clip_range),
                "min_value": float(intensity_clip_range[0]),
                "max_value": float(intensity_clip_range[1])
            },
            "output_file": clipped_path
        })
        print("Running segmentation...")
        segmented_path = run_segmentation(clipped_path, patient_output_dir, patient_id)
        segmented_img = nib.load(segmented_path)
        segmented_data = segmented_img.get_fdata()
        
        # Record segmentation in transform JSON
        transform_data["transformations"].append({
            "step": 2,
            "operation": "totalsegmentator_segmentation",
            "parameters": {
                "task": "head_glands_cavities",
                "labels_of_interest": labels_info
            },
            "output_file": segmented_path
        })
        
        print("Computing centroids...")
        centroids = compute_centroids(segmented_data, segmented_img.affine, clipped_path)
        left_ear = next((c[2] for c in centroids if c[0] == 17), None)
        right_ear = next((c[2] for c in centroids if c[0] == 16), None)
        if left_ear is None or right_ear is None:
            print("Could not find both ear centroids.")
            skip_path = os.path.join(patient_output_dir, f"{patient_id}_skip.txt")
            with open(skip_path, 'w') as f:
                f.write("Missing ear centroids. Skipping this patient in future runs.\n")
            return
        
        # Calculate midpoint
        midpoint_mm = (left_ear + right_ear) / 2
        
        # Record centroids and midpoint in transform JSON
        centroids_dict = {}
        for label_id, label_name, centroid in centroids:
            centroids_dict[label_name] = {
                "label_id": int(label_id),
                "centroid_mm": [float(x) for x in centroid]
            }
        
        transform_data["transformations"].append({
            "step": 3,
            "operation": "centroid_calculation",
            "parameters": {
                "centroids": centroids_dict,
                "left_ear_centroid_mm": [float(x) for x in left_ear],
                "right_ear_centroid_mm": [float(x) for x in right_ear],
                "midpoint_mm": [float(x) for x in midpoint_mm]
            }
        })
        
        print("Saving centroid CSV...")
        csv_path = os.path.join(patient_output_dir, f"{patient_id}_centroids.csv")
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Label ID', 'Label Name', 'Centroid X (mm)', 'Centroid Y (mm)', 'Centroid Z (mm)'])
            for label_id, label_name, centroid in centroids:
                writer.writerow([label_id, label_name, *centroid])
        print("Cropping to ROI...")
        cropped_data, cropped_affine = crop_to_roi_fixed(clipped_data, affine, voxel_size, midpoint_mm)
        cropped_img = nib.Nifti1Image(cropped_data, cropped_affine, header)
        cropped_path = os.path.join(patient_output_dir, f"{patient_id}_CT_cropped.nii.gz")
        nib.save(cropped_img, cropped_path)
        
        # Record cropping in transform JSON
        transform_data["transformations"].append({
            "step": 4,
            "operation": "roi_cropping",
            "parameters": {
                "roi_center_mm": [float(x) for x in midpoint_mm],
                "roi_size_mm": [200, 270, 250],  # 200x270x200 + 50 extra top
                "extra_top_mm": 50,
                "cropped_dimensions": [int(x) for x in cropped_data.shape],
                "cropped_affine_matrix": [[float(x) for x in row] for row in cropped_affine],
                "cropped_origin_mm": [float(x) for x in cropped_affine[:3, 3]]
            },
            "output_file": cropped_path
        })
        
        print("Resampling cropped volume...")
        resampled_img = resample_volume(cropped_path, new_spacing, interpolator)
        resampled_path = os.path.join(patient_output_dir, f"{patient_id}_CT_resampled.nii.gz")
        sitk.WriteImage(resampled_img, resampled_path)
        
        # Record resampling in transform JSON
        resampled_size = resampled_img.GetSize()
        resampled_spacing = resampled_img.GetSpacing()
        resampled_origin = resampled_img.GetOrigin()
        
        transform_data["transformations"].append({
            "step": 5,
            "operation": "resampling",
            "parameters": {
                "original_spacing_mm": [float(x) for x in voxel_size],
                "new_spacing_mm": [float(x) for x in new_spacing],
                "interpolator": "sitkBSpline",
                "resampled_dimensions": [int(x) for x in resampled_size],
                "resampled_origin_mm": [float(x) for x in resampled_origin]
            },
            "output_file": resampled_path
        })
        
        print("Setting origin to (0, 0, 0)...")
        # Set origin to (0, 0, 0)
        resampled_img.SetOrigin((0.0, 0.0, 0.0))
        shifted_path = os.path.join(patient_output_dir, f"{patient_id}_CT_shifted_to_origin.nii.gz")
        sitk.WriteImage(resampled_img, shifted_path)
        
        # Record origin reset in transform JSON
        transform_data["transformations"].append({
            "step": 6,
            "operation": "origin_reset",
            "parameters": {
                "original_origin_mm": [float(x) for x in resampled_origin],
                "new_origin_mm": [0.0, 0.0, 0.0]
            },
            "output_file": shifted_path
        })
        
        # ---- Step 7: Standardize orientation to match reference (LAS orientation) ----
        print("Checking and standardizing orientation...")
        shifted_nib = nib.load(shifted_path)
        shifted_data = shifted_nib.get_fdata()
        shifted_affine = shifted_nib.affine
        shifted_header = shifted_nib.header
        
        # Target orientation: negative X, negative Y, positive Z (like old scans)
        target_signs = np.array([-1, -1, 1])
        
        # Get current signs from diagonal of affine
        current_affine_diagonal = np.diag(shifted_affine[:3, :3])
        current_signs = np.sign(current_affine_diagonal)
        
        print(f"  Current affine diagonal: {current_affine_diagonal}")
        print(f"  Current signs: {current_signs}")
        print(f"  Target signs: {target_signs}")
        
        # Check if orientation needs correction
        needs_flip = ~np.isclose(current_signs, target_signs)
        
        print(f"  Needs flip: {needs_flip}")
        
        if np.any(needs_flip):
            print(f"  Orientation correction needed for axes: {np.where(needs_flip)[0].tolist()}")
            
            # Apply flips to data
            corrected_data = shifted_data.copy()
            for axis in range(3):
                if needs_flip[axis]:
                    print(f"  Flipping axis {axis}")
                    corrected_data = np.flip(corrected_data, axis=axis)
            
            # Create a standardized affine matrix with target orientation
            corrected_affine = np.eye(4)
            for axis in range(3):
                # Get the spacing magnitude from the original affine
                spacing_magnitude = abs(shifted_affine[axis, axis])
                # Apply the target sign
                corrected_affine[axis, axis] = target_signs[axis] * spacing_magnitude
            # Origin remains at (0, 0, 0)
            corrected_affine[3, 3] = 1.0
            
            # Save orientation-corrected image
            corrected_img = nib.Nifti1Image(corrected_data, corrected_affine, shifted_header)
            orientation_corrected_path = os.path.join(patient_output_dir, f"{patient_id}_CT_orientation_corrected.nii.gz")
            nib.save(corrected_img, orientation_corrected_path)
            
            print(f"  Orientation corrected and saved to: {orientation_corrected_path}")
            print(f"  New affine diagonal: {np.diag(corrected_affine[:3, :3])}")
            
            # Create comparison visualization
            print("  Creating orientation correction comparison visualization...")
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Middle slices
            z_slice = shifted_data.shape[2] // 2
            x_slice = shifted_data.shape[0] // 2
            y_slice = shifted_data.shape[1] // 2
            
            # Original (uncorrected)
            axes[0, 0].imshow(shifted_data[:, :, z_slice], cmap='gray')
            axes[0, 0].set_title(f'Original Axial (Z={z_slice})')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(shifted_data[x_slice, :, :].T, cmap='gray', origin='lower')
            axes[0, 1].set_title(f'Original Sagittal (X={x_slice})')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(shifted_data[:, y_slice, :].T, cmap='gray', origin='lower')
            axes[0, 2].set_title(f'Original Coronal (Y={y_slice})')
            axes[0, 2].axis('off')
            
            # Corrected
            axes[1, 0].imshow(corrected_data[:, :, z_slice], cmap='gray')
            axes[1, 0].set_title(f'Corrected Axial (Z={z_slice})')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(corrected_data[x_slice, :, :].T, cmap='gray', origin='lower')
            axes[1, 1].set_title(f'Corrected Sagittal (X={x_slice})')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(corrected_data[:, y_slice, :].T, cmap='gray', origin='lower')
            axes[1, 2].set_title(f'Corrected Coronal (Y={y_slice})')
            axes[1, 2].axis('off')
            
            plt.suptitle(f"Orientation Correction: Original vs Corrected\nFlipped axes: {np.where(needs_flip)[0].tolist()}")
            plt.tight_layout()
            comparison_path = os.path.join(output_img_dir, f"{patient_id}_orientation_correction.png")
            plt.savefig(comparison_path)
            plt.close()
            print(f"  Comparison saved to: {comparison_path}")
            
            # Record orientation correction in transform JSON
            transform_data["transformations"].append({
                "step": 7,
                "operation": "orientation_standardization",
                "parameters": {
                    "target_orientation": "LAS (Left-Anterior-Superior)",
                    "target_signs": [int(x) for x in target_signs],
                    "original_signs": [float(x) for x in current_signs],
                    "axes_flipped": [int(x) for x in np.where(needs_flip)[0]],
                    "original_affine_matrix": [[float(x) for x in row] for row in shifted_affine],
                    "corrected_affine_matrix": [[float(x) for x in row] for row in corrected_affine]
                },
                "output_file": orientation_corrected_path
            })
            
            # Use corrected data and affine for subsequent steps
            shifted_data = corrected_data
            shifted_affine = corrected_affine
        else:
            print("  Orientation already correct - no flip needed.")
            
            # Record that no correction was needed
            transform_data["transformations"].append({
                "step": 7,
                "operation": "orientation_standardization",
                "parameters": {
                    "target_orientation": "LAS (Left-Anterior-Superior)",
                    "target_signs": [int(x) for x in target_signs],
                    "original_signs": [float(x) for x in current_signs],
                    "axes_flipped": [],
                    "correction_needed": False
                },
                "output_file": shifted_path
            })
        
        # ---- Step 8: Pad to 540x540x540 ----
        print("Padding to 540x540x540...")
        
        target_shape = (540, 540, 540)
        pad_width = [(0, max(0, target_shape[i] - shifted_data.shape[i])) for i in range(3)]
        padded_data = np.pad(shifted_data, pad_width, mode='constant', constant_values=-1000)
        
        padded_img = nib.Nifti1Image(padded_data, shifted_affine, shifted_header)
        padded_path = os.path.join(patient_output_dir, f"{patient_id}_CT_padded.nii.gz")
        nib.save(padded_img, padded_path)
        
        # Record padding in transform JSON
        transform_data["transformations"].append({
            "step": 8,
            "operation": "padding",
            "parameters": {
                "original_dimensions": [int(x) for x in shifted_data.shape],
                "target_dimensions": list(target_shape),
                "pad_width": [[int(x) for x in pw] for pw in pad_width],
                "padding_value": -1000
            },
            "output_file": padded_path
        })
        
        # ---- Step 9: Resample to 256x256x256 ----
        print("Resampling to 256x256x256...")
        resample_shape = (256, 256, 256)
        zoom_factors = [resample_shape[i] / padded_data.shape[i] for i in range(3)]
        final_resampled_data = zoom(padded_data, zoom_factors, order=1)  # Linear interpolation
        
        # Update affine
        final_affine = shifted_affine.copy()
        for i in range(3):
            final_affine[i, i] *= padded_data.shape[i] / resample_shape[i]
        
        final_img = nib.Nifti1Image(final_resampled_data, final_affine, shifted_header)
        final_output_path = os.path.join(patient_output_dir, f"{patient_id}_CT_resampled_256.nii.gz")
        nib.save(final_img, final_output_path)
        
        # Record final resampling in transform JSON
        transform_data["transformations"].append({
            "step": 9,
            "operation": "final_resampling",
            "parameters": {
                "original_dimensions": [int(x) for x in padded_data.shape],
                "target_dimensions": list(resample_shape),
                "zoom_factors": [float(x) for x in zoom_factors],
                "interpolation_order": 1,
                "final_affine_matrix": [[float(x) for x in row] for row in final_affine]
            },
            "output_file": final_output_path
        })
        
        # Save the transform JSON to output_transform_dir
        json_path = save_transform_json(transform_data, patient_id, output_transform_dir)
        print(f"Transform log saved to: {json_path}")
        
        print("Creating visualization...")
        scan_name = os.path.basename(scan_path).replace('__CT.nii.gz', '.png')
        img_path = os.path.join(output_img_dir, scan_name)
        visualize_comparison(sitk.ReadImage(cropped_path), resampled_img, voxel_size, new_spacing, img_path)
        print("Done.")
    except Exception as e:
        print(f"Failed to process scan {scan_path}: {e}")

# === Batch Runner ===
def main():
    # Load acceptable patient IDs if pre-quality assessment is enabled
    acceptable_patient_ids = None
    if pre_quality_assessed:
        print(f"Loading acceptable patient IDs from: {acceptable_patientid_csv}")
        try:
            df = pd.read_csv(acceptable_patientid_csv)
            acceptable_patient_ids = set(df['PatientID'].dropna().astype(str).tolist())
            print(f"Loaded {len(acceptable_patient_ids)} acceptable patient IDs for quality-filtered processing.")
        except Exception as e:
            print(f"Warning: Could not load acceptable patient IDs CSV: {e}")
            print("Proceeding without quality filter.")
            acceptable_patient_ids = None
    
    # Collect all scan paths ending with __CT.nii.gz
    scan_paths = []
    for root, dirs, files in os.walk(Raw_scans_dir):
        for file in files:
            if file.endswith('__CT.nii.gz'):
                scan_paths.append(os.path.join(root, file))
    
    # Filter scans if debug mode is enabled
    if debug_mode:
        scan_paths = [path for path in scan_paths if os.path.basename(path) == debug_scan_name]
        if not scan_paths:
            print(f"ERROR: Debug scan '{debug_scan_name}' not found in {Raw_scans_dir}")
            return
        print(f"\n{'='*60}")
        print(f"DEBUG MODE ENABLED - Processing only: {debug_scan_name}")
        print(f"{'='*60}\n")
    
    print(f"Found {len(scan_paths)} scans to process.")
    print(f"\n{'='*60}")
    print(f"Starting P1 Processing Pipeline")
    print(f"{'='*60}\n")
    
    for scan_path in scan_paths:
        patient_id = os.path.basename(scan_path).replace('__CT.nii.gz', '')
        
        # Skip if patient not in acceptable list (when quality assessment is enabled)
        if acceptable_patient_ids is not None and patient_id not in acceptable_patient_ids:
            print(f"Skipping patient {patient_id} - not in acceptable patient IDs list.")
            continue
        
        patient_output_dir = os.path.join(Processed_scans_dir, patient_id)
        p1_output = os.path.join(patient_output_dir, f"{patient_id}_CT_resampled_256.nii.gz")
        skip_path = os.path.join(patient_output_dir, f"{patient_id}_skip.txt")
        
        # Check if patient should be skipped entirely
        if os.path.exists(skip_path):
            print(f"Skipping patient {patient_id} - marked to skip due to missing centroids.")
            continue
        
        # Run P1 if output doesn't exist
        if os.path.exists(p1_output):
            print(f"P1 already complete for {patient_id}. Skipping.")
        else:
            print(f"\n{'='*60}")
            print(f"Starting P1 for {patient_id}")
            print(f"{'='*60}")
            process_single_scan(scan_path)


if __name__ == "__main__":
    main()