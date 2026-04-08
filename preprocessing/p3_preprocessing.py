import os
import sys
import numpy as np
import nibabel as nib
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Default configuration (can be overridden by command-line arguments)
Processed_scans_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Processed-Data"
Processed_data_output_dir = Processed_scans_dir
output_dir = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Output"
output_transform_dir = os.path.join(output_dir, "transform_logs")
Aligned_landmarks_dir = os.path.join(output_dir, "Aligned_Landmarks")
Landmarks_dir = os.path.join(output_dir, "Landmarks")  # Predicted (non-aligned) landmarks from P2
excluded_scans_csv= r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Excluded_scans_cropping.csv"

no_eyes = False
skip_alignment = False 
# Landmark configuration
landmark_ids = [8, 9, 10, 11, 12, 13]

# Ear cropping configuration
ROI_SIZE = [90, 90, 90]  # voxels (x, y, z)
MIRROR_AXIS = 0  # Axis to use for mirroring left ear (confirmed: axis 0)

# Define which landmarks to crop
# Each entry: (landmark_index_in_array, landmark_id, name, offset_mm, is_left_ear)
# offset_mm: [x, y, z] offset in mm from landmark position
# is_left_ear: True for left ear (positive x offset), False for right ear (negative x offset)
LANDMARKS_TO_CROP = [
    (2, 10, 'right_ear', [20, 10, -5], True),     # landmark_10 - right ear
    (3, 11, 'left_ear', [20, 10, -5], False),   # landmark_11 - left ear (x will be flipped)
]


# === Helper Functions ===
def initialize_transform_json(scan_path, scan_name, aligned_scan_shape, affine, step_number=11):
    """Initialize the transform JSON for P3, continuing from P2."""
    transform_data = {
        "scan_name": scan_name,
        "aligned_scan_path": scan_path,
        "processing_timestamp": datetime.now().isoformat(),
        "processing_stage": "P3_ROI_Cropping",
        "input_metadata": {
            "dimensions": [int(x) for x in aligned_scan_shape],
            "affine_matrix": [[float(x) for x in row] for row in affine],
            "origin_mm": [float(x) for x in affine[:3, 3]],
            "voxel_spacing_mm": [float(np.abs(affine[i, i])) for i in range(3)]
        },
        "transformations": [],
        "starting_step": step_number
    }
    return transform_data

def save_transform_json(transform_data, scan_name, output_dir):
    """Save the transform JSON to file."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{scan_name}_transform_log_P3.json")
    with open(json_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    return json_path

def voxel_to_world(voxel_coords, affine):
    """Convert voxel coordinates to world coordinates."""
    voxel_h = np.append(voxel_coords, 1.0)
    world_coords = affine @ voxel_h
    return world_coords[:3]

def world_to_voxel(world_coords, affine):
    """Convert world coordinates to voxel coordinates."""
    # Extract voxel spacing from diagonal of affine matrix
    voxel_spacing = np.array([np.abs(affine[i, i]) for i in range(3)])
    # Subtract origin and divide by voxel spacing
    origin = affine[:3, 3]
    voxel_coords = (world_coords - origin) / voxel_spacing
    return voxel_coords

def crop_roi_around_landmark(scan_data, landmark_world, affine, roi_size, offset_mm=None):
    """
    Crop a cubic ROI centered at a landmark position with optional offset.
    
    Args:
        scan_data: 3D numpy array of the scan
        landmark_world: (3,) array of landmark position in world coordinates (mm)
        affine: 4x4 affine matrix
        roi_size: [x, y, z] size of ROI in voxels
        offset_mm: [x, y, z] offset in mm from landmark position (default: [0, 0, 0])
        
    Returns:
        cropped_data: Cropped ROI
        cropped_affine: Affine matrix for cropped volume
        roi_start_voxel: Starting voxel coordinates of ROI
        roi_end_voxel: Ending voxel coordinates of ROI
        center_world: Actual center position in world coordinates (after offset)
    """
    # Apply offset to landmark position
    if offset_mm is None:
        offset_mm = np.array([0, 0, 0])
    else:
        offset_mm = np.array(offset_mm)
    
    center_world = landmark_world + offset_mm
    print(f"  Offset (mm): {offset_mm}")
    print(f"  Center position (world): {center_world}")
    
    # Convert center from world to voxel coordinates
    landmark_voxel = world_to_voxel(center_world, affine)
    landmark_voxel = np.round(landmark_voxel).astype(int)
    
    print(f"  Landmark voxel coords: {landmark_voxel}")
    print(f"  Scan shape: {scan_data.shape}")
    
    # Calculate ROI bounds
    roi_size = np.array(roi_size)
    half_size = roi_size // 2
    
    roi_start = landmark_voxel - half_size
    roi_end = landmark_voxel + half_size
    
    # Adjust if ROI size is odd to ensure proper centering
    for i in range(3):
        if roi_size[i] % 2 == 1:
            roi_end[i] += 1
    
    print(f"  Initial ROI start: {roi_start}")
    print(f"  Initial ROI end: {roi_end}")
    
    # Initialize ROI with padding value
    roi = np.full(tuple(roi_size), -1000, dtype=scan_data.dtype)
    
    # Calculate valid region (intersection with scan bounds)
    data_shape = np.array(scan_data.shape)
    data_start = np.maximum(roi_start, 0)
    data_end = np.minimum(roi_end, data_shape)
    
    print(f"  Data start (clipped): {data_start}")
    print(f"  Data end (clipped): {data_end}")
    
    # Check if there's any overlap
    if np.any(data_end <= data_start):
        raise ValueError(f"ROI does not overlap with scan data. ROI range: {roi_start} to {roi_end}, Scan shape: {data_shape}")
    
    # Calculate corresponding positions in ROI
    roi_start_in_roi = data_start - roi_start
    roi_end_in_roi = roi_start_in_roi + (data_end - data_start)
    
    print(f"  ROI start in ROI: {roi_start_in_roi}")
    print(f"  ROI end in ROI: {roi_end_in_roi}")
    print(f"  Copy shape: {data_end - data_start}")
    
    # Copy valid data
    data_slices = tuple(slice(int(data_start[i]), int(data_end[i])) for i in range(3))
    roi_slices = tuple(slice(int(roi_start_in_roi[i]), int(roi_end_in_roi[i])) for i in range(3))
    
    print(f"  Data slices: {data_slices}")
    print(f"  ROI slices: {roi_slices}")
    
    roi[roi_slices] = scan_data[data_slices]
    
    # Update affine matrix for cropped volume
    new_origin = voxel_to_world(roi_start.astype(float), affine)
    cropped_affine = affine.copy()
    cropped_affine[:3, 3] = new_origin
    
    return roi, cropped_affine, roi_start, roi_end, center_world


# === Main Processing Pipeline ===
def process_scans():
    """Main function to crop ROIs from aligned scans."""
    
    print(f"\n{'='*60}")
    print(f"P3 ROI CROPPING PIPELINE")
    print(f"{'='*60}")
    print(f"ROI size: {ROI_SIZE} voxels")
    print(f"Landmarks to crop: {[f'LM{lm_id} ({name})' for _, lm_id, name, _, _ in LANDMARKS_TO_CROP]}")
    print(f"{'='*60}\n")
    
    # Create centralized output directory for final cropped ears
    final_cropped_ears_dir = os.path.join(output_dir, "Final_Cropped_Ears_256")
    os.makedirs(final_cropped_ears_dir, exist_ok=True)
    print(f"Final cropped ears will be saved to: {final_cropped_ears_dir}\n")
    
    # Load excluded scans from CSV
    excluded_patient_ids = set()
    if os.path.exists(excluded_scans_csv):
        print(f"Loading excluded scans from: {excluded_scans_csv}")
        try:
            df = pd.read_csv(excluded_scans_csv, sep=';')
            # Filter rows where status is NOK and extract patient_id
            excluded_df = df[df['OK/NOK'].str.strip().str.upper() == 'NOK']
            excluded_patient_ids = set(excluded_df['patient_id'].str.strip().tolist())
            print(f"Loaded {len(excluded_patient_ids)} excluded patient IDs.")
            if excluded_patient_ids:
                print(f"Excluded patients: {sorted(excluded_patient_ids)}")
        except Exception as e:
            print(f"Warning: Could not load excluded scans CSV: {e}")
            print("Proceeding without exclusion filter.")
    else:
        print(f"No excluded scans CSV found at: {excluded_scans_csv}")
        print("Proceeding without exclusion filter.")
    print()
    
    # List to collect transformed landmark positions for CSV export
    landmark_records = []
    
    # Find all scans to process
    # When skip_alignment=True, use the original resampled scans; otherwise use aligned scans
    aligned_scans = []
    if skip_alignment:
        scan_suffix = '_CT_resampled_256.nii.gz'
        for root, dirs, files in os.walk(Processed_scans_dir):
            for file in files:
                if file.endswith(scan_suffix):
                    aligned_scans.append(os.path.join(root, file))
        print(f"skip_alignment=True: using resampled scans (*{scan_suffix})")
    else:
        scan_suffix = '_aligned.nii.gz'
        for root, dirs, files in os.walk(Processed_scans_dir):
            for file in files:
                if file.endswith(scan_suffix):
                    aligned_scans.append(os.path.join(root, file))
        print(f"skip_alignment=False: using aligned scans (*{scan_suffix})")
    
    aligned_scans = sorted(aligned_scans)
    print(f"Found {len(aligned_scans)} scans to process\n")
    
    if len(aligned_scans) == 0:
        print("No scans found. Please run P2 first.")
        return
    
    # Process each scan
    for aligned_scan_path in tqdm(aligned_scans, desc="Processing scans"):
        # Extract scan name
        filename = os.path.basename(aligned_scan_path)
        scan_name = filename.replace(scan_suffix, '')
        
        # Check if scan is in excluded list
        if scan_name in excluded_patient_ids:
            print(f"\n{'='*60}")
            print(f"SKIPPING: {scan_name} (in excluded scans list)")
            print(f"{'='*60}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {scan_name}")
        print(f"{'='*60}")
        
        # Load scan
        print(f"Loading scan from {aligned_scan_path}")
        aligned_img = nib.load(aligned_scan_path)
        aligned_data = aligned_img.get_fdata()
        affine = aligned_img.affine
        
        # Load landmarks
        if skip_alignment:
            # Load predicted landmarks (8-11) from P2 JSON output
            landmark_file = os.path.join(Landmarks_dir, scan_name, "predicted_landmarks.json")
            if not os.path.exists(landmark_file):
                print(f"ERROR: Predicted landmarks not found: {landmark_file}")
                print(f"Skipping {scan_name}")
                continue
            print(f"Loading predicted landmarks from {landmark_file}")
            with open(landmark_file) as f:
                lm_json = json.load(f)
            # Build array ordered by landmark_ids; set missing landmarks (12, 13) to zeros
            aligned_landmarks = np.zeros((len(landmark_ids), 3))
            for i, lm_id in enumerate(landmark_ids):
                if str(lm_id) in lm_json['landmarks']:
                    aligned_landmarks[i] = lm_json['landmarks'][str(lm_id)]['world_mm']
        else:
            # Load aligned landmarks from P2 NPY output
            landmark_file = os.path.join(Aligned_landmarks_dir, f"{scan_name}_lm_aligned.npy")
            if not os.path.exists(landmark_file):
                print(f"ERROR: Aligned landmarks not found: {landmark_file}")
                print(f"Skipping {scan_name}")
                continue
            print(f"Loading aligned landmarks from {landmark_file}")
            aligned_landmarks = np.load(landmark_file)  # Shape: (6, 3)
        
        # Initialize transform logging
        transform_data = initialize_transform_json(
            aligned_scan_path, scan_name, aligned_data.shape, affine, step_number=11
        )
        
        # Create output directory for this scan under Processed_Data_no_alignment
        patient_dir = os.path.join(Processed_data_output_dir, scan_name)
        os.makedirs(patient_dir, exist_ok=True)
        
        # Process each landmark to crop
        for lm_idx, lm_id, lm_name, offset_mm, is_left_ear in LANDMARKS_TO_CROP:
            print(f"\nCropping ROI centered at Landmark {lm_id} ({lm_name})...")
            
            # Get landmark position
            landmark_world = aligned_landmarks[lm_idx]
            print(f"  Landmark position (world): {landmark_world}")
            
            # Apply offset with x-axis flip for right ear
            applied_offset = np.array(offset_mm)
            if not is_left_ear:
                applied_offset[0] = -applied_offset[0]  # Flip x offset for right ear
            
            # Crop ROI
            try:
                cropped_data, cropped_affine, roi_start, roi_end, center_world = crop_roi_around_landmark(
                    aligned_data, landmark_world, affine, ROI_SIZE, offset_mm=applied_offset
                )
                
                print(f"  ROI voxel range: {roi_start} to {roi_end}")
                print(f"  Cropped shape: {cropped_data.shape}")
                
                # Save cropped volume
                cropped_path = os.path.join(patient_dir, f"{scan_name}_{lm_name}_cropped.nii.gz")
                cropped_img = nib.Nifti1Image(cropped_data, cropped_affine)
                nib.save(cropped_img, cropped_path)
                print(f"  ✓ Saved cropped ROI to {cropped_path}")
                
                # Record cropping in transform JSON
                step_idx = 11 + LANDMARKS_TO_CROP.index((lm_idx, lm_id, lm_name, offset_mm, is_left_ear))
                transform_data["transformations"].append({
                    "step": step_idx,
                    "operation": "roi_cropping",
                    "parameters": {
                        "landmark_id": int(lm_id),
                        "landmark_name": lm_name,
                        "landmark_index": int(lm_idx),
                        "is_left_ear": bool(is_left_ear),
                        "landmark_position_world_mm": [float(x) for x in landmark_world],
                        "offset_mm": [float(x) for x in offset_mm],
                        "applied_offset_mm": [float(x) for x in applied_offset],
                        "roi_center_world_mm": [float(x) for x in center_world],
                        "roi_size_voxels": ROI_SIZE,
                        "roi_start_voxel": [int(x) for x in roi_start],
                        "roi_end_voxel": [int(x) for x in roi_end],
                        "cropped_dimensions": [int(x) for x in cropped_data.shape],
                        "cropped_affine_matrix": [[float(x) for x in row] for row in cropped_affine],
                        "padding_value": -1000
                    },
                    "output_file": cropped_path
                })
                
                # === Step: Mirror left ear (lm11) (BEFORE origin reset) ===
                data_to_reset = cropped_data
                affine_to_reset = cropped_affine.copy()
                
                if lm_id == 11:  # Only mirror left ear
                    print(f"  Mirroring left ear along axis {MIRROR_AXIS}...")
                    
                    # Flip the data along axis 0 to match right ear orientation
                    # Keep affine matrix unchanged to change anatomical orientation
                    mirrored_data = np.flip(cropped_data, axis=MIRROR_AXIS)
                    mirrored_affine = cropped_affine.copy()
                    
                    # Save mirrored volume (before origin reset)
                    mirrored_path = os.path.join(patient_dir, f"{scan_name}_{lm_name}_cropped_mirrored.nii.gz")
                    mirrored_img = nib.Nifti1Image(mirrored_data, mirrored_affine)
                    nib.save(mirrored_img, mirrored_path)
                    print(f"  ✓ Saved mirrored volume to {mirrored_path}")
                    
                    # Record mirroring in transform JSON
                    transform_data["transformations"].append({
                        "step": step_idx + 1,
                        "operation": "axis_mirroring",
                        "parameters": {
                            "landmark_id": int(lm_id),
                            "landmark_name": lm_name,
                            "flip_axis": int(MIRROR_AXIS),
                            "flip_method": "numpy.flip",
                            "affine_modification": "none",
                            "mirrored_dimensions": [int(x) for x in mirrored_data.shape]
                        },
                        "output_file": mirrored_path
                    })
                    
                    # Use mirrored data for origin reset
                    data_to_reset = mirrored_data
                    affine_to_reset = mirrored_affine.copy()
                
                # === Step: Reset Origin to (0, 0, 0) ===
                print(f"  Setting origin to (0, 0, 0)...")
                
                # Store original origin
                original_origin = affine_to_reset[:3, 3].copy()
                
                # Create new affine with origin at (0, 0, 0)
                origin_reset_affine = affine_to_reset.copy()
                origin_reset_affine[:3, 3] = [0.0, 0.0, 0.0]
                
                # Save volume with reset origin
                if lm_id == 11:
                    origin_reset_path = os.path.join(patient_dir, f"{scan_name}_{lm_name}_cropped_mirrored_origin_reset.nii.gz")
                else:
                    origin_reset_path = os.path.join(patient_dir, f"{scan_name}_{lm_name}_cropped_origin_reset.nii.gz")
                    
                origin_reset_img = nib.Nifti1Image(data_to_reset, origin_reset_affine)
                nib.save(origin_reset_img, origin_reset_path)
                print(f"  ✓ Saved origin-reset volume to {origin_reset_path}")
                
                # Also save to centralized Final_Cropped_Ears_256 directory with simplified name
                final_filename = f"{scan_name}_{lm_name}.nii.gz"  # e.g., CHUM-001_right_ear.nii.gz
                final_centralized_path = os.path.join(final_cropped_ears_dir, final_filename)
                nib.save(origin_reset_img, final_centralized_path)
                print(f"  ✓ Saved to centralized folder: {final_centralized_path}")
                
                # Record origin reset in transform JSON
                transform_data["transformations"].append({
                    "step": step_idx + 2 if lm_id == 11 else step_idx + 1,
                    "operation": "origin_reset",
                    "parameters": {
                        "landmark_id": int(lm_id),
                        "landmark_name": lm_name,
                        "original_origin_mm": [float(x) for x in original_origin],
                        "new_origin_mm": [0.0, 0.0, 0.0],
                        "affine_matrix": [[float(x) for x in row] for row in origin_reset_affine],
                        "applied_after_mirroring": bool(lm_id == 11)
                    },
                    "output_file": origin_reset_path
                })
                
                # === Compute transformed landmark positions for CSV ===
                voxel_spacing = np.array([np.abs(affine[i, i]) for i in range(3)])
                # Right ear: landmarks 8, 10, 12; Left ear: landmarks 9, 11, 13
                ear_landmark_ids = [8, 10, 12] if lm_id == 10 else [9, 11, 13]
                for li, lid in enumerate(landmark_ids):
                    if lid not in ear_landmark_ids:
                        continue
                    # If no_eyes, set landmarks 12 and 13 to None
                    if no_eyes and lid in [12, 13]:
                        landmark_records.append({
                            "scan_name": scan_name,
                            "ear_side": lm_name,
                            "landmark_id": int(lid),
                            "x_mm": None,
                            "y_mm": None,
                            "z_mm": None
                        })
                        continue
                    lm_world_pos = aligned_landmarks[li]
                    # Voxel coords in the original aligned volume
                    lm_voxel_orig = world_to_voxel(lm_world_pos, affine)
                    # Voxel coords in the cropped volume
                    lm_voxel_cropped = lm_voxel_orig - roi_start.astype(float)
                    # Mirror x-axis for left ear
                    if lm_id == 11:
                        lm_voxel_cropped[MIRROR_AXIS] = (ROI_SIZE[MIRROR_AXIS] - 1) - lm_voxel_cropped[MIRROR_AXIS]
                    # Convert to mm using origin-reset affine (origin=0, so mm = voxel * spacing)
                    lm_mm = lm_voxel_cropped * voxel_spacing
                    # Check if landmark falls inside the ROI
                    inside = all(0 <= lm_voxel_cropped[ax] < ROI_SIZE[ax] for ax in range(3))
                    landmark_records.append({
                        "scan_name": scan_name,
                        "ear_side": lm_name,
                        "landmark_id": int(lid),
                        "x_mm": -float(lm_mm[0]),
                        "y_mm": -float(lm_mm[1]),
                        "z_mm": float(lm_mm[2])
                    })
                
            except Exception as e:
                print(f"  ERROR: Failed to crop ROI for landmark {lm_id}")
                print(f"  Error: {e}")
                
                # Record failure in transform JSON
                transform_data["transformations"].append({
                    "step": 11 + LANDMARKS_TO_CROP.index((lm_idx, lm_id, lm_name, offset_mm, is_left_ear)),
                    "operation": "roi_cropping",
                    "status": "FAILED",
                    "parameters": {
                        "landmark_id": int(lm_id),
                        "landmark_name": lm_name,
                        "offset_mm": [float(x) for x in offset_mm],
                        "applied_offset_mm": [float(x) for x in applied_offset]
                    },
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e)
                    }
                })
        
        # Save transform JSON
        try:
            json_path = save_transform_json(transform_data, scan_name, output_transform_dir)
            print(f"\n  Transform log saved to: {json_path}")
        except Exception as e:
            print(f"\n  WARNING: Failed to save transform JSON: {e}")
        
        print(f"\n✓ Completed {scan_name}")
    
    # Save transformed landmark positions to CSV
    if landmark_records:
        landmarks_csv_path = os.path.join(Aligned_landmarks_dir, "landmark_positions_after_cropping.csv")
        landmarks_df = pd.DataFrame(landmark_records)
        landmarks_df.to_csv(landmarks_csv_path, index=False)
        print(f"\nLandmark positions saved to: {landmarks_csv_path}")
        print(f"Total landmark entries: {len(landmark_records)}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"P3 PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Scans processed: {len(aligned_scans)}")
    print(f"ROIs per scan: {len(LANDMARKS_TO_CROP)}")
    print(f"Total ROIs created: {len(aligned_scans) * len(LANDMARKS_TO_CROP)}")
    print(f"{'='*60}\n")


# === Argument Parser ===
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='P3 Preprocessing: ROI cropping around ear landmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--processed_scans_dir', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Processed-Data",
                        help='Directory containing processed scans')
    parser.add_argument('--output_dir', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Output_no_alignment",
                        help='Directory for outputs (logs, landmarks, cropped ears)')
    parser.add_argument('--excluded_scans_csv', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Excluded_scans_cropping.csv",
                        help='CSV file with excluded scans')
    parser.add_argument('--no_eyes', type=str, default='False',
                        choices=['True', 'False'],
                        help='No eyes mode (must match P2 setting)')
    parser.add_argument('--skip_alignment', type=str, default='False',
                        choices=['True', 'False'],
                        help='Skip alignment mode (must match P2 setting)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with command-line arguments
    Processed_scans_dir = args.processed_scans_dir
    output_dir = args.output_dir
    excluded_scans_csv = args.excluded_scans_csv
    no_eyes = (args.no_eyes == 'True')
    skip_alignment = (args.skip_alignment == 'True')
    
    # Update derived paths
    Processed_data_output_dir = Processed_scans_dir
    output_transform_dir = os.path.join(output_dir, "transform_logs")
    Aligned_landmarks_dir = os.path.join(output_dir, "Aligned_Landmarks")
    Landmarks_dir = os.path.join(output_dir, "Landmarks")
    
    print(f"\n{'='*60}")
    print(f"ROI CROPPING PIPELINE - P3 Final Preprocessing")
    print(f"{'='*60}")
    print(f"Input directory: {Processed_scans_dir}")
    print(f"Aligned landmarks directory: {Aligned_landmarks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    process_scans()

