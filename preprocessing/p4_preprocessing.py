import os
import numpy as np
import nibabel as nib
import argparse
from scipy import ndimage
from pathlib import Path
from tqdm import tqdm

# upsampling and normalization 

# Default configuration (can be overridden by command-line arguments)
PROCESS_DATA_DIR = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Processed-Data"
OUTPUT_DATA_DIR = r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Output/Preprocessing/P4_Normalized_Ears"

# Target resolution
TARGET_SHAPE = (128, 128, 128)

# Normalization parameters (HU units)
MIN_HU = -1000
MAX_HU = 2007


def upsample_to_128(image_data, original_shape):
    """
    Upsample image to 128x128x128 using trilinear interpolation.
    
    Args:
        image_data: 3D numpy array
        original_shape: original shape tuple
    
    Returns:
        Upsampled 3D numpy array of shape (128, 128, 128)
    """
    # Calculate zoom factors for each dimension
    zoom_factors = [TARGET_SHAPE[i] / original_shape[i] for i in range(3)]
    
    # Perform trilinear interpolation
    upsampled = ndimage.zoom(image_data, zoom_factors, order=1, mode='nearest')
    
    return upsampled


def normalize_scan(image_data, min_hu=MIN_HU, max_hu=MAX_HU):
    """
    Normalize CT scan to [0, 1] range.
    
    Args:
        image_data: 3D numpy array in HU units
        min_hu: minimum HU value (default: -1000)
        max_hu: maximum HU value (default: 2007)
    
    Returns:
        Normalized 3D numpy array in [0, 1] range
    """
    # Clip to expected HU range
    clipped = np.clip(image_data, min_hu, max_hu)
    
    # Normalize to [0, 1]
    normalized = (clipped - min_hu) / (max_hu - min_hu)
    
    return normalized.astype(np.float32)


def process_scan(input_path, output_path, patient_id, ear_side):
    """
    Process a single scan: upsample and normalize.
    
    Args:
        input_path: path to input NIfTI file
        output_path: path to save output NIfTI file
        patient_id: patient identifier
        ear_side: 'left' or 'right'
    """
    try:
        # Load the scan
        img = nib.load(input_path)
        image_data = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()
        original_shape = image_data.shape
        
        print(f"  Processing {patient_id}_{ear_side}_ear...")
        print(f"    Original shape: {original_shape}")
        print(f"    Original range: [{image_data.min():.2f}, {image_data.max():.2f}]")
        
        # Upsample to 128^3
        upsampled = upsample_to_128(image_data, original_shape)
        print(f"    Upsampled shape: {upsampled.shape}")
        
        # Normalize to [0, 1]
        normalized = normalize_scan(upsampled)
        print(f"    Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
        
        # Create new affine matrix for 128^3 volume
        # Adjust voxel spacing to maintain physical dimensions
        new_affine = affine.copy()
        for i in range(3):
            scale_factor = original_shape[i] / TARGET_SHAPE[i]
            new_affine[i, :3] *= scale_factor
        
        # Save the processed scan with nnU-Net naming convention (_0000 suffix)
        output_filename = f"{patient_id}_{ear_side}_0000.nii.gz"
        output_filepath = os.path.join(output_path, output_filename)
        
        new_img = nib.Nifti1Image(normalized, new_affine)
        nib.save(new_img, output_filepath)
        
        print(f"    ✓ Saved to: {output_filename}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing {patient_id}_{ear_side}_ear: {str(e)}")
        return False


def process_all_participants():
    """
    Process all participants in the processed data directory.
    """
    print("="*80)
    print("PREPROCESSING SCANS FOR INFERENCE")
    print("="*80)
    print(f"\nInput directory: {PROCESS_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DATA_DIR}")
    print(f"Target resolution: {TARGET_SHAPE}")
    print(f"Normalization range: [{MIN_HU}, {MAX_HU}] HU -> [0, 1]")
    print("="*80 + "\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    
    # Centralized cropped ears directory from P3
    cropped_ears_dir = os.path.join(os.path.dirname(OUTPUT_DATA_DIR), "P3_Cropped_Ears")
    
    # Get all participant folders
    participant_folders = [f for f in os.listdir(PROCESS_DATA_DIR) 
                          if os.path.isdir(os.path.join(PROCESS_DATA_DIR, f))]
    
    print(f"Found {len(participant_folders)} participant folders")
    print(f"Centralized cropped ears dir: {cropped_ears_dir}\n")
    
    # Track statistics
    total_processed = 0
    total_failed = 0
    
    # Process each participant
    for participant_folder in tqdm(sorted(participant_folders), desc="Processing participants"):
        participant_path = os.path.join(PROCESS_DATA_DIR, participant_folder)
        
        # Extract patient ID (folder name)
        patient_id = participant_folder
        
        print(f"\n{'='*80}")
        print(f"Participant: {patient_id}")
        print(f"{'='*80}")
        
        # Look for left and right ear scans
        # Try per-patient dir first (long names), then centralized dir (short names)
        left_ear_candidates = [
            os.path.join(participant_path, f"{patient_id}_left_ear_cropped_mirrored_origin_reset.nii.gz"),
            os.path.join(cropped_ears_dir, f"{patient_id}_left_ear.nii.gz"),
        ]
        right_ear_candidates = [
            os.path.join(participant_path, f"{patient_id}_right_ear_cropped_origin_reset.nii.gz"),
            os.path.join(cropped_ears_dir, f"{patient_id}_right_ear.nii.gz"),
        ]
        
        left_ear_path = next((p for p in left_ear_candidates if os.path.exists(p)), None)
        right_ear_path = next((p for p in right_ear_candidates if os.path.exists(p)), None)
        
        # Process left ear
        if left_ear_path:
            success = process_scan(left_ear_path, OUTPUT_DATA_DIR, patient_id, 'left')
            if success:
                total_processed += 1
            else:
                total_failed += 1
        else:
            print(f"  Warning: Left ear scan not found for {patient_id}")
        
        # Process right ear
        if right_ear_path:
            success = process_scan(right_ear_path, OUTPUT_DATA_DIR, patient_id, 'right')
            if success:
                total_processed += 1
            else:
                total_failed += 1
        else:
            print(f"  Warning: Right ear scan not found for {patient_id}")
    
    # Final summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Total scans processed successfully: {total_processed}")
    print(f"Total scans failed: {total_failed}")
    print(f"Output directory: {OUTPUT_DATA_DIR}")
    print("="*80 + "\n")


# === Argument Parser ===
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='P4 Preprocessing: Upsampling and normalization for inference',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--processed_scans_dir', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Processed-Data",
                        help='Directory containing processed scans')
    parser.add_argument('--output_dir', type=str,
                        default=r"/projects/oticon/erhdata/Processed-Data/SBEO/High-quality-scans/Output",
                        help='Base output directory')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with command-line arguments
    PROCESS_DATA_DIR = args.processed_scans_dir
    OUTPUT_DATA_DIR = os.path.join(args.output_dir, "Preprocessing", "P4_Normalized_Ears")
    
    process_all_participants()