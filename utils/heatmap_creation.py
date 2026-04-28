

# Precompute heatmaps
import os
import json
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel as nib
import argparse

SIGMA = 3
landmark_ids = [8, 9, 10, 11, 12, 13]  # your landmark IDs


parser = argparse.ArgumentParser(description="Precompute landmark heatmaps from CT and JSON markups.")
parser.add_argument('--nii_dir', type=str, default="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_CTs_resampled_128", help='Directory with input NIfTI files')
parser.add_argument('--json_dir', type=str, default="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_Markups_Corrected", help='Directory with input JSON markup files')
parser.add_argument('--heatmap_dir', type=str, default="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_Precomputed_Heatmaps_resampled_128_Corrected", help='Output directory for heatmaps')
parser.add_argument('--fh_alignment', action='store_true', help='Enable folder-based logic for FH alignment retraining')
args = parser.parse_args()

nii_dir = args.nii_dir
json_dir = args.json_dir
heatmap_dir = args.heatmap_dir
os.makedirs(heatmap_dir, exist_ok=True)

if args.fh_alignment:
    # Folder-based logic for FH alignment retraining
    for patient_id in sorted(os.listdir(nii_dir)):
        patient_folder = os.path.join(nii_dir, patient_id)
        if not os.path.isdir(patient_folder):
            continue
        nii_file = f"{patient_id}_CT_resampled_256.nii.gz"
        nii_path = os.path.join(patient_folder, nii_file)
        if not os.path.exists(nii_path):
            print(f"[WARN] NIfTI not found for patient {patient_id}: {nii_path}")
            continue
        json_path = os.path.join(json_dir, f"{patient_id}.json")
        heatmap_path = os.path.join(heatmap_dir, f"{patient_id}_heatmaps.pt")
        

        if os.path.exists(heatmap_path):
            print(f"Skipping {nii_file} - heatmap already exists")
            continue

        if not os.path.exists(json_path):
            print(f"[WARN] JSON markup not found for {nii_file}: {json_path}")
            continue

        try:
            img = nib.load(nii_path)
            orig_affine = img.affine.copy()
            orig_shape = img.shape
        except Exception as e:
            print(f"[ERROR] Failed to load NIfTI file {nii_file}: {e}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON file {json_path}: {e}")
            continue

        heatmaps = torch.zeros((len(landmark_ids), *orig_shape), dtype=torch.float32)

        # Access landmarks array for FH alignment
        try:
            landmarks = data['landmarks']
        except Exception as e:
            print(f"[ERROR] Failed to access 'landmarks' in {json_path}: {e}")
            continue

        for i, lm_id in enumerate(landmark_ids):
            # Convert lm_id to int for comparison with JSON id field
            pos_world = next((lm['position'] for lm in landmarks if int(lm['id']) == lm_id), None)
            if pos_world:
                pos_ras = np.array([-pos_world[0], -pos_world[1], pos_world[2]])
                voxel_pos = np.linalg.inv(orig_affine) @ np.append(pos_ras, 1)
                voxel_pos = voxel_pos[:3]
                heatmap = np.zeros(orig_shape, dtype=np.float32)
                voxel_idx = tuple(np.round(voxel_pos).astype(int))
                if all(0 <= voxel_idx[d] < orig_shape[d] for d in range(3)):
                    heatmap[voxel_idx] = 1.0
                    heatmap = gaussian_filter(heatmap, sigma=(SIGMA, SIGMA, SIGMA))
                    heatmap /= heatmap.max()
                    heatmaps[i] = torch.from_numpy(heatmap)
        torch.save(heatmaps, heatmap_path)
else:
    # Original logic (flat nii_dir)
    for nii_file in sorted(os.listdir(nii_dir)):
        if not nii_file.endswith('.nii.gz'):
            continue

        print(f"[DEBUG] Processing NIfTI file: {nii_file}")
        nii_base = nii_file.replace('.nii.gz', '')
        if '__CT' in nii_base:
            patient_id = nii_base.split('__CT')[0]
        else:
            patient_id = nii_base  # fallback if no __CT

        json_path = os.path.join(json_dir, f"{patient_id}.json")
        heatmap_path = os.path.join(heatmap_dir, f"{patient_id}_heatmaps.pt")
        

        if os.path.exists(heatmap_path):
            print(f"Skipping {nii_file} - heatmap already exists")
            continue

        if not os.path.exists(json_path):
            print(f"[WARN] JSON markup not found for {nii_file}: {json_path}")
            continue

        try:
            img = nib.load(os.path.join(nii_dir, nii_file))
            orig_affine = img.affine.copy()
            orig_shape = img.shape
        except Exception as e:
            print(f"[ERROR] Failed to load NIfTI file {nii_file}: {e}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON file {json_path}: {e}")
            continue

        heatmaps = torch.zeros((len(landmark_ids), *orig_shape), dtype=torch.float32)

        # Access controlPoints from markups array
        try:
            control_points = data['markups'][0]['controlPoints']
        except Exception as e:
            print(f"[ERROR] Failed to access controlPoints in {json_path}: {e}")
            continue

        for i, lm_id in enumerate(landmark_ids):
            # Convert lm_id to string for comparison with JSON id field
            pos_world = next((lm['position'] for lm in control_points if lm['id'] == str(lm_id)), None)
            if pos_world:
                # JSON coordinate system is LPS, convert to RAS for NIfTI
                pos_ras = np.array([-pos_world[0], -pos_world[1], pos_world[2]])
                voxel_pos = np.linalg.inv(orig_affine) @ np.append(pos_ras, 1)
                voxel_pos = voxel_pos[:3]

                heatmap = np.zeros(orig_shape, dtype=np.float32)
                voxel_idx = tuple(np.round(voxel_pos).astype(int))

                if all(0 <= voxel_idx[d] < orig_shape[d] for d in range(3)):
                    heatmap[voxel_idx] = 1.0
                    heatmap = gaussian_filter(heatmap, sigma=(SIGMA, SIGMA, SIGMA))
                    heatmap /= heatmap.max()
                    heatmaps[i] = torch.from_numpy(heatmap)

        torch.save(heatmaps, os.path.join(heatmap_dir, nii_file.replace('.nii.gz', '_heatmaps.pt')))
