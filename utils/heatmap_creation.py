

# Precompute heatmaps
import os
import json
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel as nib
import argparse

SIGMA = 3
landmark_ids = [1, 2, 3, 4, 5, 6, 7]  # your landmark IDs

parser = argparse.ArgumentParser(description="Precompute landmark heatmaps from CT and JSON markups.")
parser.add_argument('--nii_dir', type=str, default="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_CTs_resampled_128", help='Directory with input NIfTI files')
parser.add_argument('--json_dir', type=str, default="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_Markups_Corrected", help='Directory with input JSON markup files')
parser.add_argument('--heatmap_dir', type=str, default="/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_Precomputed_Heatmaps_resampled_128_Corrected", help='Output directory for heatmaps')
args = parser.parse_args()

nii_dir = args.nii_dir
json_dir = args.json_dir
heatmap_dir = args.heatmap_dir
os.makedirs(heatmap_dir, exist_ok=True)

for nii_file in sorted(os.listdir(nii_dir)):
    if not nii_file.endswith('.nii.gz'):
        continue

    heatmap_path = os.path.join(heatmap_dir, nii_file.replace('.nii.gz', '_heatmaps.pt'))
    if os.path.exists(heatmap_path):
        print(f"Skipping {nii_file} - heatmap already exists")
        continue

    json_path = os.path.join(json_dir, nii_file.replace('.nii.gz', '.mrk.json'))
    if not os.path.exists(json_path):
        continue
    

    img = nib.load(os.path.join(nii_dir, nii_file))
    orig_affine = img.affine.copy()
    orig_shape = img.shape

    with open(json_path, 'r') as f:
        data = json.load(f)

    heatmaps = torch.zeros((len(landmark_ids), *orig_shape), dtype=torch.float32)

    # Access controlPoints from markups array
    control_points = data['markups'][0]['controlPoints']
    
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
