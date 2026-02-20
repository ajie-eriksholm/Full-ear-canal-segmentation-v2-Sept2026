import os
import numpy as np
import nibabel as nib
import nrrd
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BF_32_RUN = "run_20260210_102624"
BF_48_RUN = "run_20260210_103433"
BF_64_RUN = "run_20260210_104123"
BF_32_ED_RUN = "run_20260210_104800"
BF_48_ED_RUN = "run_20260210_105409"

TEST_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Test Ears-Thin Canal"
MODEL_PATH_TEMPLATE = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Logs/{}/best_model.pth"
OUTPUT_PREDICTIONS_DIR_TEMPLATE = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Logs/{}/test_predictions"

NUM_LANDMARKS = 7  # Number of heatmap channels
LANDMARK_IDS = [1, 2, 3, 4, 5, 6, 7]  # Landmark IDs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations: (run_name, architecture_type, base_features, seg_channels)
MODEL_CONFIGS = [
    (BF_32_RUN, "late_divergence", 32, 2),
    (BF_48_RUN, "late_divergence", 48, 2),
    (BF_64_RUN, "late_divergence", 64, 2),
    (BF_32_ED_RUN, "early_divergence", 32, 2),
    (BF_48_ED_RUN, "early_divergence", 48, 2),
]

# ============================================================
# MODEL (3D UNET)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3D_LateDivergence(nn.Module):
    """UNet3D with late divergence (at dec2 level)"""
    def __init__(self, in_channels=1, seg_channels=2, heatmap_channels=7, base_features=32):
        super().__init__()
        f = base_features
        # Shared Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.enc5 = DoubleConv(f * 8, f * 16)

        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # Shared decoder (high-level features)
        self.dec4 = DoubleConv(f * 16 + f * 8, f * 8)
        self.dec3 = DoubleConv(f * 8 + f * 4, f * 4)
        
        # Task-specific decoders (diverge at dec2 level)
        # Segmentation branch
        self.seg_dec2 = DoubleConv(f * 4 + f * 2, f * 2)
        self.seg_dec1 = DoubleConv(f * 2 + f, f)
        self.seg_conv = nn.Conv3d(f, seg_channels, kernel_size=1)
        
        # Heatmap branch
        self.heat_dec2 = DoubleConv(f * 4 + f * 2, f * 2)
        self.heat_dec1 = DoubleConv(f * 2 + f, f)
        self.heatmap_conv = nn.Conv3d(f, heatmap_channels, kernel_size=1)

    def forward(self, x):
        # Shared encoder
        e1 = self.enc1(x)              # B, f, ...
        e2 = self.enc2(self.pool(e1))  # B, f*2, ...
        e3 = self.enc3(self.pool(e2))  # B, f*4, ...
        e4 = self.enc4(self.pool(e3))  # B, f*8, ...
        e5 = self.enc5(self.pool(e4))  # B, f*16, ... (bottleneck)

        # Shared decoder (high-level)
        d4 = self.up(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        # Task-specific branches (diverge here at dec2 level)
        # Segmentation path
        seg_d2 = self.up(d3)
        seg_d2 = torch.cat([seg_d2, e2], dim=1)
        seg_d2 = self.seg_dec2(seg_d2)
        
        seg_d1 = self.up(seg_d2)
        seg_d1 = torch.cat([seg_d1, e1], dim=1)
        seg_d1 = self.seg_dec1(seg_d1)
        seg_out = self.seg_conv(seg_d1)
        
        # Heatmap path
        heat_d2 = self.up(d3)
        heat_d2 = torch.cat([heat_d2, e2], dim=1)
        heat_d2 = self.heat_dec2(heat_d2)
        
        heat_d1 = self.up(heat_d2)
        heat_d1 = torch.cat([heat_d1, e1], dim=1)
        heat_d1 = self.heat_dec1(heat_d1)
        heatmap_out = self.heatmap_conv(heat_d1)

        return seg_out, heatmap_out


class UNet3D_EarlyDivergence(nn.Module):
    """UNet3D with early divergence (at dec3 level)"""
    def __init__(self, in_channels=1, seg_channels=2, heatmap_channels=7, base_features=48):
        super().__init__()
        f = base_features
        # Shared Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.enc5 = DoubleConv(f * 8, f * 16)

        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # Shared decoder (high-level features)
        self.dec4 = DoubleConv(f * 16 + f * 8, f * 8)
        
        # Task-specific decoders (diverge at dec3 level for more task-specific capacity)
        # Segmentation branch
        self.seg_dec3 = DoubleConv(f * 8 + f * 4, f * 4)
        self.seg_dec2 = DoubleConv(f * 4 + f * 2, f * 2)
        self.seg_dec1 = DoubleConv(f * 2 + f, f)
        self.seg_conv = nn.Conv3d(f, seg_channels, kernel_size=1)
        
        # Heatmap branch
        self.heat_dec3 = DoubleConv(f * 8 + f * 4, f * 4)
        self.heat_dec2 = DoubleConv(f * 4 + f * 2, f * 2)
        self.heat_dec1 = DoubleConv(f * 2 + f, f)
        self.heatmap_conv = nn.Conv3d(f, heatmap_channels, kernel_size=1)

    def forward(self, x):
        # Shared encoder
        e1 = self.enc1(x)              # B, f, ...
        e2 = self.enc2(self.pool(e1))  # B, f*2, ...
        e3 = self.enc3(self.pool(e2))  # B, f*4, ...
        e4 = self.enc4(self.pool(e3))  # B, f*8, ...
        e5 = self.enc5(self.pool(e4))  # B, f*16, ... (bottleneck)

        # Shared decoder (high-level)
        d4 = self.up(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        # Task-specific branches (diverge here at dec3 level)
        # Segmentation path
        seg_d3 = self.up(d4)
        seg_d3 = torch.cat([seg_d3, e3], dim=1)
        seg_d3 = self.seg_dec3(seg_d3)
        
        seg_d2 = self.up(seg_d3)
        seg_d2 = torch.cat([seg_d2, e2], dim=1)
        seg_d2 = self.seg_dec2(seg_d2)
        
        seg_d1 = self.up(seg_d2)
        seg_d1 = torch.cat([seg_d1, e1], dim=1)
        seg_d1 = self.seg_dec1(seg_d1)
        seg_out = self.seg_conv(seg_d1)
        
        # Heatmap path
        heat_d3 = self.up(d4)
        heat_d3 = torch.cat([heat_d3, e3], dim=1)
        heat_d3 = self.heat_dec3(heat_d3)
        
        heat_d2 = self.up(heat_d3)
        heat_d2 = torch.cat([heat_d2, e2], dim=1)
        heat_d2 = self.heat_dec2(heat_d2)
        
        heat_d1 = self.up(heat_d2)
        heat_d1 = torch.cat([heat_d1, e1], dim=1)
        heat_d1 = self.heat_dec1(heat_d1)
        heatmap_out = self.heatmap_conv(heat_d1)

        return seg_out, heatmap_out


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_ct(ct_path):
    """Load and preprocess a CT scan (same as training)."""
    # Load CT
    img = nib.load(ct_path)
    image = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()
    
    # Debug: Check original range
    print(f"  Original image range: [{image.min():.2f}, {image.max():.2f}]")
    
    # Normalize CT scan to [0, 1] range
    # Min value: -1000 HU -> 0
    # Max value: 2007 HU -> 1
    image = np.clip(image, -1000, 2007)  # Clip to expected HU range
    image = (image + 1000) / (2007 + 1000)  # Normalize to [0, 1]
    
    # Debug: Check normalized range
    print(f"  Normalized image range: [{image.min():.4f}, {image.max():.4f}]")
    
    # Convert to tensor and add batch + channel dimensions
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    return image_tensor, affine, img.shape, image


# ============================================================
# LANDMARK EXTRACTION
# ============================================================
def extract_landmark_centroids(heatmaps, affine, landmark_ids, threshold=0.5):
    """
    Extract landmark centroids from predicted heatmaps.
    heatmaps: (num_landmarks, D, H, W) numpy array
    affine: (4,4) numpy array (voxel->world)
    landmark_ids: list of landmark IDs
    threshold: threshold for weighted centroid computation
    Returns: array of shape (num_landmarks, 3) with world coordinates in mm
    """
    num = len(landmark_ids)
    coords_mm = np.zeros((num, 3), dtype=float)
    
    for i, lm_id in enumerate(landmark_ids):
        heatmap = heatmaps[i]
        mask = heatmap >= threshold
        
        if np.any(mask):
            # Compute weighted centroid
            coords = np.argwhere(mask)  # N x 3 (z,y,x)
            weights = heatmap[mask]
            centroid = np.average(coords, axis=0, weights=weights)
            rounded_centroid = np.round(centroid).astype(int)
        else:
            # Fallback to global argmax
            flat_idx = np.argmax(heatmap)
            rounded_centroid = np.array(np.unravel_index(flat_idx, heatmap.shape))
        
        # Convert voxel indices (z,y,x) to world coordinates (x,y,z) in mm
        voxel_coord = np.append(rounded_centroid[::-1], 1)  # convert (z,y,x) -> (x,y,z,1)
        world_coord = affine @ voxel_coord
        coords_mm[i] = world_coord[:3]
    
    return coords_mm


# ============================================================
# INFERENCE
# ============================================================
def run_inference_single_model(run_name, architecture_type, base_features, seg_channels):
    """Run inference on all test CT scans for a single model."""
    model_path = MODEL_PATH_TEMPLATE.format(run_name)
    output_dir = OUTPUT_PREDICTIONS_DIR_TEMPLATE.format(run_name)
    
    print("="*80)
    print(f"RUNNING INFERENCE: {run_name}")
    print("="*80)
    print(f"Architecture: {architecture_type}")
    print(f"Base features: {base_features}")
    print(f"Segmentation channels: {seg_channels}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    if architecture_type == "late_divergence":
        model = UNet3D_LateDivergence(
            in_channels=1, 
            seg_channels=seg_channels, 
            heatmap_channels=NUM_LANDMARKS, 
            base_features=base_features
        )
    else:  # early_divergence
        model = UNet3D_EarlyDivergence(
            in_channels=1, 
            seg_channels=seg_channels, 
            heatmap_channels=NUM_LANDMARKS, 
            base_features=base_features
        )
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"✓ Model loaded successfully (epoch {checkpoint['epoch']})\n")
    
    # Get all test CT files
    ct_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.nii.gz') or f.endswith('.nii')])
    print(f"Found {len(ct_files)} test CT scans\n")
    
    if len(ct_files) == 0:
        print("No CT files found in test directory!")
        return
    
    # Process each CT scan
    successful = 0
    failed = 0
    
    # Storage for landmark coordinates CSV
    landmark_data = []
    
    with torch.no_grad():
        for ct_file in tqdm(ct_files, desc="Processing test scans"):
            try:
                ct_path = os.path.join(TEST_DIR, ct_file)
                
                # Preprocess CT
                image_tensor, affine, original_shape, image_normalized = preprocess_ct(ct_path)
                image_tensor = image_tensor.to(DEVICE)
                
                # Verify tensor range before inference
                print(f"  Tensor range before inference: [{image_tensor.min().item():.4f}, {image_tensor.max().item():.4f}]")
                
                # Run inference
                seg_logits, heatmap_logits = model(image_tensor)
                
                # Apply sigmoid to get probabilities
                seg_pred = torch.sigmoid(seg_logits)
                heatmap_pred = torch.sigmoid(heatmap_logits)
                
                # Convert to numpy and remove batch dimension
                seg_pred_np = seg_pred[0].cpu().numpy()  # (seg_channels, D, H, W)
                heatmap_pred_np = heatmap_pred[0].cpu().numpy()  # (NUM_LANDMARKS, D, H, W)
                
                # Convert multi-channel segmentation to single label map using argmax
                seg_pred_labels = np.argmax(seg_pred_np, axis=0).astype(np.uint8)  # (D, H, W)
                
                # Save segmentation mask as NIfTI (preserves exact spatial alignment)
                # Remove extension and add new suffix
                if ct_file.endswith('.nii.gz'):
                    seg_output_name = ct_file[:-7] + '_pred_seg.nii.gz'
                else:
                    seg_output_name = ct_file[:-4] + '_pred_seg.nii.gz'
                seg_output_path = os.path.join(output_dir, seg_output_name)
                
                # Create NIfTI image with same affine as input CT
                seg_nifti = nib.Nifti1Image(seg_pred_labels, affine)
                nib.save(seg_nifti, seg_output_path)
                
                # Save heatmap predictions as PyTorch tensor
                if ct_file.endswith('.nii.gz'):
                    heatmap_output_name = ct_file[:-7] + '_pred_heatmaps.pt'
                else:
                    heatmap_output_name = ct_file[:-4] + '_pred_heatmaps.pt'
                heatmap_output_path = os.path.join(output_dir, heatmap_output_name)
                torch.save(torch.from_numpy(heatmap_pred_np), heatmap_output_path)
                
                # Save normalized input CT scan to output directory (the one used for inference)
                if ct_file.endswith('.nii.gz'):
                    input_ct_output_name = ct_file[:-7] + '_normalized.nii.gz'
                else:
                    input_ct_output_name = ct_file[:-4] + '_normalized.nii.gz'
                input_ct_output_path = os.path.join(output_dir, input_ct_output_name)
                normalized_ct_nifti = nib.Nifti1Image(image_normalized, affine)
                nib.save(normalized_ct_nifti, input_ct_output_path)
                
                # Extract landmark centroids
                landmark_coords = extract_landmark_centroids(heatmap_pred_np, affine, LANDMARK_IDS)
                
                # Store landmark data for CSV
                for i, lm_id in enumerate(LANDMARK_IDS):
                    landmark_data.append({
                        'scan_name': ct_file,
                        'landmark_id': lm_id,
                        'x_mm': -landmark_coords[i, 2],
                        'y_mm': landmark_coords[i, 1],
                        'z_mm': -landmark_coords[i, 0]
                    })
                
                successful += 1
                
            except Exception as e:
                failed += 1
                print(f"\n✗ Error processing {ct_file}: {str(e)}")
    
    # Save landmark coordinates to CSV
    if landmark_data:
        csv_path = os.path.join(output_dir, "predicted_landmark_coordinates.csv")
        df = pd.DataFrame(landmark_data)
        try:
            df.to_csv(csv_path, index=False)
            print(f"\n✓ Landmark coordinates saved to: {csv_path}")
        except PermissionError:
            # Fallback to home directory if permission denied
            fallback_path = os.path.join(os.path.expanduser("~"), "Documents", f"predicted_landmark_coordinates_{run_name}.csv")
            df.to_csv(fallback_path, index=False)
            print(f"\n⚠ Permission denied for {csv_path}")
            print(f"✓ Landmark coordinates saved to: {fallback_path}")
    
    # Summary
    print("\n" + "="*80)
    print(f"INFERENCE SUMMARY - {run_name}")
    print("="*80)
    print(f"Total scans: {len(ct_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print("\nPredictions saved:")
    print(f"  - Normalized CT scans (used for inference): *_normalized.nii.gz")
    print(f"  - Segmentation masks (multiclass): *_pred_seg.nii.gz")
    print(f"  - Landmark heatmaps: *_pred_heatmaps.pt")
    print(f"  - Landmark coordinates: predicted_landmark_coordinates.csv")
    print("="*80 + "\n\n")
    
    return successful, failed


def run_inference():
    """Run inference on all test CT scans using all 5 models."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*20 + "MULTI-MODEL INFERENCE PIPELINE" + " "*27 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    print(f"\nTest directory: {TEST_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Number of models: {len(MODEL_CONFIGS)}")
    print("\nModel configurations:")
    for i, (run_name, arch_type, base_feat, seg_ch) in enumerate(MODEL_CONFIGS, 1):
        print(f"  {i}. {run_name[:20]:20s} | {arch_type:17s} | BF={base_feat:2d} | seg_ch={seg_ch}")
    print("\n" + "#"*80 + "\n")
    
    # Track overall results
    overall_results = []
    
    # Run inference for each model
    for run_name, architecture_type, base_features, seg_channels in MODEL_CONFIGS:
        try:
            successful, failed = run_inference_single_model(
                run_name, architecture_type, base_features, seg_channels
            )
            overall_results.append({
                'run_name': run_name,
                'architecture': architecture_type,
                'base_features': base_features,
                'successful': successful,
                'failed': failed
            })
        except Exception as e:
            print(f"\n✗ FAILED to run inference for {run_name}: {str(e)}\n")
            overall_results.append({
                'run_name': run_name,
                'architecture': architecture_type,
                'base_features': base_features,
                'successful': 0,
                'failed': 'ERROR'
            })
    
    # Print final summary
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + " "*25 + "FINAL SUMMARY" + " "*40 + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    print("\n{:30s} {:17s} {:4s} {:10s} {:10s}".format(
        "Run Name", "Architecture", "BF", "Successful", "Failed"
    ))
    print("-"*80)
    for result in overall_results:
        print("{:30s} {:17s} {:4d} {:10s} {:10s}".format(
            result['run_name'][:30],
            result['architecture'],
            result['base_features'],
            str(result['successful']),
            str(result['failed'])
        ))
    print("#"*80 + "\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run_inference()
