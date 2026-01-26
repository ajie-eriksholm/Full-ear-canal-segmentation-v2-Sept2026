import os
import time
import random
import numpy as np
import pandas as pd
import nibabel as nib
import nrrd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F


from monai.transforms import (
    Compose,
    RandAffined
)

# ============================================================
# CONFIGURATION
# ============================================================
BATCH_SIZE = 1
NUM_WORKERS = 8
LR = 1e-4
MAX_EPOCHS = 500
SEED = 12345
TRAIN_RATIO = 0.85
NUM_LANDMARKS = 7  # Number of heatmap channels to predict
LANDMARK_IDS = [1, 2, 3, 4, 5, 6, 7]  # Landmark IDs for tracking
# Visualization slice for each landmark (None = middle slice, or specify integer)
HEATMAP_VIZ_SLICE = {
    1: 53,  # Landmark 1: middle slice
    2: 55,  # Landmark 2: middle slice
    3: 52,  # Landmark 3: middle slice
    4: 64,  # Landmark 4: middle slice
    5: 63,  # Landmark 5: middle slice
    6: 57,  # Landmark 6: middle slice
    7: 51   # Landmark 7: middle slice
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATHS (YOUR DIRECTORIES)
# ============================================================
CT_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_CTs_resampled_128"
SEG_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_Masks_resampled_128"
HEATMAPS_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_Precomputed_Heatmaps_resampled_128_Corrected"

BASE_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline"
SPLIT_CSV = os.path.join(BASE_DIR, "train_val_split.csv")
LOG_DIR = os.path.join(BASE_DIR, "Logs")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# TRAIN / VAL SPLIT (REPRODUCIBLE + CSV)
# ============================================================
def create_or_load_split(ct_dir, csv_path, train_ratio=0.85):
    if os.path.exists(csv_path):
        print(f"Loading existing split from {csv_path}")
        return pd.read_csv(csv_path)

    scans = sorted([f for f in os.listdir(ct_dir) if f.endswith(".nii.gz")])

    train_files, val_files = train_test_split(
        scans,
        train_size=train_ratio,
        random_state=SEED,
        shuffle=True
    )

    df = pd.DataFrame({
        "scan_name": train_files + val_files,
        "split": ["train"] * len(train_files) + ["val"] * len(val_files)
    })

    df.to_csv(csv_path, index=False)
    print(f"Saved split CSV to {csv_path}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")

    return df

# ============================================================
# DATASET
# ============================================================
class SegmentationDataset(Dataset):
    def __init__(self, ct_dir, seg_dir, heatmap_dir, file_list, augment=False):
        self.ct_dir = ct_dir
        self.seg_dir = seg_dir
        self.heatmap_dir = heatmap_dir
        self.file_list = file_list
        self.augment = augment
        
        # Cache for storing original NRRD headers (for saving predictions with correct spatial info)
        self.nrrd_headers = {}

        if augment:
            self.transform = Compose([
                RandAffined(
                    keys=["image", "mask", "heatmap"],
                    prob=0.8,
                    rotate_range=(0.35, 0.35, 0.35),
                    translate_range=(10, 10, 10),
                    mode=["bilinear", "nearest", "bilinear"],
                    padding_mode="zeros"
                )
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        # ---------- Load CT ----------
        ct_path = os.path.join(self.ct_dir, fname)
        img = nib.load(ct_path)
        image = img.get_fdata(dtype=np.float32)
        affine = img.affine.copy()

        # HU normalization
        image = np.clip(image, -1000, 2007)
        image = (image + 1000) / (2007 + 1000)
        image = torch.from_numpy(image).unsqueeze(0)

        # ---------- Load Mask ----------
        seg_name = fname.replace(".nii.gz", ".nrrd")
        seg_path = os.path.join(self.seg_dir, seg_name)

        mask, header = nrrd.read(seg_path)
        
        # Store original header for later use when saving predictions
        # This preserves space directions, space origin, and other spatial metadata
        self.nrrd_headers[fname] = header
        
        # NRRD and NIfTI may have different axis ordering
        # NRRD is typically (x,y,z) while our CT from NIfTI might be (x,y,z) or different
        # Check if transpose is needed to match CT dimensions
        if mask.shape != image.shape[1:]:  # Compare with image shape (without channel dim)
            print(f"[WARNING] Mask shape {mask.shape} doesn't match CT shape {image.shape[1:]}")
            # Try transposing to match
            if mask.T.shape == image.shape[1:]:
                mask = mask.T
                print(f"[INFO] Transposed mask to {mask.shape}")
        
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # ---------- Load Heatmaps ----------
        heatmap_name = fname.replace(".nii.gz", "_heatmaps.pt")
        heatmap_path = os.path.join(self.heatmap_dir, heatmap_name)
        heatmaps = torch.load(heatmap_path, map_location='cpu').float()

        # ---------- Augmentation ----------
        if self.transform is not None:
            data = {"image": image, "mask": mask, "heatmap": heatmaps}
            data = self.transform(data)
            image, mask, heatmaps = data["image"], data["mask"], data["heatmap"]

        return image, mask, heatmaps, torch.from_numpy(affine).float()

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


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, seg_channels=1, heatmap_channels=5, base_features=32):
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
        
        # Task-specific branches (diverge here)
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
# ============================================================
# LOSS
# ============================================================
class DiceBCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets.sum(dim=(2, 3, 4))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return bce + (1 - dice.mean())

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        B, C = probs.shape[:2]
        probs_flat = probs.view(B, C, -1)
        targ_flat = targets.view(B, C, -1)

        intersection = (probs_flat * targ_flat).sum(-1)
        union = probs_flat.sum(-1) + targ_flat.sum(-1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=1.0, heatmap_weight=1.0):
        super().__init__()
        self.seg_loss = DiceBCELoss()
        self.heatmap_loss = SoftDiceLoss()
        self.seg_weight = seg_weight
        self.heatmap_weight = heatmap_weight

    def forward(self, seg_logits, heatmap_logits, seg_targets, heatmap_targets):
        seg_l = self.seg_loss(seg_logits, seg_targets)
        heatmap_l = self.heatmap_loss(heatmap_logits, heatmap_targets)
        return self.seg_weight * seg_l + self.heatmap_weight * heatmap_l, seg_l, heatmap_l

# ============================================================
# LANDMARK DISTANCE COMPUTATION
# ============================================================
def compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5, return_coords=False):
    """
    Compute distances between predicted and ground truth landmark centroids.
    pred_heatmaps: (num_landmarks, D, H, W) numpy array
    gt_heatmaps: (num_landmarks, D, H, W) numpy array
    orig_affine: (4,4) numpy array (voxel->world)
    return_coords: if True, also return pred and gt world coordinates
    Returns: distances_mm (array of distances in mm), or (distances_mm, pred_coords, gt_coords)
    """
    num = len(landmark_ids)
    distances_mm = np.zeros(num, dtype=float)
    pred_coords_mm = np.zeros((num, 3), dtype=float)
    gt_coords_mm = np.zeros((num, 3), dtype=float)

    for i, lm_id in enumerate(landmark_ids):
        pred_heatmap = pred_heatmaps[i]
        mask_pred = pred_heatmap >= threshold
        if np.any(mask_pred):
            coords_pred = np.argwhere(mask_pred)  # N x 3 (z,y,x)
            weights_pred = pred_heatmap[mask_pred]
            pred_centroid = np.average(coords_pred, axis=0, weights=weights_pred)
            rounded_pred_centroid = np.round(pred_centroid).astype(int)
        else:
            # fallback to global argmax
            flat_idx = np.argmax(pred_heatmap)
            rounded_pred_centroid = np.array(np.unravel_index(flat_idx, pred_heatmap.shape))

        gt_heatmap = gt_heatmaps[i]
        mask_gt = gt_heatmap >= threshold
        if np.any(mask_gt):
            coords_gt = np.argwhere(mask_gt)
            weights_gt = gt_heatmap[mask_gt]
            gt_centroid = np.average(coords_gt, axis=0, weights=weights_gt)
        else:
            flat_idx = np.argmax(gt_heatmap)
            gt_centroid = np.array(np.unravel_index(flat_idx, gt_heatmap.shape)).astype(float)

        # Convert voxel indices (z,y,x) to world (mm)
        pred_voxel = np.append(rounded_pred_centroid[::-1], 1)  # convert (z,y,x) -> (x,y,z,1)
        gt_voxel = np.append(gt_centroid[::-1], 1)
        pred_world = orig_affine @ pred_voxel
        gt_world = orig_affine @ gt_voxel
        dist_mm = np.linalg.norm(pred_world[:3] - gt_world[:3])
        distances_mm[i] = dist_mm
        pred_coords_mm[i] = pred_world[:3]
        gt_coords_mm[i] = gt_world[:3]

    if return_coords:
        return distances_mm, pred_coords_mm, gt_coords_mm
    return distances_mm

# ============================================================
# TRAINING LOOP
# ============================================================
def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, run_dir, train_dataset, val_dataset, landmark_ids):
    writer = SummaryWriter(run_dir)
    model.to(DEVICE)

    best_val_loss = float("inf")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.5)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_heatmap_loss = 0.0

        for imgs, masks, heatmaps, _ in tqdm(train_loader, desc="Training"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            heatmaps = heatmaps.to(DEVICE)

            optimizer.zero_grad()
            seg_logits, heatmap_logits = model(imgs)
            
            loss, seg_l, heatmap_l = loss_fn(seg_logits, heatmap_logits, masks, heatmaps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_seg_loss += seg_l.item()
            train_heatmap_loss += heatmap_l.item()

        train_loss /= len(train_loader)
        train_seg_loss /= len(train_loader)
        train_heatmap_loss /= len(train_loader)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_heatmap_loss = 0.0
        val_distances_mm = []
        
        with torch.no_grad():
            for imgs, masks, heatmaps, affines in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                heatmaps = heatmaps.to(DEVICE)

                seg_logits, heatmap_logits = model(imgs)
                
                loss, seg_l, heatmap_l = loss_fn(seg_logits, heatmap_logits, masks, heatmaps)
                val_loss += loss.item()
                val_seg_loss += seg_l.item()
                val_heatmap_loss += heatmap_l.item()
                
                # Compute landmark distances
                pred_heatmaps = torch.sigmoid(heatmap_logits[0]).cpu().numpy()
                gt_heatmaps = heatmaps[0].cpu().numpy()
                orig_affine = affines[0].cpu().numpy()
                distances = compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids)
                val_distances_mm.append(distances)

        val_loss /= len(val_loader)
        val_seg_loss /= len(val_loader)
        val_heatmap_loss /= len(val_loader)
        avg_val_distances = np.mean(val_distances_mm, axis=0) if val_distances_mm else np.zeros(len(landmark_ids))
        
        # Debug: Print GT vs Predicted coordinates for first validation sample
        if len(val_dataset) > 0:
            val_img_debug, val_mask_debug, val_heatmaps_debug, val_affine_debug = val_dataset[0]
            val_img_debug_batch = val_img_debug.unsqueeze(0).to(DEVICE)
            seg_debug, heatmap_debug = model(val_img_debug_batch)
            pred_heatmaps_debug = torch.sigmoid(heatmap_debug[0]).cpu().detach().numpy()
            gt_heatmaps_debug = val_heatmaps_debug.cpu().numpy()
            orig_affine_debug = val_affine_debug.cpu().numpy()
            distances_debug, pred_coords_debug, gt_coords_debug = compute_landmark_distances(
                pred_heatmaps_debug, gt_heatmaps_debug, orig_affine_debug, landmark_ids, return_coords=True
            )
            print("\n" + "="*80)
            print(f"VALIDATION DEBUG - First sample ({val_dataset.file_list[0]})")
            print("="*80)
            for i, lm_id in enumerate(landmark_ids):
                print(f"Landmark {lm_id}:")
                print(f"  GT (mm):   [{gt_coords_debug[i, 0]:7.2f}, {gt_coords_debug[i, 1]:7.2f}, {gt_coords_debug[i, 2]:7.2f}]")
                print(f"  Pred (mm): [{pred_coords_debug[i, 0]:7.2f}, {pred_coords_debug[i, 1]:7.2f}, {pred_coords_debug[i, 2]:7.2f}]")
                print(f"  Distance:  {distances_debug[i]:.2f} mm")
            print("="*80 + "\n")
        
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Loss/Train_Total", train_loss, epoch)
        writer.add_scalar("Loss/Train_Seg", train_seg_loss, epoch)
        writer.add_scalar("Loss/Train_Heatmap", train_heatmap_loss, epoch)
        writer.add_scalar("Loss/Val_Total", val_loss, epoch)
        writer.add_scalar("Loss/Val_Seg", val_seg_loss, epoch)
        writer.add_scalar("Loss/Val_Heatmap", val_heatmap_loss, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        # Log landmark distances
        for i, lm_id in enumerate(landmark_ids):
            writer.add_scalar(f"Distance_MM/Val/LM{lm_id}", avg_val_distances[i], epoch)

        # ---------- Visualize predictions ----------
        model.eval()
        with torch.no_grad():
            # Validation sample visualization
            val_img, val_mask, val_heatmaps_gt, val_affine = val_dataset[0]
            val_img_batch = val_img.unsqueeze(0).to(DEVICE)
            val_mask_batch = val_mask.unsqueeze(0)
            val_heatmaps_gt_batch = val_heatmaps_gt.unsqueeze(0)
            
            seg_pred, heatmap_pred = model(val_img_batch)
            
            seg_pred = torch.sigmoid(seg_pred)
            heatmap_pred = torch.sigmoid(heatmap_pred)
            
            # Get middle slice for segmentation visualization
            seg_slice_idx = val_mask.shape[2] // 2
            
            # Segmentation visualization
            img_slice = val_img[0, :, seg_slice_idx, :].cpu().numpy()  # (D, W)
            mask_slice = val_mask[0, :, seg_slice_idx, :].cpu().numpy()  # (D, W)
            seg_pred_slice = seg_pred[0, 0, :, seg_slice_idx, :].cpu().numpy()  # (D, W)
            
            # Enhance contrast for CT image visualization
            img_slice_vis = np.clip(img_slice, 0, 1)
            if img_slice_vis.max() > img_slice_vis.min():
                img_slice_vis = (img_slice_vis - img_slice_vis.min()) / (img_slice_vis.max() - img_slice_vis.min())
            
            seg_vis = np.hstack([img_slice_vis, mask_slice, seg_pred_slice])
            writer.add_image("Val/Segmentation_Image_GT_Pred", seg_vis[None, :, :], epoch, dataformats='CHW')
            
            # Heatmap visualization for each landmark (with per-landmark slice selection)
            for i, lm_id in enumerate(landmark_ids):
                # Get slice index for this specific landmark
                lm_slice_idx = HEATMAP_VIZ_SLICE.get(lm_id, None)
                if lm_slice_idx is None:
                    lm_slice_idx = val_mask.shape[2] // 2  # Default to middle slice
                
                # Extract slices for this landmark's visualization
                lm_img_slice = val_img[0, :, lm_slice_idx, :].cpu().numpy()  # (D, W)
                heatmap_gt_slice = val_heatmaps_gt[i, :, lm_slice_idx, :].cpu().numpy()  # (D, W)
                heatmap_pred_slice = heatmap_pred[0, i, :, lm_slice_idx, :].cpu().numpy()  # (D, W)
                
                # Enhance contrast for this landmark's image slice
                lm_img_slice_vis = np.clip(lm_img_slice, 0, 1)
                if lm_img_slice_vis.max() > lm_img_slice_vis.min():
                    lm_img_slice_vis = (lm_img_slice_vis - lm_img_slice_vis.min()) / (lm_img_slice_vis.max() - lm_img_slice_vis.min())
                
                heatmap_vis = np.hstack([lm_img_slice_vis, heatmap_gt_slice, heatmap_pred_slice])
                writer.add_image(f"Val/Heatmap_LM{lm_id}_Image_GT_Pred", heatmap_vis[None, :, :], epoch, dataformats='CHW')

        print(f"Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Heatmap: {train_heatmap_loss:.4f})")
        print(f"Val Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Heatmap: {val_heatmap_loss:.4f})")
        print(f"Val Landmark Distances (mm): {[f'{d:.2f}' for d in avg_val_distances]}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_seg_loss': val_seg_loss,
                'val_heatmap_loss': val_heatmap_loss
            }, os.path.join(run_dir, "best_model.pth"))
            print("✔ Best model saved")

    writer.close()

# ============================================================
# INFERENCE ON VALIDATION SET
# ============================================================
def save_validation_predictions(model, val_dataset, run_dir, device, landmark_ids):
    """Load best model and save predictions for all validation samples."""
    # Load best model
    best_model_path = os.path.join(run_dir, "best_model.pth")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory for predictions
    pred_dir = os.path.join(run_dir, "validation_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Generating predictions for {len(val_dataset)} validation samples...")
    print(f"Saving to: {pred_dir}")
    print(f"{'='*80}\n")
    
    all_distances = []
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Saving predictions"):
            img, mask, heatmaps_gt, affine = val_dataset[idx]
            fname = val_dataset.file_list[idx]
            
            # Add batch dimension and predict
            img_batch = img.unsqueeze(0).to(device)
            seg_pred, heatmap_pred = model(img_batch)
            
            seg_pred = torch.sigmoid(seg_pred)
            heatmap_pred = torch.sigmoid(heatmap_pred)
            
            # Convert to numpy and threshold
            pred_mask = seg_pred[0, 0].cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Get the original NRRD header to preserve spatial information
            original_header = val_dataset.nrrd_headers.get(fname, None)
            
            # Save segmentation prediction as nrrd WITH original header
            output_name = fname.replace(".nii.gz", "_pred_seg.nrrd")
            output_path = os.path.join(pred_dir, output_name)
            
            if original_header is not None:
                # Use original header to preserve space directions, space origin, etc.
                nrrd.write(output_path, pred_mask, header=original_header)
            else:
                print(f"[WARNING] No header found for {fname}, saving without spatial metadata")
                nrrd.write(output_path, pred_mask)
            
            # Save heatmap predictions
            pred_heatmaps = heatmap_pred[0].cpu().numpy()  # (num_landmarks, D, H, W)
            heatmap_output_name = fname.replace(".nii.gz", "_pred_heatmaps.pt")
            heatmap_output_path = os.path.join(pred_dir, heatmap_output_name)
            torch.save(torch.from_numpy(pred_heatmaps), heatmap_output_path)
            
            # Compute and save landmark distances
            gt_heatmaps = heatmaps_gt.cpu().numpy()
            distances = compute_landmark_distances(pred_heatmaps, gt_heatmaps, affine.numpy(), landmark_ids)
            all_distances.append(distances)
    
    # Save distance statistics
    all_distances = np.array(all_distances)
    distance_stats = {
        'mean': np.mean(all_distances, axis=0),
        'std': np.std(all_distances, axis=0),
        'median': np.median(all_distances, axis=0),
        'min': np.min(all_distances, axis=0),
        'max': np.max(all_distances, axis=0)
    }
    
    stats_path = os.path.join(pred_dir, "landmark_distance_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("Landmark Distance Statistics (mm)\n")
        f.write("=" * 50 + "\n\n")
        for i, lm_id in enumerate(landmark_ids):
            f.write(f"Landmark {lm_id}:\n")
            f.write(f"  Mean: {distance_stats['mean'][i]:.2f} mm\n")
            f.write(f"  Std:  {distance_stats['std'][i]:.2f} mm\n")
            f.write(f"  Median: {distance_stats['median'][i]:.2f} mm\n")
            f.write(f"  Min: {distance_stats['min'][i]:.2f} mm\n")
            f.write(f"  Max: {distance_stats['max'][i]:.2f} mm\n\n")
    
    print(f"\n✓ All validation predictions saved to: {pred_dir}")
    print(f"✓ Distance statistics saved to: {stats_path}\n")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = create_or_load_split(CT_DIR, SPLIT_CSV, TRAIN_RATIO)

    train_files = df[df.split == "train"].scan_name.tolist()
    val_files = df[df.split == "val"].scan_name.tolist()

    train_dataset = SegmentationDataset(CT_DIR, SEG_DIR, HEATMAPS_DIR, train_files, augment=True)
    val_dataset = SegmentationDataset(CT_DIR, SEG_DIR, HEATMAPS_DIR, val_files, augment=False)

    # Load first sample to check alignment
    print("\n" + "="*80)
    print("CHECKING DATA ALIGNMENT")
    print("="*80)
    sample_img, sample_mask, sample_heatmaps, sample_affine = train_dataset[0]
    print(f"CT shape: {sample_img.shape}")
    print(f"Mask shape: {sample_mask.shape}")
    print(f"Heatmaps shape: {sample_heatmaps.shape}")
    print(f"Affine shape: {sample_affine.shape}")
    
    # Check if shapes match (excluding channel dimension)
    if sample_img.shape[1:] != sample_mask.shape[1:]:
        print(f"[ERROR] Shape mismatch! CT spatial dims: {sample_img.shape[1:]}, Mask spatial dims: {sample_mask.shape[1:]}")
    else:
        print("[OK] CT and Mask spatial dimensions match!")
    print("="*80 + "\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet3D(in_channels=1, seg_channels=1, heatmap_channels=NUM_LANDMARKS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    loss_fn = CombinedLoss(seg_weight=1.0, heatmap_weight=1.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print(f"Training dual-task model: Segmentation + {NUM_LANDMARKS} Landmark Heatmaps")
    print(f"Landmark IDs: {LANDMARK_IDS}")
    print(f"TensorBoard logs will be saved to: {run_dir}")
    print(f"To visualize, run: tensorboard --logdir={run_dir}")
    print("=" * 80)

    train(train_loader, val_loader, model, optimizer, loss_fn, MAX_EPOCHS, run_dir, 
          train_dataset, val_dataset, LANDMARK_IDS)
    
    # Generate and save validation predictions
    save_validation_predictions(model, val_dataset, run_dir, DEVICE, LANDMARK_IDS)
