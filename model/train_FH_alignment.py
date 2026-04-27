import os
import time
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# ---------------------------
# Configuration
# ---------------------------
BATCH_SIZE = 1
NUM_WORKERS = 8
LR = 1e-4
MAX_EPOCHS = 500
COMPUTE_FULL_METRICS_EVERY_N_EPOCHS = 1  # Compute metrics on all batches every N epochs
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Initialize CUDA / cuDNN
if torch.cuda.is_available():
    # don't call non-existent torch.cuda.init()
    torch.backends.cudnn.benchmark = True  # enable cuDNN autotuner (good for fixed-size inputs)
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
    DEVICE = torch.device('cuda')
    try:
        dev_name = torch.cuda.get_device_name(0)
    except Exception:
        dev_name = "cuda"
    print(f"Using device: {DEVICE} - {dev_name}")
else:
    DEVICE = torch.device('cpu')
    print("Using device: cpu")


# ---------------------------
# Argument Parser
# ---------------------------
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Train FH Alignment Model")
    parser.add_argument('--ct_dir', type=str, required=True, help='Directory with processed CTs (P1 output)')
    parser.add_argument('--json_dir', type=str, required=True, help='Directory with ground truth JSONs (not used directly, but for reference)')
    parser.add_argument('--heatmap_dir', type=str, required=True, help='Directory with precomputed heatmaps')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to save logs and tensorboard runs')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save trained models')
    parser.add_argument('--split_csv', type=str, default=None, help='CSV file with train/val split (optional)')
    return parser.parse_args()

args = None
if __name__ == '__main__':
    args = parse_args()
    pred_dir = args.model_dir
    os.makedirs(pred_dir, exist_ok=True)
    base_log_dir = args.log_dir
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = os.path.join(base_log_dir, f"run_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"Logs will be saved in: {run_log_dir}")

# ---------------------------
# Helpers
# ---------------------------
def load_nii(nii_path: str):
    """Load NIfTI and return image tensor (1,C,D,H,W) and affine (4x4 numpy)."""
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()

    # Robust intensity normalization to 0..1 using clipped percentiles (works for CT)
    p1, p99 = np.percentile(data, (1.0, 99.0))
    data = np.clip(data, p1, p99)
    if p99 - p1 > 1e-6:
        data = (data - p1) / (p99 - p1)
    else:
        data = data - p1  # fallback

    tensor = torch.from_numpy(data).unsqueeze(0).float()  # shape (1, D, H, W)
    # Convert to (1, 1, D, H, W) expected by Conv3d with batch dim handled by dataloader
    # Actually return (C,D,H,W) and the dataset will unsqueeze batch when needed
    return tensor, affine

# ---------------------------
# Dataset
# ---------------------------
class LandmarkDataset(Dataset):
    def __init__(self, nii_dir, heatmap_dir):
        self.nii_dir = nii_dir
        self.heatmap_dir = heatmap_dir
        self.nii_files = sorted([f for f in os.listdir(nii_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.nii_files)

    def __getitem__(self, idx):
        # Load CT scan
        nii_path = os.path.join(self.nii_dir, self.nii_files[idx])
        img_tensor, affine = load_nii(nii_path)  # img_tensor shape (1, D, H, W)

        # Load precomputed heatmaps (torch .pt) - assume shape (num_landmarks, D, H, W)
        heatmap_name = self.nii_files[idx].replace('.nii.gz', '_heatmaps.pt')
        heatmap_path = os.path.join(self.heatmap_dir, heatmap_name)
        # torch.load does not accept weights_only/mmap kwargs; load to CPU then move to device later
        heatmaps = torch.load(heatmap_path, map_location='cpu').float()

        # return as tensors; DataLoader will collate into batch dimension
        return img_tensor, heatmaps, torch.from_numpy(affine).float()

# ---------------------------
# Model
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_features=16):
        super().__init__()
        f = base_features
        # Encoder
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)    # fixed: previously incorrect
        self.enc4 = DoubleConv(f * 4, f * 8)

        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        # Decoder: channel sizes reflect concatenations
        self.dec3 = DoubleConv(f * 8 + f * 4, f * 4)  # input channels after concat
        self.dec2 = DoubleConv(f * 4 + f * 2, f * 2)
        self.dec1 = DoubleConv(f * 2 + f, f)

        self.out_conv = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape expected (B, C, D, H, W)
        e1 = self.enc1(x)              # B, f, ...
        e2 = self.enc2(self.pool(e1))  # B, f*2, ...
        e3 = self.enc3(self.pool(e2))  # B, f*4, ...
        e4 = self.enc4(self.pool(e3))  # B, f*8, ...

        d3 = self.up(e4)
        # ensure same spatial dims (in case of odd sizes) by center crop or interpolation - here we rely on upsample
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out

# ---------------------------
# Loss and Metrics
# ---------------------------
class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # ensure shapes: (B, C, -1)
        B, C = probs.shape[:2]
        probs_flat = probs.view(B, C, -1)
        targ_flat = targets.view(B, C, -1)

        intersection = (probs_flat * targ_flat).sum(-1)
        union = probs_flat.sum(-1) + targ_flat.sum(-1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice
        return loss.mean()



# ---------------------------
# Distance Computation (Weighted Centroid)
# ---------------------------
def compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5):
    """
    pred_heatmaps : (num_landmarks, D, H, W) numpy array
    gt_heatmaps   : (num_landmarks, D, H, W) numpy array
    orig_affine   : (4,4) numpy array (voxel->world)
    """
    num = len(landmark_ids)
    distances_mm = np.zeros(num, dtype=float)
    distances_pixels = np.zeros(num, dtype=float)
    gt_centroids_all = np.zeros((num, 3), dtype=float)
    pred_centroids_all = np.zeros((num, 3), dtype=float)
    rounded_pred_centroids_all = np.zeros((num, 3), dtype=int)

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
            pred_centroid = rounded_pred_centroid.astype(float)

        gt_heatmap = gt_heatmaps[i]
        mask_gt = gt_heatmap >= threshold
        if np.any(mask_gt):
            coords_gt = np.argwhere(mask_gt)
            weights_gt = gt_heatmap[mask_gt]
            gt_centroid = np.average(coords_gt, axis=0, weights=weights_gt)
        else:
            flat_idx = np.argmax(gt_heatmap)
            gt_centroid = np.array(np.unravel_index(flat_idx, gt_heatmap.shape)).astype(float)

        dist_pixels = np.linalg.norm(rounded_pred_centroid - gt_centroid)
        distances_pixels[i] = dist_pixels
        rounded_pred_centroids_all[i] = rounded_pred_centroid
        gt_centroids_all[i] = gt_centroid
        pred_centroids_all[i] = pred_centroid

        # convert voxel indices (z,y,x) to world (mm) by applying affine to (x,y,z,1) -> need correct ordering:
        # nibabel uses (i,j,k) voxel coords -> world coords: affine @ [i,j,k,1]
        pred_voxel = np.append(rounded_pred_centroid[::-1], 1)  # convert (z,y,x) -> (x,y,z)
        gt_voxel = np.append(gt_centroid[::-1], 1)
        pred_world = orig_affine @ pred_voxel
        gt_world = orig_affine @ gt_voxel
        dist_mm = np.linalg.norm(pred_world[:3] - gt_world[:3])
        distances_mm[i] = dist_mm

    return distances_mm, distances_pixels, rounded_pred_centroids_all, gt_centroids_all, pred_centroids_all

# ---------------------------
# Training Loop
# ---------------------------
def train(train_loader, val_loader, model, optimizer, loss_fn, epochs=MAX_EPOCHS,
          save_path='best_model.pth', run_dir=run_log_dir, landmark_ids=None,
          min_lr=1e-6, patience_lr=10, factor_lr=0.5, early_stop_patience=20):

    # Safeguards
    if landmark_ids is None:
        raise ValueError("landmark_ids must be provided")

    print(f"Training Configuration:\nOptimizer: {optimizer}\nLoss Function: {loss_fn}\nEpochs: {epochs}\nNumber of Landmarks: {len(landmark_ids)}\nInitial LR: {optimizer.param_groups[0]['lr']}\nMin LR: {min_lr}\nPatience LR: {patience_lr}\nFactor LR: {factor_lr}\nEarly Stop Patience: {early_stop_patience}\n")

    writer = SummaryWriter(run_dir)
    model = model.to(DEVICE)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor_lr, patience=patience_lr)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        model.train()
        train_loss = 0.0
        n = 0

        data_load_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        metric_time = 0.0
        batch_start = time.time()

        for batch_idx, (imgs, heatmaps, affines) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)):
            data_end = time.time()
            data_load_time += (data_end - batch_start)

            # Move to device. imgs shape: (B,1,D,H,W) from dataset
            imgs = imgs.to(DEVICE, non_blocking=True)
            heatmaps = heatmaps.to(DEVICE, non_blocking=True)
            orig_affine = affines[0].cpu().numpy()

            optimizer.zero_grad()
            gpu_start = time.time()
            logits = model(imgs)
            # Resize logits to match heatmap size (C, D, H, W)
            logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)
            loss = loss_fn(logits, heatmaps)
            forward_end = time.time()
            forward_time += (forward_end - gpu_start)

            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_end = time.time()
            backward_time += (backward_end - backward_start)

            train_loss += loss.item()
            n += 1

            batch_start = time.time()

        # Prevent division by zero
        avg_train_loss = train_loss / max(n, 1)

        # Compute/Log metrics (either all batches every N epochs or first batch)
        compute_all_batches = (epoch % COMPUTE_FULL_METRICS_EVERY_N_EPOCHS == 0)

        # Default metric placeholders
        avg_train_mm = np.zeros(len(landmark_ids))
        avg_train_px = np.zeros(len(landmark_ids))

        if len(train_loader) > 0:
            if compute_all_batches:
                metric_start = time.time()
                train_distances_mm_list = []
                train_distances_px_list = []

                with torch.no_grad():
                    for imgs, heatmaps, affines in tqdm(train_loader, desc=f"Computing full train metrics", leave=False):
                        imgs = imgs.to(DEVICE, non_blocking=True)
                        heatmaps = heatmaps.to(DEVICE, non_blocking=True)
                        orig_affine = affines[0].cpu().numpy()

                        logits = model(imgs)
                        logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)

                        pred_heatmaps = torch.sigmoid(logits[0]).detach().cpu().numpy()
                        gt_heatmaps = heatmaps[0].cpu().numpy()

                        distances_mm, distances_px, _, _, _ = compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5)
                        train_distances_mm_list.append(distances_mm)
                        train_distances_px_list.append(distances_px)

                avg_train_mm = np.mean(train_distances_mm_list, axis=0)
                avg_train_px = np.mean(train_distances_px_list, axis=0)
                metric_time += time.time() - metric_start

                # Visualization on first batch
                first_batch = next(iter(train_loader))
                imgs, heatmaps, affines = first_batch
                imgs = imgs.to(DEVICE, non_blocking=True)
                heatmaps = heatmaps.to(DEVICE, non_blocking=True)
                orig_affine = affines[0].cpu().numpy()
                with torch.no_grad():
                    logits = model(imgs)
                    logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)

                pred_heatmaps = torch.sigmoid(logits[0]).detach().cpu().numpy()
                gt_heatmaps = heatmaps[0].cpu().numpy()
                _, _, rounded_pred_centroids_all, gt_centroids_all, pred_centroids_all = compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5)

            else:
                # Only compute on first batch (cheaper)
                first_batch = next(iter(train_loader))
                imgs, heatmaps, affines = first_batch
                imgs = imgs.to(DEVICE, non_blocking=True)
                heatmaps = heatmaps.to(DEVICE, non_blocking=True)
                orig_affine = affines[0].cpu().numpy()

                metric_start = time.time()
                with torch.no_grad():
                    logits = model(imgs)
                    logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)

                pred_heatmaps = torch.sigmoid(logits[0]).detach().cpu().numpy()
                gt_heatmaps = heatmaps[0].cpu().numpy()

                distances_mm, distances_px, rounded_pred_centroids_all, gt_centroids_all, pred_centroids_all = compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5)
                avg_train_mm = distances_mm
                avg_train_px = distances_px
                metric_time += time.time() - metric_start

                    
        train_phase_time = time.time() - epoch_start
        print(f"\n[TRAIN TIMING] Total: {train_phase_time:.2f}s | Data Load: {data_load_time:.2f}s | Forward: {forward_time:.2f}s | Backward: {backward_time:.2f}s | Metrics: {metric_time:.2f}s")

        # ---------------------------
        # Validation
        # ---------------------------
        val_start = time.time()
        model.eval()
        val_loss = 0.0
        m = 0

        # placeholders
        avg_val_mm = np.zeros(len(landmark_ids))
        avg_val_px = np.zeros(len(landmark_ids))

        with torch.no_grad():
            for batch_idx, (imgs, heatmaps, affines) in enumerate(val_loader):
                imgs = imgs.to(DEVICE, non_blocking=True)
                heatmaps = heatmaps.to(DEVICE, non_blocking=True)
                orig_affine = affines[0].cpu().numpy()

                logits = model(imgs)
                logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)
                loss = loss_fn(logits, heatmaps)
                val_loss += loss.item()
                m += 1

        if m == 0:
            avg_val_loss = float('nan')
        else:
            avg_val_loss = val_loss / m

        # Compute validation metrics similarly to training (all batches or first)
        if len(val_loader) > 0:
            compute_all_val = compute_all_batches
            if compute_all_val:
                val_distances_mm_list = []
                val_distances_px_list = []
                with torch.no_grad():
                    for imgs, heatmaps, affines in tqdm(val_loader, desc=f"Computing full val metrics", leave=False):
                        imgs = imgs.to(DEVICE, non_blocking=True)
                        heatmaps = heatmaps.to(DEVICE, non_blocking=True)
                        orig_affine = affines[0].cpu().numpy()

                        logits = model(imgs)
                        logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)

                        pred_heatmaps = torch.sigmoid(logits[0]).detach().cpu().numpy()
                        gt_heatmaps = heatmaps[0].cpu().numpy()

                        distances_mm, distances_px, _, _, _ = compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5)
                        val_distances_mm_list.append(distances_mm)
                        val_distances_px_list.append(distances_px)

                avg_val_mm = np.mean(val_distances_mm_list, axis=0)
                avg_val_px = np.mean(val_distances_px_list, axis=0)

                # visualization on first val batch
                first_batch = next(iter(val_loader))
                imgs, heatmaps, affines = first_batch
                imgs = imgs.to(DEVICE, non_blocking=True)
                heatmaps = heatmaps.to(DEVICE, non_blocking=True)
                orig_affine = affines[0].cpu().numpy()
                with torch.no_grad():
                    logits = model(imgs)
                    logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)
                pred_heatmaps = torch.sigmoid(logits[0]).detach().cpu().numpy()
                gt_heatmaps = heatmaps[0].cpu().numpy()
                _, _, rounded_pred_centroids_all, gt_centroids_all, pred_centroids_all = compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5)

            else:
                # compute on first validation batch
                first_batch = next(iter(val_loader))
                imgs, heatmaps, affines = first_batch
                imgs = imgs.to(DEVICE, non_blocking=True)
                heatmaps = heatmaps.to(DEVICE, non_blocking=True)
                orig_affine = affines[0].cpu().numpy()

                with torch.no_grad():
                    logits = model(imgs)
                    logits = torch.nn.functional.interpolate(logits, size=heatmaps.shape[2:], mode='trilinear', align_corners=False)

                pred_heatmaps = torch.sigmoid(logits[0]).detach().cpu().numpy()
                gt_heatmaps = heatmaps[0].cpu().numpy()
                distances_mm, distances_px, rounded_pred_centroids_all, gt_centroids_all, pred_centroids_all = compute_landmark_distances(pred_heatmaps, gt_heatmaps, orig_affine, landmark_ids, threshold=0.5)
                avg_val_mm = distances_mm
                avg_val_px = distances_px


                    
        val_phase_time = time.time() - val_start
        print(f"[VAL TIMING] Total: {val_phase_time:.2f}s")

        # Logging
        logging_start = time.time()
        writer.add_scalar('Loss Metrics/Train', avg_train_loss, epoch)
        if not np.isnan(avg_val_loss):
            writer.add_scalar('Loss Metrics/Val', avg_val_loss, epoch)

        metrics_scope = "ALL batches" if compute_all_batches else "first batch"
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Metrics computed on: {metrics_scope}")

        for i, lm_id in enumerate(landmark_ids):
            writer.add_scalar(f'Distance_MM/Train/LM{lm_id}', avg_train_mm[i], epoch)
            writer.add_scalar(f'Distance_MM/Val/LM{lm_id}', avg_val_mm[i], epoch)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)

        writer.flush()
        logging_time = time.time() - logging_start
        epoch_total = time.time() - epoch_start
        print(f"[TENSORBOARD LOGGING] {logging_time:.2f}s")
        print(f"[EPOCH TOTAL] {epoch_total:.2f}s\n")

        # Scheduler / checkpointing / early stopping
        if not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({'model_state_dict': model.state_dict(),
                            'epoch': epoch,
                            'val_loss': avg_val_loss}, save_path)
                epochs_no_improve = 0
                print("Validation loss improved - model saved.")
            else:
                epochs_no_improve += 1
        else:
            # If no validation batches, don't update scheduler but still consider early stop based on train loss (optional)
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"No improvement for {early_stop_patience} epochs. Early stopping.")
            break

        if current_lr <= min_lr:
            print("Learning rate has reached the minimum threshold. Stopping training.")
            break

    writer.close()

# ---------------------------
# Example usage
# ---------------------------
if __name__ == '__main__':
    # Use argparse-provided paths
    nii_dir = args.ct_dir
    heatmap_dir = args.heatmap_dir
    landmark_ids = [8, 9, 10, 11, 12, 13]
    split_csv = args.split_csv
    output_model_path = os.path.join(pred_dir, f"best_model_{timestamp}.pth")

    if split_csv is not None and os.path.exists(split_csv):
        df = pd.read_csv(split_csv)
        train_scans = df[df['split'].str.startswith('train')]['scan_name'].tolist()
        val_scans = df[df['split'].str.startswith('val')]['scan_name'].tolist()
        full_dataset = LandmarkDataset(nii_dir, heatmap_dir)
        train_indices = [i for i, f in enumerate(full_dataset.nii_files) if f.split('__')[0] in train_scans]
        val_indices = [i for i, f in enumerate(full_dataset.nii_files) if f.split('__')[0] in val_scans]
        print(f"Total files in dataset: {len(full_dataset.nii_files)}")
        print(f"Training files: {len(train_indices)}")
        print(f"Validation files: {len(val_indices)}")
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    else:
        # If no split CSV, use all data for training, no validation
        full_dataset = LandmarkDataset(nii_dir, heatmap_dir)
        print(f"Total files in dataset: {len(full_dataset.nii_files)}")
        train_dataset = full_dataset
        val_dataset = torch.utils.data.Subset(full_dataset, [])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True if NUM_WORKERS > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            persistent_workers=True if NUM_WORKERS > 0 else False)

    model = UNet3D(in_channels=1, out_channels=len(landmark_ids), base_features=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = SoftDiceLoss()

    train(train_loader, val_loader, model, optimizer, loss_fn, epochs=MAX_EPOCHS,
          save_path=output_model_path, run_dir=run_log_dir, landmark_ids=landmark_ids,
          min_lr=1e-8, patience_lr=10, factor_lr=0.5, early_stop_patience=100)
