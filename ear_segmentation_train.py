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
MAX_EPOCHS = 300
SEED = 12345
TRAIN_RATIO = 0.85

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATHS (YOUR DIRECTORIES)
# ============================================================
CT_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_CTs"
SEG_DIR = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Subset_100_all_Masks"

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
    def __init__(self, ct_dir, seg_dir, file_list, augment=False):
        self.ct_dir = ct_dir
        self.seg_dir = seg_dir
        self.file_list = file_list
        self.augment = augment

        if augment:
            self.transform = Compose([
                RandAffined(
                    keys=["image", "mask"],
                    prob=0.8,
                    rotate_range=(0.35, 0.35, 0.35),
                    translate_range=(10, 10, 10),
                    mode=["bilinear", "nearest"],
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

        mask, _ = nrrd.read(seg_path)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # ---------- Augmentation ----------
        if self.transform is not None:
            data = {"image": image, "mask": mask}
            data = self.transform(data)
            image, mask = data["image"], data["mask"]

        return image, mask, torch.from_numpy(affine).float()

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
    def __init__(self, in_channels=1, out_channels=1, base=16):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)

        self.pool = nn.MaxPool3d(2)

        self.dec3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.dec1 = DoubleConv(base * 2 + base, base)

        self.out_conv = nn.Conv3d(base, out_channels, 1)

    def _upsample_to(self, x, ref):
        # Make x exactly the same D×H×W as ref
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x):
        e1 = self.enc1(x)                      # 90
        e2 = self.enc2(self.pool(e1))          # 45
        e3 = self.enc3(self.pool(e2))          # 22
        e4 = self.enc4(self.pool(e3))          # 11

        d3 = self.dec3(torch.cat([self._upsample_to(e4, e3), e3], dim=1))  # 22
        d2 = self.dec2(torch.cat([self._upsample_to(d3, e2), e2], dim=1))  # 45 (not 44)
        d1 = self.dec1(torch.cat([self._upsample_to(d2, e1), e1], dim=1))  # 90

        return self.out_conv(d1)

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

# ============================================================
# TRAINING LOOP
# ============================================================
def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, run_dir, train_dataset, val_dataset):
    writer = SummaryWriter(run_dir)
    model.to(DEVICE)

    best_val_loss = float("inf")
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=15, factor=0.5)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        train_loss = 0.0

        for imgs, masks, _ in tqdm(train_loader, desc="Training"):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            logits = nn.functional.interpolate(logits, size=masks.shape[2:], mode="trilinear", align_corners=False)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(imgs)
                logits = nn.functional.interpolate(logits, size=masks.shape[2:], mode="trilinear", align_corners=False)
                loss = loss_fn(logits, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)

        # ---------- Visualize predictions ----------
        model.eval()
        with torch.no_grad():
                # Training sample (augmented) - get fresh sample from augmented dataset
                aug_img, aug_mask, _ = train_dataset[0]
                aug_img = aug_img.unsqueeze(0).to(DEVICE)
                aug_mask = aug_mask.unsqueeze(0)
                aug_pred = model(aug_img)
                aug_pred = nn.functional.interpolate(aug_pred, size=aug_mask.shape[2:], mode="trilinear", align_corners=False)
                aug_pred = torch.sigmoid(aug_pred)
                
                # Validation sample - add batch dimension
                val_img, val_mask, _ = val_dataset[0]
                val_img = val_img.unsqueeze(0).to(DEVICE)  # Add batch dim
                val_mask = val_mask.unsqueeze(0)  # Add batch dim
                val_pred = model(val_img)
                val_pred = nn.functional.interpolate(val_pred, size=val_mask.shape[2:], mode="trilinear", align_corners=False)
                val_pred = torch.sigmoid(val_pred)
                
                # Get middle slice for visualization
                def get_visualization_slice(img, mask, pred):
                    # Get middle slice across x axis (sagittal view)
                    mask_cpu = mask[0, 0].cpu().numpy()
                    middle_x = mask_cpu.shape[2] // 2  # W dimension
                    
                    img_slice = img[0, 0, :, :, middle_x].cpu().numpy()
                    mask_slice = mask_cpu[:, :, middle_x]
                    pred_slice = pred[0, 0, :, :, middle_x].cpu().numpy()
                    
                    # Normalize image slice for better visualization (0-1 range with contrast)
                    img_slice = np.clip(img_slice, 0, 1)
                    if img_slice.max() > img_slice.min():
                        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
                    
                    # Stack horizontally: [image, ground truth, prediction]
                    vis = np.hstack([img_slice, mask_slice, pred_slice])
                    return vis
                
                aug_vis = get_visualization_slice(aug_img, aug_mask, aug_pred)
                val_vis = get_visualization_slice(val_img, val_mask, val_pred)
                
                # Log to tensorboard (add channel dimension for grayscale)
                writer.add_image("Train/Image_GT_Pred_Augmented", aug_vis[None, :, :], epoch, dataformats='CHW')
                writer.add_image("Val/Image_GT_Pred", val_vis[None, :, :], epoch, dataformats='CHW')

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            print("✔ Best model saved")

    writer.close()

# ============================================================
# INFERENCE ON VALIDATION SET
# ============================================================
def save_validation_predictions(model, val_dataset, run_dir, device):
    """Load best model and save predictions for all validation samples."""
    # Load best model
    best_model_path = os.path.join(run_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # Create output directory for predictions
    pred_dir = os.path.join(run_dir, "validation_predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Generating predictions for {len(val_dataset)} validation samples...")
    print(f"Saving to: {pred_dir}")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Saving predictions"):
            img, mask, affine = val_dataset[idx]
            fname = val_dataset.file_list[idx]
            
            # Add batch dimension and predict
            img_batch = img.unsqueeze(0).to(device)
            pred = model(img_batch)
            pred = nn.functional.interpolate(pred, size=mask.shape[1:], mode="trilinear", align_corners=False)
            pred = torch.sigmoid(pred)
            
            # Convert to numpy and threshold
            pred_mask = pred[0, 0].cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # Save as nrrd with same name as input
            output_name = fname.replace(".nii.gz", "_pred.nrrd")
            output_path = os.path.join(pred_dir, output_name)
            nrrd.write(output_path, pred_mask)
    
    print(f"\n✓ All validation predictions saved to: {pred_dir}\n")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = create_or_load_split(CT_DIR, SPLIT_CSV, TRAIN_RATIO)

    train_files = df[df.split == "train"].scan_name.tolist()
    val_files = df[df.split == "val"].scan_name.tolist()

    train_dataset = SegmentationDataset(CT_DIR, SEG_DIR, train_files, augment=True)
    val_dataset = SegmentationDataset(CT_DIR, SEG_DIR, val_files, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = UNet3D()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = DiceBCELoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(LOG_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print(f"TensorBoard logs will be saved to: {run_dir}")
    print(f"To visualize, run: tensorboard --logdir={run_dir}")
    print("=" * 80)

    train(train_loader, val_loader, model, optimizer, loss_fn, MAX_EPOCHS, run_dir, train_dataset, val_dataset)
    
    # Generate and save validation predictions
    save_validation_predictions(model, val_dataset, run_dir, DEVICE)
