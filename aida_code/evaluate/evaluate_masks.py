import os
import nibabel as nib
import numpy as np

pred_dir = "/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results/TISSUE_TEST_ok"
gt_dir   = "/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_raw/Dataset002_Ear/labelsTs"

def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)

files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

for f in files:
    pred_path = os.path.join(pred_dir, f)
    gt_path   = os.path.join(gt_dir, f)

    if not os.path.exists(gt_path):
        print(f"No GT para {f}")
        continue

    pred_img = nib.load(pred_path).get_fdata()
    gt_img   = nib.load(gt_path).get_fdata()

    d = dice_score(pred_img, gt_img)
    print(f"{f} → Dice: {d:.4f}")