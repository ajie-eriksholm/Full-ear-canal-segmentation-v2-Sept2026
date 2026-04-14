#CODE TO RUN BEFORE RUNNING NNUNET

#import libraries
import os
import json
import math
import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import nibabel as nib
from nibabel.affines import apply_affine

#chageable parameters
LANDMARK_RADIUS_MM = 1.5     # radio de esfera en mm
LANDMARK_CLASSES = [3, 4, 5, 6]  # classes for landmarks
LANDMARK_MODE = "overwrite"  # "overwrite" o "background_only"

#load landmarks from .mrk.json 
def load_landmarks_from_mrk(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "markups" not in data or not data["markups"]:
        raise ValueError(f"Formato inesperado en {json_path}")

    m = data["markups"][0]
    coord = m.get("coordinateSystem", "RAS").upper()
    pts = []
    for cp in m.get("controlPoints", []):
        pos = cp.get("position", None)
        if isinstance(pos, dict):
            pos = pos.get("value", None)
        if pos is not None and len(pos) >= 3:
            pts.append([float(pos[0]), float(pos[1]), float(pos[2])])

    if not pts:
        raise ValueError(f"Sin puntos en {json_path}")
    return np.asarray(pts, dtype=np.float64), coord

def lps_to_ras(points_mm):
    pts = np.asarray(points_mm, dtype=np.float64).copy()
    pts[..., 0] *= -1.0  # L->R
    pts[..., 1] *= -1.0  # P->A
    return pts

def world_ras_mm_to_vox(affine_vox2ras, points_ras_mm):
    inv_aff = np.linalg.inv(affine_vox2ras)
    vox = np.zeros_like(points_ras_mm, dtype=np.float64)
    for i, p in enumerate(points_ras_mm):
        vox[i, :] = apply_affine(inv_aff, p)  # (i,j,k) en XYZ
    return vox

# paint sphere in label_xyz (XYZ) with given center and radius in voxels, using class_id
def paint_sphere_xyz(label_xyz, center_xyz, radius_vox, class_id, mode="overwrite"):
    X, Y, Z = label_xyz.shape
    ci, cj, ck = center_xyz
    r = float(radius_vox)
    if r <= 0:
        return
    rx = int(math.ceil(r))
    i0 = max(0, int(math.floor(ci)) - rx); i1 = min(X - 1, int(math.floor(ci)) + rx)
    j0 = max(0, int(math.floor(cj)) - rx); j1 = min(Y - 1, int(math.floor(cj)) + rx)
    k0 = max(0, int(math.floor(ck)) - rx); k1 = min(Z - 1, int(math.floor(ck)) + rx)
    if i0 > i1 or j0 > j1 or k0 > k1:
        return
    ii = np.arange(i0, i1 + 1, dtype=np.float32)[:, None, None]
    jj = np.arange(j0, j1 + 1, dtype=np.float32)[None, :, None]
    kk = np.arange(k0, k1 + 1, dtype=np.float32)[None, None, :]
    d2 = (ii - ci) ** 2 + (jj - cj) ** 2 + (kk - ck) ** 2
    mask = d2 <= (r ** 2 + 1e-6)
    sub = label_xyz[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
    if mode == "background_only":
        sub[(mask) & (sub == 0)] = class_id
    else:
        sub[mask] = class_id
    label_xyz[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = sub

def same_geometry(img1, img2):
    return (img1.GetSize() == img2.GetSize() and
            np.allclose(img1.GetSpacing(), img2.GetSpacing()) and
            np.allclose(img1.GetOrigin(), img2.GetOrigin()) and
            np.allclose(list(img1.GetDirection()), list(img2.GetDirection())))

def resample_label_to(ct_sitk, lbl_sitk):
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(ct_sitk)
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    res.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    res.SetDefaultPixelValue(0)
    out = res.Execute(lbl_sitk)
    if out.GetPixelID() != sitk.sitkUInt16:
        out = sitk.Cast(out, sitk.sitkUInt16)
    return out


# CT:    ID_right_ear.nii.gz  /  ID_left_ear.nii.gz  (detecta 'right'/'left')
# Mask:  id_lower_right_ear_clean.nii.gz.nii.seg.nrrd
# LMK:   ID_right_ear.mrk.json

def find_mask_path(masks_dir: Path, pid: str, side: str) -> Path:
    pid_lower = pid.lower()
    candidates = [
        f"{pid_lower}_{side}_ear_clean.nii.gz.nii.seg.nrrd", 
        f"{pid_lower}_{side}_ear_clean.nii.gz",               
        f"{pid_lower}_{side}_ear.nii.gz.nii.seg.nrrd",        
        f"{pid_lower}_{side}_ear.nii.gz",
    ]
    for c in candidates:
        p = masks_dir / c
        if p.exists():
            return p
    for p in masks_dir.glob(f"*{pid_lower}*{side}*ear*"):
        if p.suffix in [".nrrd", ".gz"] or p.name.endswith(".nii.gz.nii.seg.nrrd"):
            return p
    return None

def find_landmark_path(landmarks_dir: Path, pid: str, side: str) -> Path:
    candidates = [
        f"{pid}_{side}_ear.mrk.json",  
        f"{pid}_{side}.mrk.json"      
    ]
    for c in candidates:
        p = landmarks_dir / c
        if p.exists():
            return p
    return None


def process_case(ct_path: Path, mask_path: Path, lmk_path: Path,
                 out_img: Path, out_lbl: Path):
    print(f"\nProcessing {ct_path.name}")

    # 1) CT
    ct_sitk = sitk.ReadImage(str(ct_path))
    ct_nib = nib.load(str(ct_path))       # affine RAS
    affine = ct_nib.affine
    zooms = ct_nib.header.get_zooms()[:3]

    # 2) mask (resample if needed)
    if mask_path.suffix == ".nrrd" or mask_path.name.endswith(".nii.gz.nii.seg.nrrd"):
        tmp_nii = out_lbl.parent / (mask_path.stem + ".nii.gz")
        sitk.WriteImage(sitk.ReadImage(str(mask_path)), str(tmp_nii))
        mask_path = tmp_nii

    mask_sitk = sitk.ReadImage(str(mask_path))
    if not same_geometry(ct_sitk, mask_sitk):
        print("  Fixing mask geometry...")
        mask_sitk = resample_label_to(ct_sitk, mask_sitk)

    mask_zyx = sitk.GetArrayFromImage(mask_sitk).astype(np.uint16)  # (Z,Y,X)
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0))                    # -> (X,Y,Z)

    # 3) Landmarks 
    if lmk_path and lmk_path.exists():
        pts_mm, coord = load_landmarks_from_mrk(lmk_path)
        if coord == "LPS":
            pts_mm = lps_to_ras(pts_mm)
        pts_vox_xyz = world_ras_mm_to_vox(affine, pts_mm)
        radius_vox = float(LANDMARK_RADIUS_MM / max(np.mean(zooms), 1e-6))
        num = min(len(LANDMARK_CLASSES), len(pts_vox_xyz))
        for i in range(num):
            class_id = LANDMARK_CLASSES[i]
            center = pts_vox_xyz[i]
            X, Y, Z = mask_xyz.shape
            if not (0 <= center[0] < X and 0 <= center[1] < Y and 0 <= center[2] < Z):
                print(f"  [WARN] LM{i+1} out of bounds (vox): {center} vs {mask_xyz.shape}")
            paint_sphere_xyz(mask_xyz, center, radius_vox, class_id, mode=LANDMARK_MODE)
        if len(pts_vox_xyz) != len(LANDMARK_CLASSES):
            print(f"  [INFO] Landmarks detectadas: {len(pts_vox_xyz)} (se esperaban 4)")
    else:
        print("  Landmark file not found; skipping landmarks.")

    # 4) save output
    out_zyx = np.transpose(mask_xyz, (2, 1, 0))
    out_sitk = sitk.GetImageFromArray(out_zyx.astype(np.uint16))
    out_sitk.CopyInformation(ct_sitk)
    sitk.WriteImage(out_sitk, str(out_lbl))
    shutil.copy(ct_path, out_img)



def main():
    ap = argparse.ArgumentParser(description="Prepare nnU-Net and fusion of 4 landmarks (classes 3..6) in the mask.")
    ap.add_argument("--cts_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--landmarks_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cts_dir = Path(args.cts_dir)
    masks_dir = Path(args.masks_dir)
    lmk_dir = Path(args.landmarks_dir)
    out = Path(args.out_dir)
    imagesTr = out / "imagesTr"
    labelsTr = out / "labelsTr"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)

    for fname in os.listdir(cts_dir):
        if not fname.lower().endswith(".nii.gz"):
            continue
        stem = fname[:-7]
        name_l = stem.lower()

        # Detect side
        if "right" in name_l:
            side = "right"
        elif "left" in name_l:
            side = "left"
        else:
            print(f"[SKIP] No side (right/left) in: {fname}")
            continue

        pid = stem.split("_")[0]  # e.g. USZ-003
        mask_path = find_mask_path(masks_dir, pid, side)
        lmk_path  = find_landmark_path(lmk_dir, pid, side)

        if mask_path is None:
            print(f"[MISS] Mask not found for {pid}_{side}")
            continue

        ct_path = cts_dir / fname
        out_img = imagesTr / f"{pid}_{side}_0000.nii.gz"
        out_lbl = labelsTr / f"{pid}_{side}.nii.gz"

        process_case(ct_path, mask_path, lmk_path, out_img, out_lbl)

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "skull": 1,
            "mandible": 2,
            "lm1": 3,
            "lm2": 4,
            "lm3": 5,
            "lm4": 6
        },
        "numTraining": len(list(imagesTr.glob("*_0000.nii.gz"))),
        "file_ending": ".nii.gz"
    }
    with open(out / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4)

    print("\n Dataset READY for nnU-Net")
    print(f"  imagesTr: {imagesTr}")
    print(f"  labelsTr: {labelsTr}")

if __name__ == "__main__":
    main()