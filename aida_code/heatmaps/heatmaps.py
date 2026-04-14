#!/usr/bin/env python3
"""
Convert JSON landmarks to Gaussian heatmaps and/or nnU-Net-compatible label maps.

- Supports 3D Slicer Markups JSON (.mrk.json) and simple JSON lists.
- Handles RAS/LPS coordinate systems.
- Saves outputs aligned to a reference NIfTI image (same affine, shape, header).

USAGE (examples):
  python landmarks_to_heatmaps.py \
      --json path/to/landmarks.mrk.json \
      --ref path/to/reference_image.nii.gz \
      --out_prefix path/to/output/CASE01 \
      --sigma_mm 2.0 \
      --radius_mm 1.5

This will produce:
  CASE01_heatmaps_4D.nii.gz   (float32, [X,Y,Z,N])
  CASE01_landmarks_labels.nii.gz (int16, [X,Y,Z])
"""

import os
import json
import math
import argparse
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine

# ---------------------------
# Parsing landmarks from JSON
# ---------------------------

def load_landmarks_from_json(json_path):
    """
    Load landmarks (in mm) and their coordinate system from a JSON file.
    Supports:
      A) 3D Slicer Markups JSON (.mrk.json):
         {
           "markups": [{
             "coordinateSystem": "LPS" or "RAS",
             "controlPoints": [{"position": [x,y,z]}, ...]
           }]
         }
      B) Simple JSON list of points (assumed RAS unless --force_coord specified):
         [[x,y,z], [x,y,z], ...]
      C) Dict with "landmarks": [[x,y,z], ...] (assumed RAS unless --force_coord specified)

    Returns:
      points_mm: (N,3) ndarray of float (mm, in the coordinate system in 'coord')
      coord: str in {"RAS", "LPS"} indicating the coordinate system of points_mm
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Try 3D Slicer Markups JSON
    if isinstance(data, dict) and "markups" in data and isinstance(data["markups"], list) and len(data["markups"]) > 0:
        m = data["markups"][0]
        coord = m.get("coordinateSystem", "RAS").upper()
        if coord not in ("RAS", "LPS"):
            coord = "RAS"
        cps = m.get("controlPoints", [])
        pts = []
        for cp in cps:
            pos = cp.get("position", None)
            if pos is None:
                # Newer Slicer may use "position": {"value": [x,y,z], ...}
                if isinstance(cp.get("position", {}), dict):
                    pos = cp["position"].get("value", None)
            if pos is not None and len(pos) >= 3:
                pts.append([float(pos[0]), float(pos[1]), float(pos[2])])
        if len(pts) == 0:
            raise ValueError(f"No controlPoints with positions found in {json_path}")
        return np.asarray(pts, dtype=np.float64), coord

    # Try simple list
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
        pts = np.asarray(data, dtype=np.float64)
        return pts, "RAS"  # default unless overridden by --force_coord

    # Try dict with "landmarks"
    if isinstance(data, dict) and "landmarks" in data and isinstance(data["landmarks"], list):
        pts = np.asarray(data["landmarks"], dtype=np.float64)
        return pts, "RAS"

    raise ValueError(f"Unrecognized JSON structure in {json_path}")


# -----------------------------------
# Coordinate transforms and utilities
# -----------------------------------

def lps_to_ras(points_mm):
    """Convert LPS (mm) -> RAS (mm)."""
    pts = np.asarray(points_mm, dtype=np.float64)
    ras = pts.copy()
    ras[..., 0] *= -1.0  # L -> R
    ras[..., 1] *= -1.0  # P -> A
    # S stays S
    return ras

def world_ras_mm_to_vox(affine, points_ras_mm):
    """
    Convert RAS (mm) world coords to voxel indices (floating).
    affine: image affine mapping voxel -> RAS mm
    """
    inv_aff = np.linalg.inv(affine)
    pts = np.asarray(points_ras_mm, dtype=np.float64)
    vox = np.zeros_like(pts, dtype=np.float64)
    for i, p in enumerate(pts):
        vox[i, :] = apply_affine(inv_aff, p)
    return vox  # floating indices (i, j, k)

def clamp_bbox(lo, hi, shape):
    """Clamp inclusive bbox to image bounds."""
    lo_clamped = [max(0, lo[d]) for d in range(3)]
    hi_clamped = [min(shape[d]-1, hi[d]) for d in range(3)]
    return lo_clamped, hi_clamped


# -----------------------------------------
# Heatmap and label (sphere) map generation
# -----------------------------------------

def add_gaussian_channel(volume, center_vox, sigma_vox, max_trunc=3.0):
    """
    Add a Gaussian blob (exp(-0.5 * sum(((x-c)/sigma)^2))) into 'volume' in-place.
    volume: 3D float array (channel slice)
    center_vox: (3,) float voxel indices [i, j, k]
    sigma_vox: (3,) float sigma in voxels per axis
    max_trunc: bbox half-width in sigma to truncate computation window
    """
    D, H, W = volume.shape
    ci, cj, ck = center_vox
    si, sj, sk = sigma_vox

    # Define computation window in voxel coordinates
    ri = int(math.ceil(max_trunc * si))
    rj = int(math.ceil(max_trunc * sj))
    rk = int(math.ceil(max_trunc * sk))

    i0 = int(math.floor(ci)) - ri
    i1 = int(math.floor(ci)) + ri
    j0 = int(math.floor(cj)) - rj
    j1 = int(math.floor(cj)) + rj
    k0 = int(math.floor(ck)) - rk
    k1 = int(math.floor(ck)) + rk

    (i0, j0, k0), (i1, j1, k1) = clamp_bbox([i0, j0, k0], [i1, j1, k1], [D, H, W])

    if i0 > i1 or j0 > j1 or k0 > k1:
        return  # entirely out of bounds

    ii = np.arange(i0, i1+1, dtype=np.float32)[:, None, None]
    jj = np.arange(j0, j1+1, dtype=np.float32)[None, :, None]
    kk = np.arange(k0, k1+1, dtype=np.float32)[None, None, :]

    # Avoid division by zero
    si = max(si, 1e-6)
    sj = max(sj, 1e-6)
    sk = max(sk, 1e-6)

    di2 = ((ii - ci) / si) ** 2
    dj2 = ((jj - cj) / sj) ** 2
    dk2 = ((kk - ck) / sk) ** 2

    g = np.exp(-0.5 * (di2 + dj2 + dk2)).astype(np.float32)

    volume[i0:i1+1, j0:j1+1, k0:k1+1] = np.maximum(volume[i0:i1+1, j0:j1+1, k0:k1+1], g)


def add_sphere_label(volume, center_vox, radius_vox, class_id):
    """
    Draw a solid sphere (binary) of class_id into integer label volume.
    volume: 3D int array
    center_vox: (3,) float indices
    radius_vox: float radius in voxels (isotropic approx using i-j-k distances)
    class_id: int class label to write
    """
    D, H, W = volume.shape
    ci, cj, ck = center_vox
    r = float(radius_vox)
    ri = int(math.ceil(r))
    rj = int(math.ceil(r))
    rk = int(math.ceil(r))

    i0 = int(math.floor(ci)) - ri
    i1 = int(math.floor(ci)) + ri
    j0 = int(math.floor(cj)) - rj
    j1 = int(math.floor(cj)) + rj
    k0 = int(math.floor(ck)) - rk
    k1 = int(math.floor(ck)) + rk

    (i0, j0, k0), (i1, j1, k1) = clamp_bbox([i0, j0, k0], [i1, j1, k1], [D, H, W])
    if i0 > i1 or j0 > j1 or k0 > k1:
        return

    ii = np.arange(i0, i1+1, dtype=np.float32)[:, None, None]
    jj = np.arange(j0, j1+1, dtype=np.float32)[None, :, None]
    kk = np.arange(k0, k1+1, dtype=np.float32)[None, None, :]

    d2 = (ii - ci)**2 + (jj - cj)**2 + (kk - ck)**2
    mask = d2 <= (r**2 + 1e-6)

    sub = volume[i0:i1+1, j0:j1+1, k0:k1+1]
    # Overwrite background only; if overlaps, keep existing (first landmark wins)
    sub[(mask) & (sub == 0)] = class_id
    volume[i0:i1+1, j0:j1+1, k0:k1+1] = sub


# -------------
# Main routine
# -------------

def main():
    ap = argparse.ArgumentParser(description="Convert landmarks JSON to heatmaps and nnU-Net labels.")
    ap.add_argument("--json", required=True, help="Path to landmarks JSON (.mrk.json or simple JSON).")
    ap.add_argument("--ref", required=True, help="Path to reference image NIfTI (.nii.gz).")
    ap.add_argument("--out_prefix", required=True, help="Output prefix (directory + base name).")
    ap.add_argument("--sigma_mm", type=float, default=2.0, help="Gaussian sigma (mm). Default=2.0")
    ap.add_argument("--radius_mm", type=float, default=1.5, help="Sphere radius for labels (mm). Default=1.5")
    ap.add_argument("--force_coord", choices=["RAS", "LPS"], default=None,
                    help="Force input coordinate system if JSON doesn't specify or is ambiguous.")
    args = ap.parse_args()

    # Load landmarks
    pts_mm, coord = load_landmarks_from_json(args.json)
    if args.force_coord is not None:
        coord = args.force_coord.upper()

    # Load reference image
    ref_img = nib.load(args.ref)
    affine = ref_img.affine
    shape = ref_img.shape[:3]
    zooms = ref_img.header.get_zooms()[:3]  # voxel spacing (mm/voxel) along i,j,k

    # Convert landmarks to RAS mm if needed
    if coord == "LPS":
        pts_ras_mm = lps_to_ras(pts_mm)
    else:
        pts_ras_mm = pts_mm

    # Convert to voxel indices (float)
    pts_vox = world_ras_mm_to_vox(affine, pts_ras_mm)

    # Report any out-of-bounds points
    for idx, (vi, vj, vk) in enumerate(pts_vox):
        if not (0 <= vi < shape[0] and 0 <= vj < shape[1] and 0 <= vk < shape[2]):
            print(f"[WARN] Landmark {idx+1} voxel center out of bounds: ({vi:.2f}, {vj:.2f}, {vk:.2f}) for shape {shape}")

    # Prepare outputs
    nL = pts_vox.shape[0]
    # Heatmaps (4D: D,H,W,C)
    heatmaps = np.zeros((shape[0], shape[1], shape[2], nL), dtype=np.float32)
    # Labels (3D int)
    labels = np.zeros(shape, dtype=np.int16)

    # Sigma in voxels per axis
    sigma_vox = np.array([
        max(args.sigma_mm / zooms[0], 1e-6),
        max(args.sigma_mm / zooms[1], 1e-6),
        max(args.sigma_mm / zooms[2], 1e-6),
    ], dtype=np.float64)

    # Sphere radius in voxels (use average spacing for isotropic distance in i-j-k)
    # Alternatively, convert mm to vox per axis and use ellipsoids; we keep it simple.
    mean_vox_size = float(np.mean(zooms))
    radius_vox = max(args.radius_mm / mean_vox_size, 0.5)

    # Generate
    for li in range(nL):
        center = pts_vox[li]
        # Heatmap channel
        add_gaussian_channel(heatmaps[..., li], center, sigma_vox, max_trunc=3.0)
        # Segmentation label sphere
        add_sphere_label(labels, center, radius_vox, class_id=li+1)

    # Save heatmaps (copy header to keep spacing, set datatype float32)
    heat_hdr = ref_img.header.copy()
    heat_hdr.set_data_dtype(np.float32)
    heat_img = nib.Nifti1Image(heatmaps, affine=affine, header=heat_hdr)
    heat_path = f"{args.out_prefix}_heatmaps_4D.nii.gz"
    nib.save(heat_img, heat_path)

    # Save labels (int16)
    lab_hdr = ref_img.header.copy()
    lab_hdr.set_data_dtype(np.int16)
    lab_img = nib.Nifti1Image(labels, affine=affine, header=lab_hdr)
    lab_path = f"{args.out_prefix}_landmarks_labels.nii.gz"
    nib.save(lab_img, lab_path)

    print(f"[OK] Saved heatmaps: {heat_path}  (shape {heatmaps.shape}, float32)")
    print(f"[OK] Saved labels:   {lab_path}  (shape {labels.shape}, int16)")
    print(f"Voxel spacing (mm): {zooms} | sigma_mm={args.sigma_mm} -> sigma_vox={sigma_vox} | radius_mm={args.radius_mm} -> radius_vox≈{radius_vox:.2f}")

if __name__ == "__main__":
    main()