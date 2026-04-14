import os
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import json
from scipy.ndimage import zoom
import argparse

# ============================================================
# apply_transforms_to_mask.py  (fixed v3)
#
# Replays the CT preprocessing pipeline on a binary mask using
# nearest-neighbour interpolation at every resampling step.
#
# Usage – single patient:
#   python apply_transforms_to_mask.py single \
#       --mask   /path/to/mask.nii.gz \
#       --json   /path/to/patient_transform_log.json \
#       --outdir /path/to/output_dir \
#       --patient_id sub36_pituitary
#
# Usage – batch:
#   python apply_transforms_to_mask.py batch \
#       --mask_dir    /path/masks/ \
#       --json_dir    /path/transform_logs/ \
#       --outdir      /path/output/ \
#       --mask_suffix _mask.nii.gz
# ============================================================


def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_step(transform_data, step_number):
    for t in transform_data["transformations"]:
        if t["step"] == step_number:
            return t
    return None


def _save_intermediate(data, affine, header, out_dir, pid, tag):
    path = os.path.join(out_dir, f"{pid}_mask_{tag}.nii.gz")
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), path)
    print(f"    -> Intermediate saved: {path}")


# ------------------------------------------------------------------ steps ----

def step1_intensity_clipping(data, params):
    print("  Step 1 - Intensity clipping: no-op for mask.")
    return data


def step4_roi_cropping(data, affine, params):
    """
    Crop mask to the same physical ROI box that was applied to the CT.

    Key fix vs previous version
    ---------------------------
    The mask may have a DIFFERENT voxel spacing than the CT.
    The JSON stores the crop geometry in mm (origin + CT-voxel dimensions).
    We must:
      1. Convert the crop box origin from mm -> mask voxels  (using mask affine)
      2. Convert the crop box physical size from mm -> mask voxels  (using mask spacing)
    That gives us the correct roi_start / roi_size in mask space.
    """
    print("  Step 4 - ROI cropping...")

    cropped_origin_mm  = np.array(params["cropped_origin_mm"],     dtype=float)  # [x,y,z] mm
    ct_cropped_dims    = np.array(params["cropped_dimensions"],    dtype=int)    # CT voxels
    ct_cropped_affine  = np.array(params["cropped_affine_matrix"], dtype=float)  # 4x4

    # Physical size of the crop box in mm (CT spacing * CT voxel dims)
    ct_spacing  = np.abs(np.diag(ct_cropped_affine[:3, :3]))
    roi_size_mm = ct_cropped_dims * ct_spacing

    # Mask voxel spacing
    mask_spacing = np.abs(np.diag(affine[:3, :3]))

    # Crop box size expressed in mask voxels
    roi_size_mask = np.round(roi_size_mm / mask_spacing).astype(int)

    # Crop box start in mask voxels
    inv_affine = np.linalg.inv(affine)
    roi_start  = np.round(
        nib.affines.apply_affine(inv_affine, cropped_origin_mm)
    ).astype(int)
    roi_end    = roi_start + roi_size_mask

    print(f"    Mask spacing (mm)       : {mask_spacing}")
    print(f"    CT   spacing (mm)       : {ct_spacing}")
    print(f"    ROI physical size (mm)  : {roi_size_mm}")
    print(f"    ROI in mask voxels      : {roi_size_mask}")
    print(f"    roi_start               : {roi_start}")
    print(f"    roi_end                 : {roi_end}")
    print(f"    Mask volume shape       : {data.shape}")

    roi        = np.zeros(roi_size_mask, dtype=data.dtype)
    data_shape = np.array(data.shape)

    data_start = np.maximum(roi_start, 0)
    data_end   = np.minimum(roi_end,   data_shape)

    if np.any(data_end <= data_start):
        print("    WARNING: crop box does not overlap with mask – output is all zeros.")
        new_affine        = affine.copy()
        new_affine[:3, 3] = cropped_origin_mm
        return roi, new_affine

    roi_offset = data_start - roi_start
    copy_shape = data_end   - data_start

    data_slices = tuple(slice(int(data_start[i]), int(data_end[i]))                      for i in range(3))
    roi_slices  = tuple(slice(int(roi_offset[i]),  int(roi_offset[i] + copy_shape[i]))   for i in range(3))

    roi[roi_slices] = data[data_slices]

    new_affine        = affine.copy()
    new_affine[:3, 3] = cropped_origin_mm

    print(f"    {data.shape}  ->  {roi.shape}")
    print(f"    Labels in output: {np.unique(roi)}")
    return roi, new_affine


def step5_resampling(data, affine, params):
    """
    Resample the cropped mask to new_spacing with nearest-neighbour
    interpolation so label values are never interpolated away.
    """
    print("  Step 5 - Resampling (nearest-neighbour)...")

    new_spacing  = list(params["new_spacing_mm"])
    target_dims  = list(params["resampled_dimensions"])   # [X, Y, Z] SimpleITK order

    mask_spacing = [abs(float(affine[i, i])) for i in range(3)]
    origin       = [float(affine[i, 3])      for i in range(3)]

    # numpy XYZ  ->  SimpleITK ZYX array
    sitk_img = sitk.GetImageFromArray(
        np.transpose(data, (2, 1, 0)).astype(np.float32)
    )
    sitk_img.SetSpacing(mask_spacing)
    sitk_img.SetOrigin(origin)

    resampled = sitk.Resample(
        sitk_img,
        target_dims,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,   # preserves label values
        sitk_img.GetOrigin(),
        new_spacing,
        sitk_img.GetDirection(),
        0,                          # background = 0
        sitk_img.GetPixelID(),
    )

    # SimpleITK ZYX  ->  numpy XYZ
    resampled_array = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))

    new_affine = affine.copy()
    for i in range(3):
        sign = np.sign(affine[i, i]) if affine[i, i] != 0 else 1.0
        new_affine[i, i]  = sign * new_spacing[i]
    new_affine[:3, 3] = list(resampled.GetOrigin())

    print(f"    {data.shape} spacing={mask_spacing}  ->  {resampled_array.shape} spacing={new_spacing}")
    return resampled_array, new_affine


def step6_origin_reset(data, affine, params):
    print("  Step 6 - Origin reset to (0, 0, 0)...")
    new_affine        = affine.copy()
    new_affine[:3, 3] = [0.0, 0.0, 0.0]
    return data, new_affine


def step7_orientation_standardization(data, affine, params):
    axes_flipped = params.get("axes_flipped", [])

    if not axes_flipped:
        print("  Step 7 - Orientation: no flip needed.")
        return data, affine

    print(f"  Step 7 - Orientation: flipping axes {axes_flipped}...")
    corrected = data.copy()
    for axis in axes_flipped:
        corrected = np.flip(corrected, axis=int(axis))

    target_signs     = np.array(params["target_signs"])
    corrected_affine = np.eye(4)
    for i in range(3):
        corrected_affine[i, i] = target_signs[i] * abs(affine[i, i])
    corrected_affine[3, 3] = 1.0

    return corrected, corrected_affine


def step8_padding(data, affine, params):
    print("  Step 8 - Padding to target shape...")
    pad_width = [(int(pw[0]), int(pw[1])) for pw in params["pad_width"]]
    padded    = np.pad(data, pad_width, mode="constant", constant_values=0)
    print(f"    {data.shape}  ->  {padded.shape}")
    return padded, affine


def step9_final_resampling(data, affine, params):
    print("  Step 9 - Final resampling to 256^3 (nearest-neighbour zoom)...")
    zoom_factors = params["zoom_factors"]
    target_dims  = params["target_dimensions"]

    resampled = zoom(data, zoom_factors, order=0)   # order=0 = nearest-neighbour

    final_affine = affine.copy()
    for i in range(3):
        final_affine[i, i] *= data.shape[i] / target_dims[i]

    print(f"    {data.shape}  ->  {resampled.shape}")
    return resampled, final_affine


# ------------------------------------------------------------------ main -----

def apply_transforms_to_mask(mask_path, json_path, output_dir, patient_id=None):
    print(f"\n{'='*60}")
    print("Applying CT transforms to binary mask")
    print(f"  Mask : {mask_path}")
    print(f"  JSON : {json_path}")
    print(f"{'='*60}\n")

    transform_data = load_json(json_path)
    pid = patient_id or transform_data.get("patient_id", "patient")

    os.makedirs(output_dir, exist_ok=True)

    mask_img  = nib.load(mask_path)
    mask_data = np.round(mask_img.get_fdata()).astype(np.uint8)
    affine    = mask_img.affine.copy()
    header    = mask_img.header

    print(f"Mask shape  : {mask_data.shape}")
    print(f"Mask labels : {np.unique(mask_data)}")
    print(f"Mask spacing: {np.abs(np.diag(affine[:3,:3]))}")
    print(f"Mask origin : {affine[:3,3]}")

    p = get_step(transform_data, 1)
    if p:
        mask_data = step1_intensity_clipping(mask_data, p["parameters"])

    p = get_step(transform_data, 4)
    if p:
        mask_data, affine = step4_roi_cropping(mask_data, affine, p["parameters"])
        _save_intermediate(mask_data, affine, header, output_dir, pid, "step4_cropped")

    p = get_step(transform_data, 5)
    if p:
        mask_data, affine = step5_resampling(mask_data, affine, p["parameters"])
        _save_intermediate(mask_data, affine, header, output_dir, pid, "step5_resampled")

    p = get_step(transform_data, 6)
    if p:
        mask_data, affine = step6_origin_reset(mask_data, affine, p["parameters"])

    p = get_step(transform_data, 7)
    if p:
        mask_data, affine = step7_orientation_standardization(mask_data, affine, p["parameters"])
        _save_intermediate(mask_data, affine, header, output_dir, pid, "step7_orientation")

    p = get_step(transform_data, 8)
    if p:
        mask_data, affine = step8_padding(mask_data, affine, p["parameters"])

    p = get_step(transform_data, 9)
    if p:
        mask_data, affine = step9_final_resampling(mask_data, affine, p["parameters"])

    final_path = os.path.join(output_dir, f"{pid}_mask_resampled_256.nii.gz")
    nib.save(nib.Nifti1Image(mask_data.astype(np.uint8), affine, header), final_path)

    print(f"\nFinal mask saved -> {final_path}")
    print(f"  Shape  : {mask_data.shape}")
    print(f"  Labels : {np.unique(mask_data)}")
    print(f"  Spacing: {np.abs(np.diag(affine[:3,:3]))}")
    return final_path


# ------------------------------------------------------------------ batch ----

def batch_apply(mask_dir, json_dir, output_dir,
                mask_suffix="_mask.nii.gz", json_suffix="_transform_log.json"):
    json_files = [f for f in os.listdir(json_dir) if f.endswith(json_suffix)]
    print(f"Found {len(json_files)} transform log(s).")

    for jf in sorted(json_files):
        pid         = jf.replace(json_suffix, "")
        mask_path   = os.path.join(mask_dir, f"{pid}{mask_suffix}")
        json_path   = os.path.join(json_dir, jf)
        patient_out = os.path.join(output_dir, pid)

        if not os.path.exists(mask_path):
            print(f"[SKIP] Mask not found for {pid}: {mask_path}")
            continue
        try:
            apply_transforms_to_mask(mask_path, json_path, patient_out, pid)
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")


# ------------------------------------------------------------------ CLI ------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply CT preprocessing transforms to a binary mask."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("single")
    s.add_argument("--mask",       required=True)
    s.add_argument("--json",       required=True)
    s.add_argument("--outdir",     required=True)
    s.add_argument("--patient_id", default=None)

    b = sub.add_parser("batch")
    b.add_argument("--mask_dir",    required=True)
    b.add_argument("--json_dir",    required=True)
    b.add_argument("--outdir",      required=True)
    b.add_argument("--mask_suffix", default="_mask.nii.gz")
    b.add_argument("--json_suffix", default="_transform_log.json")

    args = parser.parse_args()

    if args.mode == "single":
        apply_transforms_to_mask(args.mask, args.json, args.outdir, args.patient_id)
    else:
        batch_apply(args.mask_dir, args.json_dir, args.outdir,
                    args.mask_suffix, args.json_suffix)