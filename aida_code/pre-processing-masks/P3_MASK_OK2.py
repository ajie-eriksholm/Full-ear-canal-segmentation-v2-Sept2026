import os
import numpy as np
import nibabel as nib
import json
from tqdm import tqdm
import argparse

# ============================================================
# P3_mask_pipeline.py
#
# Applies P3 (ROI crop + optional mirror + origin reset) to a
# binary mask, reading all geometry directly from the P3 JSON
# transform log. No recalculation needed — roi_start_voxel,
# roi_end_voxel and affine matrices are read straight from JSON.
#
# Uses nearest-neighbour logic (integer copy, no interpolation)
# so label values are never modified.
#
# Input mask: output of P2 mask pipeline
#             (*_mask_aligned.nii.gz)
#
# Outputs (mirroring P3 CT file naming exactly):
#   {pid}_right_ear_mask_cropped.nii.gz
#   {pid}_right_ear_mask_cropped_origin_reset.nii.gz
#   {pid}_left_ear_mask_cropped.nii.gz
#   {pid}_left_ear_mask_cropped_mirrored.nii.gz
#   {pid}_left_ear_mask_cropped_mirrored_origin_reset.nii.gz
#   Final_Cropped_Ears_Masks/{pid}_right_ear_mask.nii.gz
#   Final_Cropped_Ears_Masks/{pid}_left_ear_mask.nii.gz
#
# Usage – single patient:
#   python P3_mask_pipeline.py single \
#       --mask    /path/to/sub36_mask_aligned.nii.gz \
#       --p3json  /path/to/sub36_transform_log_P3.json \
#       --outdir  /path/to/output/ \
#       --patient_id sub36_pituitary \
#       --final_ears_dir /path/to/Final_Cropped_Ears_Masks/
#
# Usage – batch:
#   python P3_mask_pipeline.py batch \
#       --mask_dir       /path/masks/ \
#       --p3json_dir     /path/transform_logs/ \
#       --outdir         /path/output/ \
#       --mask_suffix    _mask_aligned.nii.gz \
#       --json_suffix    _transform_log_P3.json \
#       --final_ears_dir /path/Final_Cropped_Ears_Masks/
# ============================================================

MIRROR_AXIS = 0   # must match P3 CT config


# ------------------------------------------------------------------ helpers --

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_steps_by_operation(transform_data, operation):
    """Return all transformation dicts matching a given operation name."""
    return [t for t in transform_data["transformations"] if t["operation"] == operation]


def get_step_by_landmark(steps, landmark_id):
    """From a list of steps, return the one matching a specific landmark_id."""
    for s in steps:
        if s.get("parameters", {}).get("landmark_id") == landmark_id:
            return s
    return None


def _save(data, affine, header, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), path)
    print(f"    -> Saved: {path}")


# ------------------------------------------------------------------ crop step

def apply_crop(mask_data, mask_affine, crop_params):
    """
    Crop the mask using the exact voxel indices stored in the P3 JSON.

    Reads:
      roi_start_voxel      : [x, y, z] start of crop box in aligned volume voxels
      roi_end_voxel        : [x, y, z] end   of crop box in aligned volume voxels
      cropped_dimensions   : [x, y, z] expected output shape (== roi_end - roi_start)
      cropped_affine_matrix: 4x4 affine of the cropped CT volume
    """
    roi_start      = np.array(crop_params["roi_start_voxel"],    dtype=int)
    roi_end        = np.array(crop_params["roi_end_voxel"],      dtype=int)
    out_shape      = np.array(crop_params["cropped_dimensions"], dtype=int)
    ct_crop_affine = np.array(crop_params["cropped_affine_matrix"], dtype=float)

    lm_name        = crop_params["landmark_name"]
    lm_id          = crop_params["landmark_id"]

    print(f"\n  Cropping {lm_name} (LM {lm_id})...")
    print(f"    roi_start : {roi_start}")
    print(f"    roi_end   : {roi_end}")
    print(f"    out_shape : {out_shape}")

    # Allocate output filled with 0 (mask background)
    roi        = np.zeros(tuple(out_shape), dtype=mask_data.dtype)
    data_shape = np.array(mask_data.shape)

    # Clamp to valid mask range
    data_start = np.maximum(roi_start, 0)
    data_end   = np.minimum(roi_end,   data_shape)

    if np.any(data_end <= data_start):
        raise ValueError(
            f"Crop box [{roi_start} .. {roi_end}] does not overlap "
            f"with mask shape {tuple(mask_data.shape)}"
        )

    roi_offset = data_start - roi_start
    copy_shape = data_end   - data_start

    data_slices = tuple(slice(int(data_start[i]), int(data_end[i]))                   for i in range(3))
    roi_slices  = tuple(slice(int(roi_offset[i]),  int(roi_offset[i]+copy_shape[i]))  for i in range(3))

    roi[roi_slices] = mask_data[data_slices]

    # Build affine for the cropped mask:
    # Keep the mask's own spacing/signs but use the CT crop origin (stored in JSON)
    cropped_affine        = mask_affine.copy()
    cropped_affine[:3, 3] = ct_crop_affine[:3, 3]   # origin in mm from JSON

    print(f"    Labels in crop: {np.unique(roi)}")
    return roi, cropped_affine


# ------------------------------------------------------------------ main -----

def process_single(mask_path, p3json_path, output_dir,
                   patient_id=None, final_ears_dir=None):

    print(f"\n{'='*60}")
    print(f"P3 mask pipeline")
    print(f"  Mask   : {mask_path}")
    print(f"  P3 JSON: {p3json_path}")
    print(f"{'='*60}")

    p3_data = load_json(p3json_path)
    pid     = patient_id or p3_data.get("scan_name", "patient")

    os.makedirs(output_dir, exist_ok=True)
    if final_ears_dir:
        os.makedirs(final_ears_dir, exist_ok=True)

    # Load aligned mask (output of P2 mask pipeline)
    mask_img  = nib.load(mask_path)
    mask_data = np.round(mask_img.get_fdata()).astype(np.uint8)
    affine    = mask_img.affine.copy()
    header    = mask_img.header

    print(f"\nMask shape  : {mask_data.shape}")
    print(f"Mask labels : {np.unique(mask_data)}")
    print(f"Mask spacing: {np.abs(np.diag(affine[:3,:3]))}")

    # Gather all crop steps from JSON (one per ear)
    crop_steps   = get_steps_by_operation(p3_data, "roi_cropping")
    mirror_steps = get_steps_by_operation(p3_data, "axis_mirroring")
    reset_steps  = get_steps_by_operation(p3_data, "origin_reset")

    # Process each ear crop recorded in the JSON
    for crop_step in crop_steps:
        params  = crop_step["parameters"]
        lm_id   = params["landmark_id"]
        lm_name = params["landmark_name"]

        # ── Crop ────────────────────────────────────────────────────────────
        try:
            cropped, cropped_affine = apply_crop(mask_data, affine, params)
        except Exception as e:
            print(f"  ERROR cropping {lm_name}: {e}")
            continue

        crop_path = os.path.join(output_dir, f"{pid}_{lm_name}_mask_cropped.nii.gz")
        _save(cropped, cropped_affine, header, crop_path)

        data_for_reset   = cropped
        affine_for_reset = cropped_affine.copy()

        # ── Mirror (left ear only, lm_id == 11) ─────────────────────────────
        mirror_step = get_step_by_landmark(mirror_steps, lm_id)
        if mirror_step is not None:
            flip_axis = int(mirror_step["parameters"].get("flip_axis", MIRROR_AXIS))
            print(f"  Mirroring {lm_name} along axis {flip_axis}...")

            mirrored        = np.flip(cropped, axis=flip_axis).copy()
            mirrored_affine = cropped_affine.copy()   # affine unchanged (matches P3 CT)

            mirror_path = os.path.join(
                output_dir, f"{pid}_{lm_name}_mask_cropped_mirrored.nii.gz"
            )
            _save(mirrored, mirrored_affine, header, mirror_path)

            data_for_reset   = mirrored
            affine_for_reset = mirrored_affine.copy()

        # ── Origin reset ────────────────────────────────────────────────────
        reset_step = get_step_by_landmark(reset_steps, lm_id)
        if reset_step is not None:
            new_origin_mm = reset_step["parameters"].get("new_origin_mm", [0.0, 0.0, 0.0])
        else:
            new_origin_mm = [0.0, 0.0, 0.0]   # default: reset to zero

        print(f"  Resetting origin to {new_origin_mm}...")
        origin_reset_affine        = affine_for_reset.copy()
        origin_reset_affine[:3, 3] = new_origin_mm

        if mirror_step is not None:
            reset_fname = f"{pid}_{lm_name}_mask_cropped_mirrored_origin_reset.nii.gz"
        else:
            reset_fname = f"{pid}_{lm_name}_mask_cropped_origin_reset.nii.gz"

        reset_path = os.path.join(output_dir, reset_fname)
        _save(data_for_reset, origin_reset_affine, header, reset_path)

        # ── Copy to centralised Final_Cropped_Ears_Masks/ folder ────────────
        if final_ears_dir:
            final_path = os.path.join(final_ears_dir, f"{pid}_{lm_name}_mask.nii.gz")
            _save(data_for_reset, origin_reset_affine, header, final_path)

    print(f"\nDone: {pid}")


# ------------------------------------------------------------------ batch ----

def batch_process(mask_dir, p3json_dir, output_dir,
                  mask_suffix="_mask_aligned.nii.gz",
                  json_suffix="_transform_log_P3.json",
                  final_ears_dir=None):

    json_files = [f for f in os.listdir(p3json_dir) if f.endswith(json_suffix)]
    print(f"Found {len(json_files)} P3 JSON file(s).")

    for jf in tqdm(sorted(json_files), desc="Patients"):
        pid         = jf.replace(json_suffix, "")
        mask_path   = os.path.join(mask_dir,   f"{pid}{mask_suffix}")
        json_path   = os.path.join(p3json_dir, jf)
        patient_out = os.path.join(output_dir, pid)

        if not os.path.exists(mask_path):
            print(f"[SKIP] Mask not found for {pid}: {mask_path}")
            continue

        try:
            process_single(mask_path, json_path, patient_out, pid, final_ears_dir)
        except Exception as e:
            print(f"[ERROR] {pid}: {e}")


# ------------------------------------------------------------------ CLI ------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply P3 ROI crop + mirror + origin reset to a binary mask."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("single", help="Process one patient.")
    s.add_argument("--mask",           required=True,  help="Path to aligned mask .nii.gz")
    s.add_argument("--p3json",         required=True,  help="Path to *_transform_log_P3.json")
    s.add_argument("--outdir",         required=True,  help="Output directory")
    s.add_argument("--patient_id",     default=None,   help="Override patient ID")
    s.add_argument("--final_ears_dir", default=None,   help="Centralised folder for final masks")

    b = sub.add_parser("batch", help="Process all patients in a directory.")
    b.add_argument("--mask_dir",       required=True)
    b.add_argument("--p3json_dir",     required=True)
    b.add_argument("--outdir",         required=True)
    b.add_argument("--mask_suffix",    default="_mask_aligned.nii.gz")
    b.add_argument("--json_suffix",    default="_transform_log_P3.json")
    b.add_argument("--final_ears_dir", default=None)

    args = parser.parse_args()

    if args.mode == "single":
        process_single(
            mask_path=args.mask,
            p3json_path=args.p3json,
            output_dir=args.outdir,
            patient_id=args.patient_id,
            final_ears_dir=args.final_ears_dir,
        )
    else:
        batch_process(
            mask_dir=args.mask_dir,
            p3json_dir=args.p3json_dir,
            output_dir=args.outdir,
            mask_suffix=args.mask_suffix,
            json_suffix=args.json_suffix,
            final_ears_dir=args.final_ears_dir,
        )