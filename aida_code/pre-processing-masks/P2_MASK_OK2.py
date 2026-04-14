import os
import sys
import numpy as np
import nibabel as nib
import json
from datetime import datetime
from scipy.ndimage import affine_transform
from tqdm import tqdm
import argparse

# ============================================================
# P2_P3_mask_pipeline.py
#
# Applies P2 (Frankfort plane rotation) and P3 (ROI crop +
# optional mirror + origin reset) to a binary mask, using
# nearest-neighbour interpolation to preserve label values.
#
# Reads:
#   - *_transform_log_P2.json   (produced by P2)
#   - mask input file           (output of P1 mask pipeline,
#                                e.g. *_mask_resampled_256.nii.gz)
#
# Produces (matching P3 CT outputs):
#   - {pid}_{ear}_mask_cropped.nii.gz
#   - {pid}_left_ear_mask_cropped_mirrored.nii.gz          (left ear only)
#   - {pid}_{ear}_mask_cropped[_mirrored]_origin_reset.nii.gz
#   - {pid}_{ear}_mask.nii.gz  →  Final_Cropped_Ears_Masks/
#
# Usage – single patient:
#   python P2_P3_mask_pipeline.py single \
#       --mask   /path/to/sub36_pituitary_mask_resampled_256.nii.gz \
#       --p2json /path/to/sub36_pituitary_transform_log_P2.json \
#       --outdir /path/to/output/ \
#       --patient_id sub36_pituitary
#
# Usage – batch:
#   python P2_P3_mask_pipeline.py batch \
#       --mask_dir    /path/masks/ \
#       --p2json_dir  /path/transform_logs/ \
#       --outdir      /path/output/ \
#       --mask_suffix _mask_resampled_256.nii.gz \
#       --json_suffix _transform_log_P2.json
# ============================================================


# ------------------------------------------------------------------ config ---
# Matches P3 CT configuration exactly
ROI_SIZE      = [90, 90, 90]   # voxels (x, y, z)
MIRROR_AXIS   = 0              # axis for left-ear mirror

# (lm_index_in_array, lm_id, name, offset_mm, is_left_ear)
# is_left_ear=True  → use offset as-is        (right ear in CT naming)
# is_left_ear=False → flip x of offset        (left ear  in CT naming)
LANDMARKS_TO_CROP = [
    (2, 10, 'right_ear', [20, 10, -5], True),
    (3, 11, 'left_ear',  [20, 10, -5], False),
]

LANDMARK_IDS = [8, 9, 10, 11, 12, 13]


# ------------------------------------------------------------------ helpers --

def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_step(transform_data, step_number):
    for t in transform_data["transformations"]:
        if t["step"] == step_number:
            return t
    return None


def voxel_to_world(voxel, affine):
    return (affine @ np.append(voxel, 1.0))[:3]


def world_to_voxel(world, affine):
    spacing = np.array([abs(affine[i, i]) for i in range(3)])
    origin  = affine[:3, 3]
    return (world - origin) / spacing


def _save(data, affine, header, path):
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine, header), path)
    print(f"    -> Saved: {path}")


# ------------------------------------------------------------------ P2 step --

def apply_p2_rotation(mask_data, affine, p2_params):
    """
    Apply the Frankfort plane rigid rotation stored in the P2 JSON to the mask.

    The CT pipeline uses scipy.ndimage.affine_transform with order=1.
    For the mask we use order=0 (nearest-neighbour) so label values are
    never interpolated.

    Math
    ----
    For each output voxel o, we need the corresponding input voxel i:
        world_o  = affine @ o_h
        world_i  = R^-1 @ (world_o - center) + center
        i        = affine^-1 @ world_i

    scipy.ndimage.affine_transform(input, matrix, offset) computes:
        i = matrix @ o + offset   (all in voxel space)

    So we build matrix and offset in voxel space.
    """
    print("  P2 – Frankfort plane rotation (nearest-neighbour)...")

    R        = np.array(p2_params["rotation_matrix"])       # 3x3, world-space
    center   = np.array(p2_params["rotation_center_mm"])    # mm
    shape    = mask_data.shape

    # Voxel spacing (assume isotropic diagonal affine)
    spacing  = np.array([abs(affine[i, i]) for i in range(3)])
    origin   = affine[:3, 3]

    # We need R^-1 (inverse rotation = transpose for orthogonal matrices)
    R_inv = R.T

    # Transform in voxel space:
    #   i = (1/spacing) * (R_inv @ (spacing * o + origin - center) + center - origin)
    # Expand:
    #   i = (1/spacing) * R_inv * spacing * o
    #     + (1/spacing) * (R_inv @ (origin - center) + center - origin)
    #
    # scipy convention: i = matrix @ o + offset

    # Matrix in voxel space (maps output voxel → input voxel)
    M_world = R_inv                                       # world → world
    # Scale: world = spacing * voxel + origin  →  voxel = (world - origin) / spacing
    # M_vox[i,j] = (1/spacing[i]) * M_world[i,j] * spacing[j]
    M_vox = (1.0 / spacing[:, None]) * M_world * spacing[None, :]

    # Offset in voxel space
    t_world  = R_inv @ (origin - center) + center - origin   # world translation
    t_vox    = t_world / spacing

    print(f"    Rotation angles (deg): {p2_params['rotation_angles_xyz_degrees']}")
    print(f"    Rotation center (mm) : {center}")
    print(f"    Input mask shape     : {shape}")

    rotated = affine_transform(
        mask_data.astype(np.float32),
        M_vox,
        offset=t_vox,
        output_shape=shape,
        order=0,           # nearest-neighbour → preserves label values
        mode='constant',
        cval=0,            # background = 0 for masks
        prefilter=False,
    )

    rotated = np.round(rotated).astype(np.uint8)
    print(f"    Output mask shape    : {rotated.shape}")
    print(f"    Labels after rotation: {np.unique(rotated)}")
    return rotated


# ------------------------------------------------------------------ P3 step --

def crop_roi_around_landmark(mask_data, landmark_world, affine, roi_size, offset_mm=None):
    """
    Identical logic to P3 CT crop_roi_around_landmark but:
      - fills with 0 instead of -1000
      - returns same tuple as CT version for drop-in compatibility
    """
    if offset_mm is None:
        offset_mm = np.zeros(3)
    else:
        offset_mm = np.array(offset_mm)

    center_world   = landmark_world + offset_mm
    landmark_voxel = np.round(world_to_voxel(center_world, affine)).astype(int)

    roi_size  = np.array(roi_size)
    half_size = roi_size // 2
    roi_start = landmark_voxel - half_size
    roi_end   = landmark_voxel + half_size
    for i in range(3):
        if roi_size[i] % 2 == 1:
            roi_end[i] += 1

    roi        = np.zeros(tuple(roi_size), dtype=mask_data.dtype)   # 0 = background
    data_shape = np.array(mask_data.shape)
    data_start = np.maximum(roi_start, 0)
    data_end   = np.minimum(roi_end,   data_shape)

    if np.any(data_end <= data_start):
        raise ValueError(
            f"ROI does not overlap with mask. "
            f"roi_start={roi_start}, roi_end={roi_end}, mask_shape={data_shape}"
        )

    roi_offset = data_start - roi_start
    copy_shape = data_end   - data_start

    data_slices = tuple(slice(int(data_start[i]), int(data_end[i]))                    for i in range(3))
    roi_slices  = tuple(slice(int(roi_offset[i]),  int(roi_offset[i]+copy_shape[i]))   for i in range(3))
    roi[roi_slices] = mask_data[data_slices]

    new_origin     = voxel_to_world(roi_start.astype(float), affine)
    cropped_affine = affine.copy()
    cropped_affine[:3, 3] = new_origin

    return roi, cropped_affine, roi_start, roi_end, center_world


# ------------------------------------------------------------------ main -----

def process_single(mask_path, p2json_path, output_dir, patient_id=None,
                   final_ears_dir=None):
    print(f"\n{'='*60}")
    print(f"P2+P3 mask pipeline")
    print(f"  Mask  : {mask_path}")
    print(f"  P2 JSON: {p2json_path}")
    print(f"{'='*60}\n")

    p2_data    = load_json(p2json_path)
    pid        = patient_id or p2_data.get("scan_name", "patient")

    os.makedirs(output_dir, exist_ok=True)
    if final_ears_dir:
        os.makedirs(final_ears_dir, exist_ok=True)

    # Load mask
    mask_img  = nib.load(mask_path)
    mask_data = np.round(mask_img.get_fdata()).astype(np.uint8)
    affine    = mask_img.affine.copy()
    header    = mask_img.header

    print(f"Mask shape  : {mask_data.shape}")
    print(f"Mask labels : {np.unique(mask_data)}")
    print(f"Mask spacing: {np.abs(np.diag(affine[:3,:3]))}")

    # ── P2: Frankfort plane rotation ──────────────────────────────────────────
    step10 = get_step(p2_data, 10)
    if step10 is None:
        raise ValueError("Step 10 (frankfort_plane_alignment) not found in P2 JSON.")

    mask_rotated = apply_p2_rotation(mask_data, affine, step10["parameters"])
    # Affine is unchanged by rotation (origin stays, shape stays, spacing stays)
    rotated_affine = affine.copy()

    # Save rotated mask (equivalent of *_aligned.nii.gz for CT)
    rotated_path = os.path.join(output_dir, f"{pid}_mask_aligned.nii.gz")
    _save(mask_rotated, rotated_affine, header, rotated_path)

    # ── Recover aligned landmarks from P2 JSON ────────────────────────────────
    step10_aligned = step10.get("aligned_landmarks", {})
    aligned_landmarks = np.zeros((len(LANDMARK_IDS), 3))
    for i, lm_id in enumerate(LANDMARK_IDS):
        key = f"landmark_{lm_id}"
        if key in step10_aligned:
            aligned_landmarks[i] = step10_aligned[key]["aligned_world_mm"]
        else:
            print(f"  WARNING: aligned landmark {lm_id} not found in P2 JSON.")

    # ── P3: ROI crop + mirror + origin reset ──────────────────────────────────
    for lm_idx, lm_id, lm_name, offset_mm, is_left_ear in LANDMARKS_TO_CROP:
        print(f"\n  Cropping ROI for Landmark {lm_id} ({lm_name})...")

        landmark_world  = aligned_landmarks[lm_idx]
        applied_offset  = np.array(offset_mm)
        if not is_left_ear:
            applied_offset[0] = -applied_offset[0]   # flip x for right ear

        print(f"    Landmark world (mm): {landmark_world}")
        print(f"    Applied offset (mm): {applied_offset}")

        try:
            cropped, cropped_affine, roi_start, roi_end, center_world = \
                crop_roi_around_landmark(
                    mask_rotated, landmark_world, rotated_affine,
                    ROI_SIZE, offset_mm=applied_offset
                )

            print(f"    Cropped shape : {cropped.shape}")
            print(f"    Labels in crop: {np.unique(cropped)}")

            # Save raw crop
            crop_path = os.path.join(output_dir, f"{pid}_{lm_name}_mask_cropped.nii.gz")
            _save(cropped, cropped_affine, header, crop_path)

            # ── Mirror left ear (lm_id == 11) ────────────────────────────────
            data_for_reset  = cropped
            affine_for_reset = cropped_affine.copy()

            if lm_id == 11:
                print(f"    Mirroring left ear along axis {MIRROR_AXIS}...")
                mirrored        = np.flip(cropped, axis=MIRROR_AXIS).copy()
                mirrored_affine = cropped_affine.copy()

                mirror_path = os.path.join(
                    output_dir, f"{pid}_{lm_name}_mask_cropped_mirrored.nii.gz"
                )
                _save(mirrored, mirrored_affine, header, mirror_path)

                data_for_reset   = mirrored
                affine_for_reset = mirrored_affine.copy()

            # ── Origin reset ─────────────────────────────────────────────────
            print(f"    Resetting origin to (0, 0, 0)...")
            origin_reset_affine        = affine_for_reset.copy()
            origin_reset_affine[:3, 3] = [0.0, 0.0, 0.0]

            if lm_id == 11:
                reset_name = f"{pid}_{lm_name}_mask_cropped_mirrored_origin_reset.nii.gz"
            else:
                reset_name = f"{pid}_{lm_name}_mask_cropped_origin_reset.nii.gz"

            reset_path = os.path.join(output_dir, reset_name)
            _save(data_for_reset, origin_reset_affine, header, reset_path)

            # ── Copy to centralised Final_Cropped_Ears_Masks/ folder ─────────
            if final_ears_dir:
                final_name = f"{pid}_{lm_name}_mask.nii.gz"
                final_path = os.path.join(final_ears_dir, final_name)
                _save(data_for_reset, origin_reset_affine, header, final_path)

        except Exception as e:
            print(f"    ERROR cropping {lm_name}: {e}")

    print(f"\nDone: {pid}")


# ------------------------------------------------------------------ batch ----

def batch_process(mask_dir, p2json_dir, output_dir,
                  mask_suffix="_mask_resampled_256.nii.gz",
                  json_suffix="_transform_log_P2.json",
                  final_ears_dir=None):

    json_files = [f for f in os.listdir(p2json_dir) if f.endswith(json_suffix)]
    print(f"Found {len(json_files)} P2 JSON file(s).")

    for jf in tqdm(sorted(json_files), desc="Patients"):
        pid        = jf.replace(json_suffix, "")
        mask_path  = os.path.join(mask_dir,   f"{pid}{mask_suffix}")
        json_path  = os.path.join(p2json_dir, jf)
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
        description="Apply P2 rotation + P3 ROI crop to a binary mask."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # single
    s = sub.add_parser("single", help="Process one patient.")
    s.add_argument("--mask",        required=True,  help="Path to mask .nii.gz")
    s.add_argument("--p2json",      required=True,  help="Path to *_transform_log_P2.json")
    s.add_argument("--outdir",      required=True,  help="Output directory for this patient")
    s.add_argument("--patient_id",  default=None,   help="Override patient ID")
    s.add_argument("--final_ears_dir", default=None,
                   help="Centralised folder for final ear masks (optional)")

    # batch
    b = sub.add_parser("batch", help="Process all patients in a directory.")
    b.add_argument("--mask_dir",    required=True)
    b.add_argument("--p2json_dir",  required=True)
    b.add_argument("--outdir",      required=True)
    b.add_argument("--mask_suffix", default="_mask_resampled_256.nii.gz")
    b.add_argument("--json_suffix", default="_transform_log_P2.json")
    b.add_argument("--final_ears_dir", default=None)

    args = parser.parse_args()

    if args.mode == "single":
        process_single(
            mask_path=args.mask,
            p2json_path=args.p2json,
            output_dir=args.outdir,
            patient_id=args.patient_id,
            final_ears_dir=args.final_ears_dir,
        )
    else:
        batch_process(
            mask_dir=args.mask_dir,
            p2json_dir=args.p2json_dir,
            output_dir=args.outdir,
            mask_suffix=args.mask_suffix,
            json_suffix=args.json_suffix,
            final_ears_dir=args.final_ears_dir,
        )