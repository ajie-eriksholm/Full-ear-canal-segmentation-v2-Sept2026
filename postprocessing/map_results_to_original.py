"""
Map pipeline results (masks, STL, markups) from the ear-local result space back
to the ORIGINAL scan space, using the P1/P2/P3 transform logs.

The whole forward chain result->original is a composition of voxel-space affine
operations, so per ear it collapses to a single 4x4 matrix M (result voxel ->
original voxel):

    result 128^3 --zoom--> ear 90^3 --(unmirror if left)+crop_offset-->
    aligned 256^3 --inv_rot--> P1out 256^3 --zoom--> padded 540^3 -->(identity)
    resampled --affine--> cropped --affine--> original

Points (markup landmarks, STL vertices) are mapped result_world -> original_world.
Masks are resampled onto the original CT grid (nearest neighbour).

Run --validate first: it maps the canal-mask centroid to original world and
compares against the ear centroid P1 recorded in original space.
"""

import os
import re
import json
import struct
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')


# ----------------------------------------------------------------------------
# small IO helpers
# ----------------------------------------------------------------------------
def strip_nifti_extension(name):
    for ext in NIFTI_EXTENSIONS:
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def find_nifti(path_no_ext):
    for ext in NIFTI_EXTENSIONS:
        if os.path.exists(path_no_ext + ext):
            return path_no_ext + ext
    return None


def load_json(path):
    with open(path) as f:
        return json.load(f)


def homog(mat3x3, t):
    M = np.eye(4)
    M[:3, :3] = mat3x3
    M[:3, 3] = t
    return M


def scale_matrix(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])


# ----------------------------------------------------------------------------
# transform log lookups
# ----------------------------------------------------------------------------
def get_step(log, operation, landmark_name=None):
    """Return the first transformation dict matching operation (and name)."""
    for t in log.get("transformations", []):
        if t.get("operation") != operation:
            continue
        if landmark_name is not None and t.get("parameters", {}).get("landmark_name") != landmark_name:
            continue
        return t
    return None


def build_result_to_original_voxel(pid, side, p1_log, p2_log, p3_log,
                                    processed_dir, result_affine, result_shape,
                                    rotation_center_scale=1.0):
    """Compose the 4x4 matrix mapping result voxel -> original scan voxel.

    processed_dir holds the intermediate NIfTI files for this patient.
    rotation_center_scale multiplies the logged rotation center (1.0 if the
    center is stored in voxel units; ~1/spacing if stored in mm).
    """
    ear_name = f"{side}_ear"

    # --- P3: locate this ear's crop entry, mirror flag, dims ---
    crop = get_step(p3_log, "roi_cropping", landmark_name=ear_name)
    if crop is None:
        raise ValueError(f"P3 roi_cropping for {ear_name} not found")
    roi_start = np.array(crop["parameters"]["roi_start_voxel"], dtype=float)
    ear_dims = np.array(crop["parameters"]["cropped_dimensions"], dtype=float)
    lm_id = crop["parameters"]["landmark_id"]
    mirrored = get_step(p3_log, "axis_mirroring", landmark_name=ear_name) is not None
    mirror_axis = 0
    if mirrored:
        mirror_axis = get_step(p3_log, "axis_mirroring", landmark_name=ear_name)["parameters"]["flip_axis"]

    # --- A1: result 128^3 -> ear 90^3 (P4 zoom, inverse) ---
    s = ear_dims / np.array(result_shape, dtype=float)
    A1 = scale_matrix(s[0], s[1], s[2])

    # --- A2: ear voxel -> aligned 256 voxel (unmirror if needed, then crop offset) ---
    A2 = np.eye(4)
    if mirrored:
        m = np.eye(4)
        m[mirror_axis, mirror_axis] = -1.0
        m[mirror_axis, 3] = ear_dims[mirror_axis] - 1.0  # i' = (N-1) - i
        A2 = m @ A2
    A2 = homog(np.eye(3), roi_start) @ A2

    # --- A3: aligned 256 voxel -> P1-out 256 voxel (inverse of P2 rotation) ---
    align = get_step(p2_log, "frankfort_plane_alignment")
    R = np.array(align["parameters"]["rotation_matrix"], dtype=float)
    center = np.array(align["parameters"]["rotation_center_mm"], dtype=float) * rotation_center_scale
    inv_rot = np.linalg.inv(R)
    offset = center - inv_rot @ center
    A3 = homog(inv_rot, offset)

    # --- A4: P1-out 256 voxel -> padded 540 voxel (P1 final_resampling, inverse) ---
    fr = get_step(p1_log, "final_resampling")["parameters"]
    zoom = np.array(fr["zoom_factors"], dtype=float)  # 256 = zoom * 540
    A4 = scale_matrix(1.0 / zoom[0], 1.0 / zoom[1], 1.0 / zoom[2])

    # --- A5: padded 540 voxel -> resampled voxel (undo padding; pad was 0-before) ---
    A5 = np.eye(4)  # pad_width [before,after] has before=0 for all axes

    # --- A5b: undo P1 Step 7 orientation flip (np.flip on the pre-pad grid). ---
    # P1 flips the voxel data on the reversed axes but logs only a diagonal affine,
    # so the flip must be inverted explicitly here or flipped scans land mirrored.
    A5b = np.eye(4)
    ori = get_step(p1_log, "orientation_standardization")
    axes_flipped = (ori or {}).get("parameters", {}).get("axes_flipped", []) or []
    if axes_flipped:
        pad = get_step(p1_log, "padding")["parameters"]
        pre_pad_dims = np.array(pad["original_dimensions"], dtype=float)  # shifted/resampled grid
        for a in axes_flipped:
            A5b[a, a] = -1.0
            A5b[a, 3] = pre_pad_dims[a] - 1.0  # i' = (N-1) - i

    # --- A6: resampled voxel -> cropped voxel (share world; use file affines) ---
    resampled = find_nifti(os.path.join(processed_dir, f"{pid}_CT_resampled"))
    cropped = find_nifti(os.path.join(processed_dir, f"{pid}_CT_cropped"))
    original = _find_original(p1_log)
    A_resampled = nib.load(resampled).affine
    A_cropped = nib.load(cropped).affine
    A_original = nib.load(original).affine
    A6 = np.linalg.inv(A_cropped) @ A_resampled

    # --- A7: cropped voxel -> original voxel (share world) ---
    A7 = np.linalg.inv(A_original) @ A_cropped

    M = A7 @ A6 @ A5b @ A5 @ A4 @ A3 @ A2 @ A1
    return M, A_original


def _find_original(p1_log):
    path = p1_log.get("original_scan_path")
    if path and os.path.exists(path):
        return path
    raise FileNotFoundError(f"original scan not found: {path}")


# ----------------------------------------------------------------------------
# point / mask mapping
# ----------------------------------------------------------------------------
def result_world_to_original_world(points_world, result_affine, M, A_original,
                                   input_is_lps=True):
    """Map Nx3 points from result world to original scan world (RAS)."""
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3).copy()
    if input_is_lps:
        pts[:, 0] *= -1
        pts[:, 1] *= -1  # LPS -> RAS
    inv_res = np.linalg.inv(result_affine)
    h = np.hstack([pts, np.ones((len(pts), 1))])
    res_vox = (inv_res @ h.T).T
    orig_vox = (M @ res_vox.T).T
    orig_world = (A_original @ orig_vox.T).T[:, :3]
    return orig_world


# ----------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------
def validate(pid, side, args):
    p1_log = load_json(os.path.join(args.logs_dir, f"{pid}_transform_log.json"))
    p2_log = load_json(os.path.join(args.logs_dir, f"{pid}_transform_log_P2.json"))
    p3_log = load_json(os.path.join(args.logs_dir, f"{pid}_transform_log_P3.json"))
    processed_dir = os.path.join(args.processed_dir, pid)

    mask_path = find_nifti(os.path.join(args.masks_dir, f"{pid}_{side}"))
    if mask_path is None:
        print(f"  [{pid}_{side}] no result mask, skip")
        return
    mask = nib.load(mask_path)
    data = mask.get_fdata() > 0.5
    if data.sum() == 0:
        print(f"  [{pid}_{side}] empty mask, skip")
        return
    centroid_vox = np.argwhere(data).mean(axis=0)

    M, A_original = build_result_to_original_voxel(
        pid, side, p1_log, p2_log, p3_log, processed_dir,
        mask.affine, data.shape, rotation_center_scale=args.center_scale,
    )
    orig_vox = M @ np.append(centroid_vox, 1.0)
    orig_world = (A_original @ orig_vox)[:3]

    centroids = get_step(p1_log, "centroid_calculation")["parameters"]
    key = "left_ear_centroid_mm" if side == "left" else "right_ear_centroid_mm"
    anchor = np.array(centroids[key], dtype=float)
    dist = np.linalg.norm(orig_world - anchor)
    print(f"  [{pid}_{side}] canal centroid -> original {np.round(orig_world,1)}  "
          f"| P1 ear centroid {np.round(anchor,1)}  | dist {dist:.1f} mm")


def save_stl_binary(vertices, faces, filepath):
    """Write a binary STL from vertices (Nx3) and faces (Mx3)."""
    v0 = vertices[faces[:, 0]]; v1 = vertices[faces[:, 1]]; v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True); norms[norms == 0] = 1
    normals = normals / norms
    with open(filepath, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(faces)))
        for i, face in enumerate(faces):
            f.write(struct.pack("<3f", *normals[i]))
            f.write(struct.pack("<3f", *vertices[face[0]]))
            f.write(struct.pack("<3f", *vertices[face[1]]))
            f.write(struct.pack("<3f", *vertices[face[2]]))
            f.write(b"\x00\x00")


def world_to_output(points_ras, to_lps):
    pts = np.asarray(points_ras, dtype=float).reshape(-1, 3).copy()
    if to_lps:
        pts[:, 0] *= -1
        pts[:, 1] *= -1
    return pts


def map_mask_to_original(mask_img, M, original_img):
    """Resample a result mask onto the original CT grid (nearest neighbour)."""
    Minv = np.linalg.inv(M)  # original voxel -> result voxel
    data = np.asanyarray(mask_img.dataobj).astype(np.float32)
    out = ndimage.affine_transform(
        data, Minv[:3, :3], offset=Minv[:3, 3],
        output_shape=original_img.shape, order=0, cval=0.0,
    )
    return nib.Nifti1Image((out > 0.5).astype(np.uint8), original_img.affine, original_img.header)


def parse_args():
    ap = argparse.ArgumentParser(description="Map results back to original scan space")
    ap.add_argument("--logs_dir", required=True, help="Logs/transform_logs directory")
    ap.add_argument("--processed_dir", required=True,
                    help="Processed-Data dir containing per-patient intermediate NIfTIs")
    ap.add_argument("--masks_dir", required=True, help="Results/masks (tissue) directory")
    ap.add_argument("--masks_bone_dir", default=None, help="Results/masks_bone directory (optional)")
    ap.add_argument("--stl_dir", default=None, help="Results/stl directory (optional)")
    ap.add_argument("--stl_bone_dir", default=None, help="Results/stl_bone directory (optional)")
    ap.add_argument("--markups_dir", default=None, help="Results/markups directory (optional)")
    ap.add_argument("--slicer_markups_dir", default=None,
                    help="Results directory with 3D Slicer .mrk.json markups (optional)")
    ap.add_argument("--output_dir", default=None, help="output root for mapped results")
    ap.add_argument("--input_coordinate_system", default="LPS", choices=["LPS", "RAS"],
                    help="convention of the input STL/markups (default LPS)")
    ap.add_argument("--patients", nargs="*", default=None, help="patient ids (default: all in masks_dir)")
    ap.add_argument("--center_scale", type=float, default=1.0,
                    help="scale for logged rotation center (1.0=voxel, else 1/spacing)")
    ap.add_argument("--validate", action="store_true", help="run validation against P1 ear centroids")
    return ap.parse_args()


def apply_one(pid, side, args):
    p1_log = load_json(os.path.join(args.logs_dir, f"{pid}_transform_log.json"))
    p2_log = load_json(os.path.join(args.logs_dir, f"{pid}_transform_log_P2.json"))
    p3_log = load_json(os.path.join(args.logs_dir, f"{pid}_transform_log_P3.json"))
    processed_dir = os.path.join(args.processed_dir, pid)
    original_img = nib.load(_find_original(p1_log))

    mask_path = find_nifti(os.path.join(args.masks_dir, f"{pid}_{side}"))
    if mask_path is None:
        print(f"  [{pid}_{side}] no result mask, skip")
        return
    ref = nib.load(mask_path)
    M, A_original = build_result_to_original_voxel(
        pid, side, p1_log, p2_log, p3_log, processed_dir,
        ref.affine, ref.shape, rotation_center_scale=args.center_scale,
    )
    is_lps = args.input_coordinate_system.upper() == "LPS"

    def map_points(pts_world):
        orig = result_world_to_original_world(pts_world, ref.affine, M, A_original, input_is_lps=is_lps)
        return world_to_output(orig, to_lps=is_lps)

    # masks
    for mdir, sub in [(args.masks_dir, "masks"), (args.masks_bone_dir, "masks_bone")]:
        if not mdir:
            continue
        p = find_nifti(os.path.join(mdir, f"{pid}_{side}"))
        if p is None:
            continue
        out = map_mask_to_original(nib.load(p), M, original_img)
        outdir = os.path.join(args.output_dir, sub); os.makedirs(outdir, exist_ok=True)
        nib.save(out, os.path.join(outdir, f"{pid}_{side}.nii.gz"))

    # STL
    import pyvista as pv
    for sdir, sub in [(args.stl_dir, "stl"), (args.stl_bone_dir, "stl_bone")]:
        if not sdir:
            continue
        p = os.path.join(sdir, f"{pid}_{side}.stl")
        if not os.path.exists(p):
            continue
        mesh = pv.read(p)
        mesh.points = map_points(mesh.points)
        outdir = os.path.join(args.output_dir, sub); os.makedirs(outdir, exist_ok=True)
        mesh.save(os.path.join(outdir, f"{pid}_{side}.stl"))

    # markups
    if args.markups_dir:
        p = os.path.join(args.markups_dir, f"{pid}_{side}.json")
        if os.path.exists(p):
            mk = load_json(p)
            for lm in mk.get("landmarks", []):
                if lm.get("position") is not None:
                    lm["position"] = map_points(lm["position"])[0].tolist()
            outdir = os.path.join(args.output_dir, "markups"); os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"{pid}_{side}.json"), "w") as f:
                json.dump(mk, f, indent=2)

    # 3D Slicer markups (.mrk.json)
    if args.slicer_markups_dir:
        p = os.path.join(args.slicer_markups_dir, f"{pid}_{side}.mrk.json")
        if os.path.exists(p):
            mk = load_json(p)
            for markup in mk.get("markups", []):
                for cp in markup.get("controlPoints", []):
                    if cp.get("position") is not None:
                        cp["position"] = map_points(cp["position"])[0].tolist()
            outdir = os.path.join(args.output_dir, "markups_slicer"); os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"{pid}_{side}.mrk.json"), "w") as f:
                json.dump(mk, f, indent=2)
    print(f"  [{pid}_{side}] mapped to original space")


def main():
    args = parse_args()
    if args.patients:
        pairs = [(p, s) for p in args.patients for s in ("left", "right")]
    else:
        pairs = []
        for f in sorted(os.listdir(args.masks_dir)):
            if f.endswith(NIFTI_EXTENSIONS):
                stem = strip_nifti_extension(f)
                m = re.match(r"(.+)_(left|right)$", stem)
                if m:
                    pairs.append((m.group(1), m.group(2)))

    if args.validate:
        print("Validation (canal centroid mapped to original vs P1 ear centroid):")
        for pid, side in pairs:
            try:
                validate(pid, side, args)
            except Exception as e:
                print(f"  [{pid}_{side}] ERROR: {e}")
        return

    if not args.output_dir:
        raise SystemExit("--output_dir is required (unless --validate)")
    print(f"Mapping results to original space -> {args.output_dir}")
    for pid, side in pairs:
        try:
            apply_one(pid, side, args)
        except Exception as e:
            print(f"  [{pid}_{side}] ERROR: {e}")


if __name__ == "__main__":
    main()
