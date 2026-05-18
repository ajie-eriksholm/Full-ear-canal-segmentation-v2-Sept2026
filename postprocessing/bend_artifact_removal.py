"""
Bend removal from cropped-head ear STLs.

Context
-------
Each STL is generated from a 90x90x90 voxel ROI cropped around the ear inside
an aligned scan (Frankfort plane forced to lie along XY by P2). For the LEFT
ear the ROI is mirrored along axis 0 (X). Origin is then reset to (0, 0, 0).

Because the ROI cuts through the head, the surface created by marching cubes
includes a flat artefact ("the bend") that is the cropping bounding-box face
covered with skin/soft tissue. That sheet is approximately axis-aligned in
the STL coordinate frame (the alignment from P2 guarantees this).

This script:
  1. Reads the P2 transform log to display the alignment / rotation context.
  2. Loads the left and right STLs.
  3. Detects the bend plane with a constrained RANSAC: the plane must be
     close to one of the cardinal axes (consistent with an axis-aligned ROI
     face) AND lie near a face of the mesh bounding box (it is a crop
     boundary, not an interior anatomical plane).
  4. Clips the mesh, keeping the side that contains the dense ear-canal
     centroid (estimated by 3D point density).
  5. Saves cleaned STLs and a side-by-side PyVista visualisation
     (before / after, both ears).
"""

import json
import os
import random
import glob
from pathlib import Path

import numpy as np
import pyvista as pv

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
STL_DIR = r"\\kbnnfsserver\erhdata\Processed-Data\SBEO\HighRes_retrain\Output\Results\stl"
LOG_DIR = r"\\kbnnfsserver\erhdata\Processed-Data\SBEO\HighRes_retrain\Output\Logs\transform_logs"

# Number of random patients to sample (set to None to process all).
NUM_SAMPLES = None
RANDOM_SEED = 42

OUTPUT_DIR = r"\\kbnnfsserver\erhdata\Processed-Data\SBEO\HighRes_retrain\bend_removal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Bend detection parameters
# ---------------------------------------------------------------------------
# Tolerance (mm) for counting points ON the candidate bend plane.
FACE_INLIER_TOL = 1.5
# Step (mm) used when scanning the plane offset along the bend normal.
OFFSET_STEP_MM = 0.25
# Safety margin (mm) added to the clip plane so we cut a little inside the
# bend rather than tangent to it.
CLIP_MARGIN_MM = 1.5
# Minimum distance (mm) the bend plane offset must be from the mesh bbox
# extent along the candidate normal. Prevents locking onto the outermost
# slice (where there are very few points anyway, but just in case).
EDGE_MARGIN_MM = 2.0

# Minimum fraction of mesh points that must lie within FACE_INLIER_TOL of
# the best plane for it to be considered a real bend. If the best inlier
# fraction is below this threshold, the mesh is assumed to have NO bend
# and is returned unmodified.
MIN_BEND_INLIER_FRAC = 0.05
# Additional sanity check: the inlier slice must contain more points than
# (this multiplier) x (average points-per-slice over the sweep). A real
# flat sheet is a strong outlier; a curved-only mesh produces a roughly
# uniform sweep.
MIN_BEND_PEAK_RATIO = 4.0

# Side of the plane to keep (along the bend normal).
#   "auto"     - keep the side with more surface points
#   "auto_inv" - keep the side with fewer surface points
#   "+"        - keep the side where (n . x + d) > 0  (above the bend)
#   "-"        - keep the side where (n . x + d) < 0
# The bend normal points in the original-CT +Z direction (head up), so the
# anatomy we want to keep lies on the +normal side.
KEEP_SIDE = "+"

# Optional manual override of the bend offset (mm along the bend normal).
# Set to None to use the automatic search.
MANUAL_BEND_OFFSET = None  # e.g. 80.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_alignment_info(path):
    """Pull rotation/Frankfort context from the P2 transform log."""
    with open(path, "r") as f:
        log = json.load(f)
    info = {"path": path}
    for tr in log.get("transformations", []):
        if tr.get("operation") == "frankfort_plane_alignment":
            params = tr.get("parameters", {})
            info["rotation_matrix"] = np.array(params.get("rotation_matrix", np.eye(3)))
            info["rotation_angles_xyz_deg"] = params.get("rotation_angles_xyz_degrees")
            info["z_rotation_angle_deg"] = params.get("z_rotation_angle_degrees")
            info["original_plane_normal"] = params.get("original_plane_normal")
            info["target_plane_normal"] = params.get("target_plane_normal")
            break
    return info


def bend_normal_in_stl_frame(rotation_matrix, is_left):
    """
    Build the bend plane direction as follows:
      1. Start with a flat plane perpendicular to the original CT +Z axis
         (i.e. normal n0 = [0, 0, 1] in the original head frame).
      2. P2 recorded a rotation matrix R that maps original-frame coords to
         the aligned STL frame. To express n0 in the aligned/STL frame we
         apply the INVERSE rotation:
                 n_aligned = R^T @ n0
      3. P3 crops the ROI and, for the LEFT ear, additionally mirrors the
         volume along axis 0 (X). Mirroring flips the X component of any
         vector, so for the left STL:
                 n_stl = [-n_aligned_x, n_aligned_y, n_aligned_z]
    Returns a unit normal in the STL coordinate frame.
    """
    R_mat = np.asarray(rotation_matrix, dtype=np.float64)
    n = R_mat.T @ np.array([0.0, 0.0, 1.0])
    n = n / np.linalg.norm(n)
    if is_left:
        n = np.array([-n[0], n[1], n[2]])
    return n


def detect_bend_offset(mesh, normal):
    """
    Slide a plane with the given (fixed, tilted) `normal` along the global
    +Z direction starting from z = 0 and moving upward. For each plane
    position count the surface points within FACE_INLIER_TOL of the plane;
    the position with the most inliers is the bend.

    Implementation note: shifting the plane origin from (0,0,0) by (0,0,t)
    along world +Z is equivalent to shifting the signed offset along the
    plane normal by `t * normal_z`. So we parameterise the sweep by `t` (mm
    of vertical lift) and compute inliers as
        |normal . point - t * normal_z| < FACE_INLIER_TOL.

    Returns (normal, d, inlier_mask, scan, (z_min, z_max))
        d such that plane equation is normal . x + d = 0
        scan = list of (t_mm, inlier_count, weight=1.0)
    """
    points = np.asarray(mesh.points, dtype=np.float64)
    proj = points @ normal           # signed distance from origin along normal
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())

    nz = float(normal[2])
    if abs(nz) < 1e-6:
        raise ValueError(
            "Bend normal is nearly horizontal (n_z ~ 0); cannot sweep along +Z."
        )

    # Sweep the plane upward in world Z from t = 0 (plane at the origin) to
    # t = z_max + EDGE_MARGIN_MM, in OFFSET_STEP_MM steps.
    t_lo = 0.0
    t_hi = z_max + EDGE_MARGIN_MM
    ts = np.arange(t_lo, t_hi + OFFSET_STEP_MM * 0.5, OFFSET_STEP_MM)

    best_count = -1
    best_t = 0.0
    scan = []
    for t in ts:
        # plane offset along its own normal corresponding to vertical lift t
        off = t * nz
        count = int((np.abs(proj - off) < FACE_INLIER_TOL).sum())
        scan.append((float(t), count, 1.0))
        if count > best_count:
            best_count = count
            best_t = float(t)

    best_offset = best_t * nz   # plane equation: normal . x = best_offset
    d = -best_offset
    inliers = np.abs(proj - best_offset) < FACE_INLIER_TOL
    return normal, d, inliers, scan, (z_min, z_max)


def ear_canal_seed(mesh):
    """Densest 5 mm cell of the surface point distribution."""
    pts = np.asarray(mesh.points)
    bounds = mesh.bounds
    extents = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    bins = np.maximum((extents / 5.0).astype(int), 1)
    H, edges = np.histogramdd(pts, bins=bins)
    idx = np.unravel_index(np.argmax(H), H.shape)
    cell_min = np.array([edges[i][idx[i]]   for i in range(3)])
    cell_max = np.array([edges[i][idx[i]+1] for i in range(3)])
    return 0.5 * (cell_min + cell_max)


def clip_bend(mesh, normal, d, margin=CLIP_MARGIN_MM, keep_side=KEEP_SIDE):
    """Clip the mesh, keeping the requested side of the plane."""
    points = np.asarray(mesh.points, dtype=np.float64)
    signed = points @ normal + d
    n_pos = int((signed >  margin).sum())
    n_neg = int((signed < -margin).sum())

    if keep_side == "auto":
        keep_positive = n_pos >= n_neg
    elif keep_side == "auto_inv":
        keep_positive = n_pos <= n_neg
    elif keep_side == "+":
        keep_positive = True
    elif keep_side == "-":
        keep_positive = False
    else:
        raise ValueError(f"Unknown KEEP_SIDE: {keep_side!r}")

    origin0 = -d * normal
    if keep_positive:
        origin = origin0 + margin * normal
        invert = False
    else:
        origin = origin0 - margin * normal
        invert = True
    print(f"    keep_side={keep_side} -> {'+' if keep_positive else '-'}  "
          f"(n_pos={n_pos}, n_neg={n_neg})")
    return mesh.clip(normal=normal, origin=origin, invert=invert)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(rows, alignment_info, output_png):
    """
    rows: list of (label, mesh_before, mesh_after, plane_or_None).

    Layout: 4 rows x N columns.
      row 0: BEFORE - isometric, with detected plane in red
      row 1: BEFORE - side view (along Y), with plane
      row 2: AFTER  - isometric
      row 3: AFTER  - side view
    """
    n = len(rows)
    plotter = pv.Plotter(shape=(4, n), off_screen=True, window_size=[1600, 2000])

    for col, (label, before, after, plane) in enumerate(rows):
        # Build the visualization plane disk if available
        disk = None
        if plane is not None:
            normal, d = plane
            origin = -d * normal
            extents = np.array(before.bounds[1::2]) - np.array(before.bounds[::2])
            size = float(np.linalg.norm(extents))
            disk = pv.Plane(center=origin, direction=normal,
                            i_size=size, j_size=size)

        for row, view_name in enumerate(["iso", "side"]):
            # BEFORE
            plotter.subplot(row * 2, col)
            plotter.add_mesh(before, color="lightgray", opacity=1.0,
                             show_edges=False, smooth_shading=True)
            if disk is not None:
                plotter.add_mesh(disk, color="red", opacity=0.35,
                                 show_edges=True, edge_color="red")
            plotter.add_text(f"{label} BEFORE ({view_name})",
                             position="upper_edge", font_size=11)
            plotter.add_axes()
            if view_name == "iso":
                plotter.view_isometric()
            else:
                plotter.view_xz()

            # AFTER
            plotter.subplot(row * 2 + 1, col)
            plotter.add_mesh(after, color="lightblue", opacity=1.0,
                             show_edges=False, smooth_shading=True)
            plotter.add_text(f"{label} AFTER ({view_name})",
                             position="upper_edge", font_size=11)
            plotter.add_axes()
            if view_name == "iso":
                plotter.view_isometric()
            else:
                plotter.view_xz()

    angles = alignment_info.get("rotation_angles_xyz_deg")
    z_rot  = alignment_info.get("z_rotation_angle_deg")
    summary = "Frankfort alignment: "
    if angles is not None:
        summary += f"XYZ rot = [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}] deg"
    if z_rot is not None:
        summary += f"  |  Z extra = {z_rot:.1f} deg"
    plotter.add_text(summary, position=(10, 10), font_size=10, color="black")

    plotter.screenshot(output_png)
    plotter.close()
    print(f"[viz] saved {output_png}")


# ---------------------------------------------------------------------------
# Per-ear processing
# ---------------------------------------------------------------------------
def process_ear(stl_path, label, is_left, rotation_matrix):
    print(f"\n=== {label} ear: {stl_path} ===")
    mesh = pv.read(stl_path)
    print(f"  loaded: {mesh.n_points} points, {mesh.n_cells} faces")
    print(f"  bounds: {mesh.bounds}")

    # 1. Bend normal = inverse P2 rotation applied to original +Z axis,
    #    with X-component flipped for the left ear (P3 mirrors LEFT along X).
    normal = bend_normal_in_stl_frame(rotation_matrix, is_left=is_left)
    print(f"  bend normal (R^T @ [0,0,1]"
          f"{', X-mirrored' if is_left else ''}): {normal}")

    # 2. Slide a plane with that normal upward in world +Z starting at z=0
    #    and pick the position that maximises inlier count.
    ear_centroid = ear_canal_seed(mesh)
    print(f"  ear-canal seed: {ear_centroid}")

    normal, d, inliers, scan, (z_min, z_max) = detect_bend_offset(mesh, normal)
    if MANUAL_BEND_OFFSET is not None:
        d = -float(MANUAL_BEND_OFFSET)
        proj = np.asarray(mesh.points) @ normal
        inliers = np.abs(proj - MANUAL_BEND_OFFSET) < FACE_INLIER_TOL
        print(f"  *** manual override: bend offset set to "
              f"{MANUAL_BEND_OFFSET:.2f} mm ***")
    best_offset = -d
    nz = float(normal[2])
    best_t = best_offset / nz if abs(nz) > 1e-6 else float("nan")
    print(f"  mesh z range: [{z_min:.2f}, {z_max:.2f}] mm")
    print(f"  best vertical lift t = {best_t:.2f} mm "
          f"(plane offset along normal = {best_offset:.2f} mm)")
    # Show the top 5 lift values so the user can pick a different one if needed
    top = sorted(scan, key=lambda x: x[1], reverse=True)[:5]
    print("  top-5 vertical lifts (t_mm, inliers):")
    for t, cnt, w in top:
        marker = " <- selected" if abs(t - best_t) < OFFSET_STEP_MM else ""
        print(f"    t={t:7.2f} mm   inliers={cnt:5d}{marker}")

    # --- No-bend detection -------------------------------------------------
    # If the "best" plane has very few inliers relative to the mesh, or the
    # inlier count is not a strong peak compared to the average sweep slice,
    # there is no real flat sheet to remove. Skip clipping in that case.
    best_count = int(inliers.sum())
    n_points = int(mesh.n_points)
    inlier_frac = best_count / max(n_points, 1)
    sweep_counts = np.array([c for _, c, _ in scan], dtype=np.float64)
    avg_count = float(sweep_counts.mean()) if sweep_counts.size else 0.0
    peak_ratio = (best_count / avg_count) if avg_count > 0 else float("inf")
    print(f"  inliers: {best_count} / {n_points} = {inlier_frac:.3%}   "
          f"peak/avg = {peak_ratio:.2f}")
    if (MANUAL_BEND_OFFSET is None
            and (inlier_frac < MIN_BEND_INLIER_FRAC
                 or peak_ratio < MIN_BEND_PEAK_RATIO)):
        print(f"  >> no bend detected (inlier_frac<{MIN_BEND_INLIER_FRAC} "
              f"or peak_ratio<{MIN_BEND_PEAK_RATIO}); keeping mesh as-is")
        cleaned = mesh.extract_surface()
        out_stl = os.path.join(OUTPUT_DIR, Path(stl_path).stem + "_no_bend.stl")
        cleaned.save(out_stl)
        print(f"  saved (unclipped) STL: {out_stl}")
        return mesh, cleaned, None

    # 3. Clip on the side of the plane that has more mesh points (the ear).
    cleaned = clip_bend(mesh, normal, d)
    cleaned = cleaned.extract_surface().connectivity(extraction_mode="largest")
    print(f"  after clip + largest-component: {cleaned.n_points} points")

    out_stl = os.path.join(OUTPUT_DIR, Path(stl_path).stem + "_no_bend.stl")
    cleaned.extract_surface().save(out_stl)
    print(f"  saved cleaned STL: {out_stl}")

    return mesh, cleaned, (normal, d)


def find_patient_triples(stl_dir, log_dir):
    """
    Discover patients that have BOTH left and right STLs and a P2 log.
    Returns a list of dicts: {patient_id, left_stl, right_stl, log}.
    """
    left_stls = glob.glob(os.path.join(stl_dir, "*_left.stl"))
    triples = []
    for left in left_stls:
        pid = Path(left).stem[:-len("_left")]
        right = os.path.join(stl_dir, f"{pid}_right.stl")
        log = os.path.join(log_dir, f"{pid}_transform_log_P2.json")
        if os.path.isfile(right) and os.path.isfile(log):
            triples.append({
                "patient_id": pid,
                "left_stl": left,
                "right_stl": right,
                "log": log,
            })
    return triples


def main():
    print(f"Scanning {STL_DIR} ...")
    triples = find_patient_triples(STL_DIR, LOG_DIR)
    print(f"  found {len(triples)} patients with left+right STL and P2 log")
    if not triples:
        print("Nothing to do.")
        return

    rng = random.Random(RANDOM_SEED)
    if NUM_SAMPLES is not None and NUM_SAMPLES < len(triples):
        triples = rng.sample(triples, NUM_SAMPLES)
    print(f"  processing {len(triples)} patients (seed={RANDOM_SEED})")
    for t in triples:
        print(f"   - {t['patient_id']}")

    for t in triples:
        pid = t["patient_id"]
        print(f"\n{'#'*70}\n# {pid}\n{'#'*70}")
        try:
            alignment_info = load_alignment_info(t["log"])
            print(f"  rotation angles XYZ (deg): {alignment_info.get('rotation_angles_xyz_deg')}")
            print(f"  z extra rotation (deg):    {alignment_info.get('z_rotation_angle_deg')}")

            left_before, left_after, left_plane = process_ear(
                t["left_stl"], "LEFT", is_left=True,
                rotation_matrix=alignment_info["rotation_matrix"],
            )
            right_before, right_after, right_plane = process_ear(
                t["right_stl"], "RIGHT", is_left=False,
                rotation_matrix=alignment_info["rotation_matrix"],
            )

            out_png = os.path.join(OUTPUT_DIR, f"{pid}_bend_removal.png")
            visualize(
                [
                    ("LEFT",  left_before,  left_after,  left_plane),
                    ("RIGHT", right_before, right_after, right_plane),
                ],
                alignment_info=alignment_info,
                output_png=out_png,
            )
        except Exception as e:
            print(f"  !! FAILED for {pid}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()