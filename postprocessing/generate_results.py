import os
import json
import struct
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from skimage import measure
import pyvista as pv
import vtk as _vtk
import argparse

# === Configuration ===
FH_plane_lm = "/projects/oticon/erhdata/Processed-Data/AJIE/tcia/all/Output/Aligned_Landmarks/landmark_positions_after_cropping.csv"

predicted_landmarks = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Inference_results/run_20260210_105409/test_predictions/predicted_landmark_coordinates.csv"
masks_dir = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Inference_results/run_20260210_105409/test_predictions"

output_dir_markups = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/markups"
output_dir_markups_no_fh = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/markups"
output_dir_masks = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/masks"

# 3D Slicer fiducial markups (.mrk.json) export (optional). When enabled, the
# landmarks are also written in the native 3D Slicer markups schema so they can
# be loaded directly in Slicer. Defaults to the markups directory when empty.
export_slicer_markups = False
output_dir_slicer_markups = ""
output_dir_stl = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/stl"

bone_masks_dir = ""  # Directory with nnU-Net bone predictions (optional)
output_dir_masks_bone = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/masks_bone"
output_dir_stl_bone = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/stl_bone"

# P4 normalized inputs (_0000) shared by both models as the reference grid.
# When empty, the tissue masks in masks_dir are used as the reference instead.
reference_grid_dir = ""

# Coordinate system for exported STL vertices and markup positions.
# nibabel affines give RAS; 3D Slicer reads STL models and markup JSONs as LPS,
# so "LPS" (negate x,y) makes STL, markups and the NIfTI masks overlay in Slicer.
output_coordinate_system = "LPS"

# Label mapping for landmarks 1-7 (from predicted_landmark_coordinates.csv)
CANAL_LABELS = {
    1: "Cartilaginous Canal Point",
    2: "Bony Canal Point",
    3: "Eardrum",
    4: "Top RS",
    5: "Bottom RS",
    6: "1st bend",
    7: "2nd bend",
}

# FH plane landmark mapping per ear side
# Maps (ear_side, original_landmark_id) -> (output_id, label)
FH_LABELS = {
    ("right", 8):  (8, "Bottom Tragus"),
    ("right", 10): (9, "Top Tragus"),
    ("right", 12): (10, "Eye Orbit"),
    ("left", 9):   (8, "Bottom Tragus"),
    ("left", 11):  (9, "Top Tragus"),
    ("left", 13):  (10, "Eye Orbit"),
}

# nnU-Net bone segmentation labels
# 0: background, 1: skull, 2: mandible, 3-6: landmarks (CBJ1-CBJ4)
BONE_SEG_LABELS = [1, 2]  # skull + mandible (merged as "bone")
BONE_LANDMARK_LABELS = {
    3: (11, "CBJ1"),
    4: (12, "CBJ2"),
    5: (13, "CBJ3"),
    6: (14, "CBJ4"),
}

NIFTI_EXTENSIONS = ('.nii.gz', '.nii')


def strip_nifti_extension(filename):
    """Strip .nii or .nii.gz extension from a filename."""
    for ext in NIFTI_EXTENSIONS:
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def list_nifti_files(directory):
    """List NIfTI files (.nii or .nii.gz) in a directory."""
    return sorted(f for f in os.listdir(directory) if f.endswith(NIFTI_EXTENSIONS))


def to_output_coords(points):
    """Map RAS world coords (nibabel affine output) to the export convention.

    With output_coordinate_system == "LPS" this negates x and y, so exported STL
    and markups overlay the NIfTI masks in 3D Slicer for any input affine
    convention (x,y,z or -x,-y,z). "RAS" leaves coordinates unchanged.
    """
    pts = np.asarray(points, dtype=float)
    if output_coordinate_system.upper() == "LPS":
        pts = pts.copy()
        pts[..., 0] *= -1
        pts[..., 1] *= -1
    return pts


def parse_patient_side(filename):
    """Return (patient, ear_side) from a bone prediction filename, else (None, None)."""
    base = strip_nifti_extension(filename)
    if base.endswith("_0000"):
        base = base[:-len("_0000")]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("left", "right"):
        return parts[0], parts[1]
    return None, None


def find_reference_grid_image(patient, ear_side):
    """Locate an image on the shared grid (correct affine) for this ear.

    Prefers the P4 normalized input (reference_grid_dir), then falls back to the
    tissue prediction in masks_dir, which already carries the correct affine.
    """
    search = []
    if reference_grid_dir and os.path.isdir(reference_grid_dir):
        search += [(reference_grid_dir, suffix) for suffix in ("_0000", "")]
    if masks_dir and os.path.isdir(masks_dir):
        search.append((masks_dir, "_pred_seg"))
    for directory, suffix in search:
        for ext in NIFTI_EXTENSIONS:
            candidate = os.path.join(directory, f"{patient}_{ear_side}{suffix}{ext}")
            if os.path.exists(candidate):
                return candidate
    return None


def load_bone_prediction(bone_path, patient, ear_side):
    """Load an nnU-Net bone prediction in the shared tissue/P4 reference space.

    nnU-Net writes via SimpleITK (LPS), flipping the x/y sform sign relative to
    the nibabel-written tissue masks, so the two would otherwise be mirrored in
    world space. The prediction is voxel-index aligned with the P4 input, so we
    re-stamp the reference affine/header (no interpolation, no data flip). This
    keeps masks, STL and landmarks aligned for any input convention (x,y,z or
    -x,-y,z).
    """
    nii = nib.load(bone_path)
    ref_path = find_reference_grid_image(patient, ear_side)
    if ref_path is None:
        return nii
    ref = nib.load(ref_path)
    if ref.shape != nii.shape:
        print(f"  Warning: reference shape {ref.shape} != bone shape {nii.shape} "
              f"for {patient}_{ear_side}; keeping original affine")
        return nii
    if not np.allclose(nii.affine, ref.affine, atol=1e-4):
        print(f"  Aligning bone mask to reference grid for {patient}_{ear_side}")
    return nib.Nifti1Image(np.asanyarray(nii.dataobj), ref.affine, ref.header)


def get_pred_seg_base_name(filename):
    """Return base scan name for <name>_pred_seg.nii[.gz], else None."""
    stem = strip_nifti_extension(filename)
    suffix = '_pred_seg'
    if not stem.endswith(suffix):
        return None
    return stem[:-len(suffix)]


def save_stl_binary(vertices, faces, filepath):
    """Write a binary STL file from vertices and faces arrays."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    with open(filepath, "wb") as f:
        f.write(b"\0" * 80)  # 80-byte header
        f.write(struct.pack("<I", len(faces)))
        for i, face in enumerate(faces):
            f.write(struct.pack("<3f", *normals[i]))
            f.write(struct.pack("<3f", *vertices[face[0]]))
            f.write(struct.pack("<3f", *vertices[face[1]]))
            f.write(struct.pack("<3f", *vertices[face[2]]))
            f.write(b"\x00\x00")  # attribute byte count


def nifti_mask_to_block_stl(binary_mask, affine, output_stl_path):
    """Convert a binary 3D mask into a solid block STL using pyvista.

    Steps:
      1. Build a pyvista ImageData (voxel grid) in index space from the array.
      2. Threshold to keep only foreground voxels -> solid block geometry.
      3. Extract the outer surface of those voxels (blocky).
      4. Map voxel indices to world mm with the full affine (keeps direction
         sign, so this matches the marching-cubes tissue STL for any convention).
      5. Apply VTK smoothing (Windowed Sinc + Laplacian) for refinement.
      6. Save as STL in world coordinates.
    """
    # Build in index space (unit spacing, zero origin); the signed affine is
    # applied to the surface vertices below so -x,-y,z inputs are not mirrored.
    grid = pv.ImageData()
    grid.dimensions = np.array(binary_mask.shape) + 1  # cell-based: N+1 points per axis
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["mask"] = binary_mask.flatten(order="F").astype(float)

    # Threshold -> solid block cells
    block = grid.threshold(0.5, scalars="mask")
    if block.n_cells == 0:
        print(f"  Warning: block mesh empty, skipping STL.")
        return False

    # Extract outer surface of the voxel block
    surface = block.extract_surface()

    # Voxel index -> world mm using the full (signed) affine, then export coords
    pts = np.asarray(surface.points)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    surface.points = to_output_coords((affine @ pts_h.T).T[:, :3])

    # Windowed Sinc smoothing (high quality, aggressive)
    sinc_smooth = _vtk.vtkWindowedSincPolyDataFilter()
    sinc_smooth.SetInputData(surface)
    sinc_smooth.SetNumberOfIterations(50)
    sinc_smooth.BoundarySmoothingOff()
    sinc_smooth.FeatureEdgeSmoothingOff()
    sinc_smooth.SetPassBand(0.01)
    sinc_smooth.NonManifoldSmoothingOn()
    sinc_smooth.NormalizeCoordinatesOn()
    sinc_smooth.Update()

    # Laplacian smoothing for additional refinement
    laplacian = _vtk.vtkSmoothPolyDataFilter()
    laplacian.SetInputData(sinc_smooth.GetOutput())
    laplacian.SetNumberOfIterations(30)
    laplacian.SetRelaxationFactor(0.5)
    laplacian.FeatureEdgeSmoothingOff()
    laplacian.BoundarySmoothingOn()
    laplacian.Update()

    surface = pv.wrap(laplacian.GetOutput())
    surface.save(output_stl_path)
    return True


def extract_bone_landmarks(bone_masks_directory):
    """Extract landmark coordinates from nnU-Net bone segmentation predictions.
    
    Labels 3-6 in the nnU-Net output are landmarks. For each, compute the
    center of mass in world coordinates (mm).
    
    Returns:
        dict: {(patient, ear_side): [(output_id, label, [x,y,z]), ...]}
    """
    bone_landmarks = {}
    if not bone_masks_directory or not os.path.isdir(bone_masks_directory):
        return bone_landmarks

    nii_files = list_nifti_files(bone_masks_directory)
    for fname in nii_files:
        # Parse patient and ear_side from filename (e.g. "patient_left.nii.gz")
        patient, ear_side = parse_patient_side(fname)
        if patient is None:
            print(f"  Warning: cannot parse patient/side from bone file: {fname}, skipping landmarks")
            continue

        nii = load_bone_prediction(os.path.join(bone_masks_directory, fname), patient, ear_side)
        data = np.asanyarray(nii.dataobj).astype(np.int16)
        affine = nii.affine

        lm_list = []
        for lbl, (out_id, label) in BONE_LANDMARK_LABELS.items():
            lbl_mask = (data == lbl)
            if not np.any(lbl_mask):
                continue
            # Centroid in voxel coordinates — center_of_mass returns (i, j, k)
            # matching the array axis order from nibabel's get_fdata()
            com_ijk = np.array(ndimage.center_of_mass(lbl_mask))
            # The NIfTI affine maps (i, j, k) -> (x, y, z) directly
            voxel_ijk1 = np.array([com_ijk[0], com_ijk[1], com_ijk[2], 1.0])
            world = affine @ voxel_ijk1
            lm_list.append((out_id, label, [float(world[0]), float(world[1]), float(world[2])]))

        if lm_list:
            bone_landmarks[(patient, ear_side)] = lm_list

    print(f"Extracted bone landmarks for {len(bone_landmarks)} ears")
    return bone_landmarks


def process_bone_masks():
    """Process nnU-Net bone predictions: merge skull+mandible into binary bone mask,
    generate NIfTI masks and STL meshes."""
    if not bone_masks_dir or not os.path.isdir(bone_masks_dir):
        print("No bone masks directory provided or found, skipping bone mask processing.")
        return

    os.makedirs(output_dir_masks_bone, exist_ok=True)
    os.makedirs(output_dir_stl_bone, exist_ok=True)

    nii_files = list_nifti_files(bone_masks_dir)
    if not nii_files:
        print("No .nii or .nii.gz files found in bone_masks_dir.")
        return

    processed = 0
    for fname in nii_files:
        patient, ear_side = parse_patient_side(fname)
        if patient is None:
            print(f"  Warning: cannot parse patient/side from bone file: {fname}, skipping.")
            continue
        base = f"{patient}_{ear_side}"

        nii = load_bone_prediction(os.path.join(bone_masks_dir, fname), patient, ear_side)
        data = np.asanyarray(nii.dataobj).astype(np.int16)

        # Merge skull (1) + mandible (2) into binary bone mask
        bone_binary = np.isin(data, BONE_SEG_LABELS).astype(np.uint8)

        if bone_binary.sum() == 0:
            print(f"  Warning: no bone voxels in {fname}, skipping.")
            continue

        # Save NIfTI mask (keep all components for bone)
        bone_nii = nib.Nifti1Image(bone_binary, nii.affine, nii.header)
        nib.save(bone_nii, os.path.join(output_dir_masks_bone, f"{base}.nii.gz"))

        # Generate block STL (pyvista ImageData -> threshold -> extract_surface -> smooth)
        stl_path = os.path.join(output_dir_stl_bone, f"{base}.stl")
        if not nifti_mask_to_block_stl(bone_binary, nii.affine, stl_path):
            continue

        processed += 1
        print(f"  Processed bone mask: {base}")

    print(f"Processed {processed} bone masks -> NIfTI: '{output_dir_masks_bone}', STL: '{output_dir_stl_bone}'")


def process_masks():
    os.makedirs(output_dir_masks, exist_ok=True)
    os.makedirs(output_dir_stl, exist_ok=True)

    mask_files = sorted(f for f in os.listdir(masks_dir) if get_pred_seg_base_name(f) is not None)
    if not mask_files:
        print("No *_pred_seg.nii or *_pred_seg.nii.gz files found in masks_dir.")
        return

    processed = 0
    for mask_file in mask_files:
        base_name = get_pred_seg_base_name(mask_file)
        mask_path = os.path.join(masks_dir, mask_file)

        # Load NIfTI mask
        nii = nib.load(mask_path)
        data = nii.get_fdata().astype(np.float32)
        binary = (data > 0.5).astype(np.uint8)

        # Keep largest connected component
        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            print(f"  Warning: no foreground voxels in {mask_file}, skipping.")
            continue
        component_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        largest_mask = (labeled == (np.argmax(component_sizes) + 1)).astype(np.uint8)

        # Fill internal holes: invert -> keep largest background component -> re-invert
        inverted = 1 - largest_mask
        inv_labeled, inv_num_features = ndimage.label(inverted)
        if inv_num_features > 0:
            inv_sizes = ndimage.sum(inverted, inv_labeled, range(1, inv_num_features + 1))
            largest_bg = (inv_labeled == (np.argmax(inv_sizes) + 1)).astype(np.uint8)
            largest_mask = (1 - largest_bg).astype(np.uint8)

        # Apply Gaussian smoothing (sigma=1.5)
        smoothed = ndimage.gaussian_filter(largest_mask.astype(np.float32), sigma=1.5)

        # Rethreshold and save as integer mask
        smoothed_int = (smoothed > 0.5).astype(np.uint8)
        smoothed_nii = nib.Nifti1Image(smoothed_int, nii.affine, nii.header)
        nib.save(smoothed_nii, os.path.join(output_dir_masks, f"{base_name}.nii.gz"))

        # Generate STL via marching cubes (verts in voxel space -> apply affine)
        try:
            verts_vox, faces_mc, _, _ = measure.marching_cubes(smoothed, level=0.5)
        except Exception as e:
            print(f"  Warning: marching_cubes failed for {mask_file}: {e}")
            continue

        verts_h = np.hstack([verts_vox, np.ones((len(verts_vox), 1))])
        verts_mm = to_output_coords((nii.affine @ verts_h.T).T[:, :3])
        save_stl_binary(verts_mm, faces_mc, os.path.join(output_dir_stl, f"{base_name}.stl"))

        processed += 1
        print(f"  Processed mask: {base_name}")

    print(f"Processed {processed} masks -> NIfTI: '{output_dir_masks}', STL: '{output_dir_stl}'")


def write_slicer_markups(landmarks, coordinate_system, filepath):
    """Write landmarks as a 3D Slicer fiducial markups (.mrk.json) file."""
    control_points = []
    for lm in landmarks:
        position = lm.get("position")
        if position is None:
            continue
        control_points.append({
            "id": str(lm["id"]),
            "label": lm["label"],
            "description": "",
            "associatedNodeID": "",
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined",
        })
    markups_json = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": coordinate_system.upper(),
                "coordinateUnits": "mm",
                "locked": False,
                "controlPoints": control_points,
            }
        ],
    }
    with open(filepath, "w") as f:
        json.dump(markups_json, f, indent=2)


def main():
    # --- Determine output directory based on FH plane CSV availability ---
    fh_available = os.path.exists(FH_plane_lm)
    active_markups_dir = output_dir_markups if fh_available else output_dir_markups_no_fh
    if not fh_available:
        print(f"WARNING: FH plane landmarks CSV not found: {FH_plane_lm}")
        print(f"Markups will be saved (canal landmarks only) to: {active_markups_dir}")
    os.makedirs(active_markups_dir, exist_ok=True)

    slicer_markups_dir = None
    if export_slicer_markups:
        slicer_markups_dir = output_dir_slicer_markups or active_markups_dir
        os.makedirs(slicer_markups_dir, exist_ok=True)
        print(f"3D Slicer markups (.mrk.json) will be saved to: {slicer_markups_dir}")

    # --- Load predicted canal landmarks (1-7) ---
    pred_df = pd.read_csv(predicted_landmarks)
    # scan_name looks like "CHUM-001_left.nii.gz" (after _0000 stripping)
    # Extract patient and ear_side
    pred_df["scan_name_stem"] = pred_df["scan_name"].apply(strip_nifti_extension)
    pred_df["patient"] = pred_df["scan_name_stem"].str.rsplit("_", n=1).str[0]
    pred_df["ear_side"] = pred_df["scan_name_stem"].str.rsplit("_", n=1).str[1]

    # --- Load FH plane landmarks (8-13) ---
    if fh_available:
        fh_df = pd.read_csv(FH_plane_lm)
        # Normalize ear_side: "right_ear" -> "right", "left_ear" -> "left"
        fh_df["ear_side"] = fh_df["ear_side"].str.replace("_ear", "", regex=False)
        fh_grouped = fh_df.groupby(["scan_name", "ear_side"])
    else:
        fh_grouped = {}  # empty — no FH landmarks will be added
    # Columns: scan_name, ear_side, landmark_id, x_mm, y_mm, z_mm, is_inside_roi

    # --- Build JSON per patient ear ---
    # Group predicted landmarks by (patient, ear_side)
    pred_grouped = pred_df.groupby(["patient", "ear_side"])

    # --- Extract bone landmarks from nnU-Net predictions ---
    bone_landmarks = extract_bone_landmarks(bone_masks_dir)

    processed = set()

    for (patient, ear_side), canal_group in pred_grouped:
        landmarks = []

        # Add canal landmarks 1-7
        for _, row in canal_group.sort_values("landmark_id").iterrows():
            lm_id = int(row["landmark_id"])
            if lm_id in CANAL_LABELS:
                landmarks.append({
                    "id": str(lm_id),
                    "label": CANAL_LABELS[lm_id],
                    "position": [row["x_mm"], row["y_mm"], row["z_mm"]]
                })

        # Add FH plane landmarks 8-10
        if fh_available and (patient, ear_side) in fh_grouped.groups:
            fh_group = fh_grouped.get_group((patient, ear_side))
            for _, row in fh_group.iterrows():
                key = (ear_side, int(row["landmark_id"]))
                if key in FH_LABELS:
                    out_id, label = FH_LABELS[key]
                    # Skip if coordinates are None (no_eyes)
                    if pd.isna(row["x_mm"]):
                        landmarks.append({
                            "id": str(out_id),
                            "label": label,
                            "position": None
                        })
                    else:
                        landmarks.append({
                            "id": str(out_id),
                            "label": label,
                            "position": [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])]
                        })

        # Add bone landmarks (from nnU-Net, labels 3-6)
        if (patient, ear_side) in bone_landmarks:
            for out_id, label, position in bone_landmarks[(patient, ear_side)]:
                landmarks.append({
                    "id": str(out_id),
                    "label": label,
                    "position": position
                })

        # Sort by numeric id
        landmarks.sort(key=lambda lm: int(lm["id"]))

        # Export positions in the chosen convention (RAS -> LPS for Slicer)
        for lm in landmarks:
            if lm.get("position") is not None:
                lm["position"] = to_output_coords(lm["position"]).tolist()

        markup = {
            "coordinateSystem": output_coordinate_system.upper(),
            "landmarks": landmarks,
            "count": len(landmarks)
        }

        # Save JSON
        json_filename = f"{patient}_{ear_side}.json"
        json_path = os.path.join(active_markups_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(markup, f, indent=2)

        if slicer_markups_dir is not None:
            slicer_path = os.path.join(slicer_markups_dir, f"{patient}_{ear_side}.mrk.json")
            write_slicer_markups(landmarks, output_coordinate_system, slicer_path)

        processed.add((patient, ear_side))

    print(f"Created {len(processed)} markup JSON files in: {active_markups_dir}")
    if slicer_markups_dir is not None:
        print(f"Created {len(processed)} 3D Slicer markup files in: {slicer_markups_dir}")

    process_masks()
    process_bone_masks()


def parse_arguments():
    """Parse command-line arguments to override configuration."""
    parser = argparse.ArgumentParser(
        description='Generate markup JSONs, NIfTI masks, and STL files from inference outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--fh_plane_lm', type=str, default=None,
                        help='Path to FH plane landmarks CSV (overrides FH_plane_lm)')
    parser.add_argument('--predicted_landmarks', type=str, default=None,
                        help='Path to predicted canal landmarks CSV (overrides predicted_landmarks)')
    parser.add_argument('--masks_dir', type=str, default=None,
                        help='Directory containing predicted segmentation masks (overrides masks_dir)')
    parser.add_argument('--output_dir_markups', type=str, default=None,
                        help='Output directory for markup JSONs with FH (overrides output_dir_markups)')
    parser.add_argument('--output_dir_markups_no_fh', type=str, default=None,
                        help='Output directory for markup JSONs without FH (overrides output_dir_markups_no_fh)')
    parser.add_argument('--output_dir_masks', type=str, default=None,
                        help='Output directory for NIfTI masks (overrides output_dir_masks)')
    parser.add_argument('--output_dir_stl', type=str, default=None,
                        help='Output directory for STL files (overrides output_dir_stl)')
    parser.add_argument('--bone_masks_dir', type=str, default=None,
                        help='Directory containing nnU-Net bone segmentation predictions')
    parser.add_argument('--output_dir_masks_bone', type=str, default=None,
                        help='Output directory for bone NIfTI masks')
    parser.add_argument('--output_dir_stl_bone', type=str, default=None,
                        help='Output directory for bone STL files')
    parser.add_argument('--reference_grid_dir', type=str, default=None,
                        help='Directory with P4 normalized inputs (_0000) used as the shared '
                             'grid to align bone masks to tissue space (defaults to masks_dir)')
    parser.add_argument('--coordinate_system', type=str, default=None, choices=['LPS', 'RAS'],
                        help='Coordinate system for exported STL and markups (default: LPS, '
                             'so they overlay the NIfTI masks in 3D Slicer)')
    parser.add_argument('--export_slicer_markups', type=str, default=None, choices=['True', 'False'],
                        help='If True, also export landmarks as 3D Slicer fiducial markups (.mrk.json)')
    parser.add_argument('--output_dir_slicer_markups', type=str, default=None,
                        help='Output directory for 3D Slicer markups (defaults to the markups directory)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.fh_plane_lm is not None:
        FH_plane_lm = args.fh_plane_lm
    if args.predicted_landmarks is not None:
        predicted_landmarks = args.predicted_landmarks
    if args.masks_dir is not None:
        masks_dir = args.masks_dir
    if args.output_dir_markups is not None:
        output_dir_markups = args.output_dir_markups
    if args.output_dir_markups_no_fh is not None:
        output_dir_markups_no_fh = args.output_dir_markups_no_fh
    if args.output_dir_masks is not None:
        output_dir_masks = args.output_dir_masks
    if args.output_dir_stl is not None:
        output_dir_stl = args.output_dir_stl
    if args.bone_masks_dir is not None:
        bone_masks_dir = args.bone_masks_dir
    if args.output_dir_masks_bone is not None:
        output_dir_masks_bone = args.output_dir_masks_bone
    if args.output_dir_stl_bone is not None:
        output_dir_stl_bone = args.output_dir_stl_bone
    if args.reference_grid_dir is not None:
        reference_grid_dir = args.reference_grid_dir
    if args.coordinate_system is not None:
        output_coordinate_system = args.coordinate_system
    if args.export_slicer_markups is not None:
        export_slicer_markups = args.export_slicer_markups == 'True'
    if args.output_dir_slicer_markups is not None:
        output_dir_slicer_markups = args.output_dir_slicer_markups
    main()