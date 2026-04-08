import os
import json
import struct
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from skimage import measure
import argparse

# === Configuration ===
FH_plane_lm = "/projects/oticon/erhdata/Processed-Data/AJIE/tcia/all/Output/Aligned_Landmarks/landmark_positions_after_cropping.csv"

predicted_landmarks = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Inference_results/run_20260210_105409/test_predictions/predicted_landmark_coordinates.csv"
masks_dir = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Inference_results/run_20260210_105409/test_predictions"

output_dir_markups = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/markups"
output_dir_markups_no_fh = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/markups"
output_dir_masks = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/masks"
output_dir_stl = "/projects/oticon/erhdata/Processed-Data/SBEO/tcia/Output/Results/stl"

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
    ("right_ear", 8):  (8, "Bottom Tragus"),
    ("right_ear", 10): (9, "Top Tragus"),
    ("right_ear", 12): (10, "Eye Orbit"),
    ("left_ear", 9):   (8, "Bottom Tragus"),
    ("left_ear", 11):  (9, "Top Tragus"),
    ("left_ear", 13):  (10, "Eye Orbit"),
}


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


def process_masks():
    os.makedirs(output_dir_masks, exist_ok=True)
    os.makedirs(output_dir_stl, exist_ok=True)

    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith("_pred_seg.nii.gz"))
    if not mask_files:
        print("No *_pred_seg.nii.gz files found in masks_dir.")
        return

    processed = 0
    for mask_file in mask_files:
        base_name = mask_file.replace("_pred_seg.nii.gz", "")
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
        verts_mm = (nii.affine @ verts_h.T).T[:, :3]
        save_stl_binary(verts_mm, faces_mc, os.path.join(output_dir_stl, f"{base_name}.stl"))

        processed += 1
        print(f"  Processed mask: {base_name}")

    print(f"Processed {processed} masks -> NIfTI: '{output_dir_masks}', STL: '{output_dir_stl}'")


def main():
    # --- Determine output directory based on FH plane CSV availability ---
    fh_available = os.path.exists(FH_plane_lm)
    active_markups_dir = output_dir_markups if fh_available else output_dir_markups_no_fh
    if not fh_available:
        print(f"WARNING: FH plane landmarks CSV not found: {FH_plane_lm}")
        print(f"Markups will be saved (canal landmarks only) to: {active_markups_dir}")
    os.makedirs(active_markups_dir, exist_ok=True)

    # --- Load predicted canal landmarks (1-7) ---
    pred_df = pd.read_csv(predicted_landmarks)
    # scan_name looks like "CHUM-001_left_ear.nii.gz"
    # Extract patient and ear_side
    pred_df["patient"] = pred_df["scan_name"].str.replace(".nii.gz", "", regex=False).str.rsplit("_", n=2).str[0]
    pred_df["ear_side"] = pred_df["scan_name"].str.replace(".nii.gz", "", regex=False).str.rsplit("_", n=2).apply(lambda parts: "_".join(parts[1:]))

    # --- Load FH plane landmarks (8-13) ---
    if fh_available:
        fh_df = pd.read_csv(FH_plane_lm)
        fh_grouped = fh_df.groupby(["scan_name", "ear_side"])
    else:
        fh_grouped = {}  # empty — no FH landmarks will be added
    # Columns: scan_name, ear_side, landmark_id, x_mm, y_mm, z_mm, is_inside_roi

    # --- Build JSON per patient ear ---
    # Group predicted landmarks by (patient, ear_side)
    pred_grouped = pred_df.groupby(["patient", "ear_side"])

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

        # Sort by numeric id
        landmarks.sort(key=lambda lm: int(lm["id"]))

        markup = {
            "landmarks": landmarks,
            "count": len(landmarks)
        }

        # Save JSON
        json_filename = f"{patient}_{ear_side}.json"
        json_path = os.path.join(active_markups_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(markup, f, indent=2)

        processed.add((patient, ear_side))

    print(f"Created {len(processed)} markup JSON files in: {active_markups_dir}")

    process_masks()


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
    main()