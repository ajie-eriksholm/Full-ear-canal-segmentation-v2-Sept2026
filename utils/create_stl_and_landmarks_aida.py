import nrrd
import nibabel as nib
import json
import numpy as np
from skimage import measure
from scipy.ndimage import gaussian_filter
import trimesh
import os


# =========================
# CONFIG
# =========================
# MASK_DIR   = "/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results/Dataset001_Ear/test_and_train"
# OUTPUT_DIR = "/projects/oticon/erhdata/Processed-Data/AJIE/stl_and_landmarks"


# MASK_DIR   = "/projects/oticon/erhdata/Processed-Data/AJIE/stl_and_landmarks/masks_missing_landmarks"
# OUTPUT_DIR = "/projects/oticon/erhdata/Processed-Data/AJIE/stl_and_landmarks_missing"

MASK_DIR   = "/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results/Dataset001_Ear/tcia_corrected"
OUTPUT_DIR = "/projects/oticon/erhdata/Processed-Data/AJIE/stl_and_landmarks_tcia_corrected"

LABELS_STL = [1, 2]          # Labels that form the STL
LABELS_LANDMARKS = [3, 4, 5, 6]  # Labels for CBJ1–CBJ4

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOADING MASKS
# =========================
def read_mask(path):
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        img = nib.load(path)
        data = img.get_fdata()
        spacing = np.array(img.header.get_zooms()[:3])
        return data, spacing
    else:
        data, hdr = nrrd.read(path)
        if "space directions" in hdr:
            sd = hdr["space directions"]
            spacing = np.array([sd[0][0], sd[1][1], sd[2][2]])
        else:
            spacing = np.array([1,1,1])
        return data, spacing


# =========================
# CENTROID OF LABEL
# =========================
def extract_centroid(mask, label, spacing):
    coords = np.argwhere(mask == label)
    if len(coords) == 0:
        return None
    pts_mm = coords * spacing
    return pts_mm.mean(axis=0).tolist()


# =========================
# CREATE STL FROM LABELS 1+2
# =========================
def create_stl(mask, spacing, output_path):
    print(f"Creating STL → {output_path}")

    binary = np.isin(mask, LABELS_STL).astype(float)

    smooth = gaussian_filter(binary, sigma=1.5)

    verts, faces, normals, _ = measure.marching_cubes(
        smooth, level=0.5, spacing=spacing
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    trimesh.smoothing.filter_laplacian(mesh, iterations=5)

    mesh.export(output_path)

    print("✓ STL saved:", output_path)
    return mesh


# =========================
# CREATE JSON WITH CBJ1–CBJ4
# =========================
def create_cbj_json(mask_data, spacing, output_json_path):
    label_map = {
        3: ("1", "CBJ1"),
        4: ("2", "CBJ2"),
        5: ("3", "CBJ3"),
        6: ("4", "CBJ4"),
    }

    landmarks = []

    for lab, (id_str, lbl) in label_map.items():
        pos = extract_centroid(mask_data, lab, spacing)
        landmarks.append({
            "id": id_str,
            "label": lbl,
            "position": pos
        })

    output = {
        "landmarks": landmarks,
        "count": len(landmarks)
    }

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=2)

    print("✓ CBJ landmarks JSON saved:", output_json_path)
    return output


# =========================
# PROCESS ONE PATIENT
# =========================
def process_one(patient_id, filename):
    print("\n=================================================")
    print("PROCESSING:", patient_id)
    print("=================================================")

    mask_path = os.path.join(MASK_DIR, filename)
    mask, spacing = read_mask(mask_path)

    print("Mask loaded:", mask.shape, "Spacing:", spacing)

    # --- STL ---
    stl_path = os.path.join(OUTPUT_DIR, f"{patient_id}.stl")
    create_stl(mask, spacing, stl_path)

    # --- CBJ JSON ---
    json_path = os.path.join(OUTPUT_DIR, f"{patient_id}_CBJ.json")
    create_cbj_json(mask, spacing, json_path)

    print(f"DONE: {patient_id}")
    return True


# =========================
# MAIN BATCH
# =========================
def main():

    files = sorted([f for f in os.listdir(MASK_DIR)
                    if f.endswith((".nii.gz",".nii",".nrrd"))])

    print(f"Found {len(files)} mask files")

    for f in files:
        if f.endswith(".nii.gz"):
            pid = f[:-7]
        elif f.endswith(".nii"):
            pid = f[:-4]
        else:
            pid = f[:-5]

        process_one(pid, f)

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()