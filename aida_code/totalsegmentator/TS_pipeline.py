# This script processes CT scans using TotalSegmentator to extract skull and mandible segmentations, then merges them into a single 2-label segmentation for each patient.
# The output segmentations are saved in the "training_set/all_images_segmentation/" directory with filenames like "CHUP-012.nii-label.nii.gz".

#import libraries
import os
import subprocess
import SimpleITK as sitk


# Paths
CT_DIR = "training_set/CTs_missing/"
OUT_BASE = "training_set/all_images_segmentation/"

os.makedirs(OUT_BASE, exist_ok=True)

#run TotalSegmentator via command line
def run_totalseg(input_ct, output_folder):
    """Run TotalSegmentator via command line."""
    cmd = [
        "TotalSegmentator",
        "-i", input_ct,
        "-o", output_folder,
        "-ta", "craniofacial_structures"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# Merge mandible and skull into one segmentation with 2 labels
def merge_mandible_skull(seg_dir, pid):
    """
    Merge mandible and skull masks into one 2-label segmentation.
      - Skull     -> label 1
      - Mandible  -> label 2
    Save output as: <pid>-label.nii.gz
    """
    skull_path     = os.path.join(seg_dir, "skull.nii.gz")
    mandible_path  = os.path.join(seg_dir, "mandible.nii.gz")

    if not (os.path.exists(skull_path) and os.path.exists(mandible_path)):
        print(f"Missing mandible or skull for {pid}, skipping.")
        return

    skull = sitk.ReadImage(skull_path)
    mandible = sitk.ReadImage(mandible_path)

    skull_bin = sitk.Cast(skull > 0, sitk.sitkUInt8) * 1
    mandible_bin = sitk.Cast(mandible > 0, sitk.sitkUInt8) * 2

    merged = skull_bin + mandible_bin

    out_path = os.path.join(OUT_BASE, f"{pid}.nii-label.nii.gz")
    sitk.WriteImage(merged, out_path)

    print("Saved merged label:", out_path)

#extract patient ID from filename pattern: CHUP-012__CT.nii.gz → CHUP-012
def extract_pid(filename):
    """Extract patient ID from pattern: CHUP-012__CT.nii.gz → CHUP-012"""
    return filename.split("__")[0]


for f in os.listdir(CT_DIR):
    if f.endswith("__CT.nii.gz"):
        ct_path = os.path.join(CT_DIR, f)
        pid = extract_pid(f)

        print("\n==============================")
        print("PROCESSING:", pid)
        print("==============================")

        # Output folder for TotalSegmentator
        out_folder = os.path.join(OUT_BASE, f"RAW_{pid}")
        os.makedirs(out_folder, exist_ok=True)

        # 1. Run TotalSegmentator
        run_totalseg(ct_path, out_folder)

        # 2. Merge skull + mandible
        merge_mandible_skull(out_folder, pid)