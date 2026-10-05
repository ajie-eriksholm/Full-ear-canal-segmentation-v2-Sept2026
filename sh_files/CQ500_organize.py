import pandas as pd
from pathlib import Path
import shutil

# =====================
# PATHS
# =====================

csv_file = Path("/projects/oticon/erhdata/Processed-Data/AJIE/CT_images.csv")

output_root = Path("/projects/oticon/erhdata/Processed-Data/AJIE/Pre-processed_CT_annotations/CQ500/all_raw")
cq500_root = Path("/projects/oticon/erhdata/Raw/EarScans/Images/Head_CT/CQ500")

output_root.mkdir(parents=True, exist_ok=True)

# =====================
# READ CSV
# =====================

df = pd.read_csv(csv_file)

# Keep only CQ500 rows and valid IMAGE entries
filtered_df = df.loc[df["dataset"].eq("CQ500") & df["IMAGE"].notna()].copy()

# =====================
# COPY FILES
# =====================

copied = 0
missing = []

for row in filtered_df.itertuples(index=False):

    patient = str(row.NAME).strip()
    image_suffix = str(row.IMAGE).strip()

    patient_folder = cq500_root / patient

    if not patient_folder.exists():
        missing.append(f"Missing folder: {patient}")
        continue

    # Example:
    # CT10000 + NR0_resampled
    # -> CT10000NR0_resampled.nii.gz

    target_filename = f"{patient}{image_suffix}.nii.gz"
    source_file = patient_folder / target_filename

    if source_file.exists():

        # Option 1:
        # save directly in output folder
        output_filename = f"{patient}__CT.nii.gz"
        shutil.copy2(
            source_file,
            output_root / output_filename
        )

        copied += 1

    else:
        missing.append(str(source_file))

print(f"Copied {copied} files")

if missing:
    print("\nMissing files:")
    for m in missing:
        print(m)