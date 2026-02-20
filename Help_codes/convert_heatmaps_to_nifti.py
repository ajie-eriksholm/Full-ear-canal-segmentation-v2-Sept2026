import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd
import json
import shutil

# Directories
nii_dir = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Final_Cropped_Ears_128_Normalized"
heatmap_dir = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Logs/run_20260123_131324/test_predictions"
predictions_csv = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Logs/run_20260123_131324/test_predictions/predicted_landmark_coordinates.csv"
output_dir = "/projects/oticon/erhdata/Processed-Data/SBEO/Final_pipeline/Output/Heatmaps_as_NIfTI_test_predictions"
os.makedirs(output_dir, exist_ok=True)

# Automatically discover all heatmap files in the test predictions directory
print("Scanning for heatmap files...")
heatmap_files = [f for f in os.listdir(heatmap_dir) if f.endswith('_pred_heatmaps.pt')]
SCAN_NAMES = [f.replace('_pred_heatmaps.pt', '') for f in heatmap_files]
print(f"Found {len(SCAN_NAMES)} heatmap files to process")

landmark_names = ['C_Point', 'B_Point', 'Eardrum', 'Top_RS', 'Bottom_RS', 'First_Bend', 'Second_Bend']

# Mapping from internal names to 3D Slicer labels
landmark_labels = {
    'C_Point': 'Cartilaginous Canal Point',
    'B_Point': 'Bony Canal Point',
    'Eardrum': 'Eardrum',
    'Top_RS': 'Top RS',
    'Bottom_RS': 'Bottom RS',
    'First_Bend': '1st bend',
    'Second_Bend': '2nd bend'
}

# Load predicted landmark coordinates
print("Loading predicted landmark coordinates...")
predictions_df = pd.read_csv(predictions_csv)
print(f"Loaded {len(predictions_df)} landmark predictions")

# Process each scan
for SCAN_NAME in SCAN_NAMES:
    print(f"\n{'='*60}")
    print(f"Processing: {SCAN_NAME}")
    print('='*60)
    
    heatmap_file = f"{SCAN_NAME}_pred_heatmaps.pt"
    seg_file = f"{SCAN_NAME}_pred_seg.nii.gz"
    base_name = f"{SCAN_NAME}.nii.gz"

    heatmap_path = os.path.join(heatmap_dir, heatmap_file)
    seg_path = os.path.join(heatmap_dir, seg_file)
    nii_path = os.path.join(nii_dir, base_name)

    if not os.path.exists(heatmap_path):
        print(f"⚠ Warning: Could not find {heatmap_file}, skipping...")
        continue

    if not os.path.exists(nii_path):
        print(f"⚠ Warning: Could not find {base_name}, skipping...")
        continue

    # Load original image to get affine
    original_img = nib.load(nii_path)
    affine = original_img.affine
    
    # Copy the CT scan to output directory
    ct_output_path = os.path.join(output_dir, base_name)
    shutil.copy2(nii_path, ct_output_path)
    print(f"  ✓ Copied CT scan: {base_name}")
    
    # Copy the segmentation mask if it exists
    if os.path.exists(seg_path):
        seg_output_path = os.path.join(output_dir, seg_file)
        shutil.copy2(seg_path, seg_output_path)
        print(f"  ✓ Copied segmentation mask: {seg_file}")
    else:
        print(f"  ⚠ Warning: Segmentation mask not found: {seg_file}")
    
    # Create 3D Slicer markup file from predicted coordinates
    # Match scan name with .nii.gz extension in the CSV
    scan_predictions = predictions_df[predictions_df['scan_name'] == base_name]
    
    if len(scan_predictions) > 0:
        control_points = []
        
        for idx, landmark_name in enumerate(landmark_names, start=1):
            # landmark_id is 1-7 corresponding to the landmark order
            landmark_data = scan_predictions[scan_predictions['landmark_id'] == idx]
            
            if len(landmark_data) == 1:
                row = landmark_data.iloc[0]
                control_point = {
                    "id": str(idx),
                    "label": landmark_labels[landmark_name],
                    "description": "",
                    "associatedNodeID": "",
                    "position": [
                        -float(row['x_mm']),
                        -float(row['y_mm']),
                        float(row['z_mm'])
                    ],
                    "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                    "selected": True,
                    "locked": False,
                    "visibility": True,
                    "positionStatus": "defined"
                }
                control_points.append(control_point)
        
        # Create the full markup structure
        markup_data = {
            "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [
                {
                    "type": "Fiducial",
                    "coordinateSystem": "LPS",
                    "coordinateUnits": "mm",
                    "locked": False,
                    "fixedNumberOfControlPoints": False,
                    "labelFormat": "%N-%d",
                    "lastUsedControlPointNumber": len(control_points),
                    "controlPoints": control_points,
                    "measurements": [],
                    "display": {
                        "visibility": True,
                        "opacity": 1.0,
                        "color": [0.4, 1.0, 1.0],
                        "selectedColor": [1.0, 0.5000076295109483, 0.5000076295109483],
                        "activeColor": [0.4, 1.0, 0.0],
                        "propertiesLabelVisibility": False,
                        "pointLabelsVisibility": True,
                        "textScale": 3.0,
                        "glyphType": "Sphere3D",
                        "glyphScale": 3.0,
                        "glyphSize": 5.0,
                        "useGlyphScale": True,
                        "sliceProjection": False,
                        "sliceProjectionUseFiducialColor": True,
                        "sliceProjectionOutlinedBehindSlicePlane": False,
                        "sliceProjectionColor": [1.0, 1.0, 1.0],
                        "sliceProjectionOpacity": 0.6,
                        "lineThickness": 0.2,
                        "lineColorFadingStart": 1.0,
                        "lineColorFadingEnd": 10.0,
                        "lineColorFadingSaturation": 1.0,
                        "lineColorFadingHueOffset": 0.0,
                        "handlesInteractive": False,
                        "translationHandleVisibility": True,
                        "rotationHandleVisibility": True,
                        "scaleHandleVisibility": False,
                        "interactionHandleScale": 3.0,
                        "snapMode": "toVisibleSurface"
                    }
                }
            ]
        }
        
        # Save markup file
        markup_filename = f"{SCAN_NAME}_landmarks.mrk.json"
        markup_path = os.path.join(output_dir, markup_filename)
        with open(markup_path, 'w') as f:
            json.dump(markup_data, f, indent=2)
        print(f"  ✓ Saved markup: {markup_filename}")
    else:
        print(f"  ⚠ Warning: No landmark predictions found for {SCAN_NAME}")

    
    # Load heatmaps
    heatmaps = torch.load(heatmap_path)

    # Save each landmark heatmap as separate NIfTI
    for i, landmark_name in enumerate(landmark_names):
        heatmap_np = heatmaps[i].numpy()
        
        # Create NIfTI image with same affine as original CT
        nii_img = nib.Nifti1Image(heatmap_np, affine)
        
        # Save with descriptive name
        output_name = base_name.replace('.nii.gz', f'_heatmap_{landmark_name}.nii.gz')
        output_path = os.path.join(output_dir, output_name)
        nib.save(nii_img, output_path)
        print(f"  ✓ Saved: {output_name}")

    print(f"✓ Completed: {SCAN_NAME}")

print(f"\n{'='*60}")

print(f"\nAll heatmaps saved to: {output_dir}")
print("You can now open these .nii.gz files in 3D Slicer")
