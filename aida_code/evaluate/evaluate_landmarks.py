import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
import pandas as pd

def evaluate_landmarks(gt_dir, pred_dir, labels=[3, 4, 5, 6], output_csv="landmark_evaluation.csv"):
    results = []
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])
    
    for filename in gt_files:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        
        if not os.path.exists(pred_path):
            continue

        gt_img = sitk.ReadImage(gt_path)
        pred_img = sitk.ReadImage(pred_path)
        spacing = np.array(gt_img.GetSpacing())[::-1] 
        gt_arr = sitk.GetArrayFromImage(gt_img)
        pred_arr = sitk.GetArrayFromImage(pred_img)

        case_data = {"Patient_ID": filename}
        any_missing = False

        for lbl in labels:
            gt_mask = (gt_arr == lbl)
            pred_mask = (pred_arr == lbl)

            if not np.any(gt_mask):
                case_data[f"L{lbl}_MRE_mm"] = np.nan
                case_data[f"L{lbl}_Status"] = "Missing in GT"
                continue

            if np.any(pred_mask):
                gt_com = np.array(center_of_mass(gt_mask))
                pred_com = np.array(center_of_mass(pred_mask))
                mre = np.sqrt(np.sum(((gt_com - pred_com) * spacing)**2))
                
                case_data[f"L{lbl}_MRE_mm"] = round(mre, 4)
                case_data[f"L{lbl}_Status"] = "Found"
            else:
                case_data[f"L{lbl}_MRE_mm"] = np.nan
                case_data[f"L{lbl}_Status"] = "Missing in Prediction"
                any_missing = True

        case_data["Has_Missing_Landmarks"] = any_missing
        results.append(case_data)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # --- Calculate and Print Averages ---
    print("\n" + "="*30)
    print("      SUMMARY STATISTICS")
    print("="*30)
    
    summary_data = []
    for lbl in labels:
        mre_col = f"L{lbl}_MRE_mm"
        status_col = f"L{lbl}_Status"
        
        avg_mre = df[mre_col].mean() # Automatically ignores NaNs
        std_mre = df[mre_col].std()
        
        # Calculate Detection Rate: (Found / Total where GT exists)
        total_gt = len(df[df[status_col] != "Missing in GT"])
        found = len(df[df[status_col] == "Found"])
        det_rate = (found / total_gt * 100) if total_gt > 0 else 0
        
        summary_data.append({
            "Label": lbl,
            "Avg MRE (mm)": f"{avg_mre:.3f}",
            "Std Dev": f"{std_mre:.3f}",
            "Detection Rate": f"{det_rate:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*30)
    print(f"Full results saved to: {output_csv}")

    return df

evaluate_landmarks("/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_raw/Dataset002_Ear/labelsTs", "/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results/Dataset002_Ear/predictions_test")