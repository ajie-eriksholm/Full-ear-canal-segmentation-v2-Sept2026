#!/bin/bash
#$ -N evaluate_nnunet         # Job name
#$ -cwd                    # Run in current working directory
#$ -l host="kbnuxerhc*"        # Request 64 GB RAM
#$ -l cores=4           # Request 16 CPU cores
#$ -l mem_free=64G      
#$ -o features2_out.log    # Standard output log
#$ -e features2_error.log     # Standard error log

echo "Activating environment..."
source ~/code/nnunetv2_env/bin/activate

# ---- PATHS ----
export nnUNet_raw=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_raw
export nnUNet_preprocessed=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_preprocessed
export nnUNet_results=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results


nnUNetv2_evaluate_folder  $nnUNet_raw/Dataset002_Ear/labelsTs $nnUNet_results/Dataset002_Ear/predictions_test -djfile $nnUNet_results/Dataset002_Ear/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/dataset.json -pfile $nnUNet_results/Dataset002_Ear/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/plans.json
# nnUNetv2_predict \
#   -i $nnUNet_raw/Dataset001_Ear/imagesTs \
#   -o $nnUNet_results/Dataset001_Ear/predictions_fold0_best2 \
#   -d 1 \
#   -c 3d_fullres \
#   -f 0 \
#   -tr nnUNetTrainerNoMirroring \
#   -p nnUNetResEncUNetLPlans \
#   -chk best




echo "Validation completed for fold 0."