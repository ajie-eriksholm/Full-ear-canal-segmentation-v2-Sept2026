#!/bin/bash
#$ -N nnunetv2_fold0
#$ -cwd
#$ -l nvgpu=1
#$ -l gputype=rtx*
#$ -l cores=8
#$ -l mem_free=64G
#$ -l h_rt=06:00:00
#$ -o fold0_out_new.out
#$ -e fold0_out_new.err

echo "Activating environment..."
source ~/code/nnunetv2_env/bin/activate

# nnU-Net environment variables
export nnUNet_raw=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_raw/
export nnUNet_preprocessed=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_preprocessed/
export nnUNet_results=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results/

nnUNetv2_extract_fingerprint -d 2 --verify_dataset_integrity --verbose -pl nnUNetPlannerResEncL -c 3d_fullres

echo "fingerprint done."

nnUNetv2_plan_experiment -d 2 -c 3d_fullres -pl nnUNetPlannerResEncL -np 4

echo "plan done."

nnUNetv2_preprocess -d 2 -c 3d_fullres -pl nnUNetResEncUNetLPlans -np 8

echo "preprocess done."

echo "Training fold 0..."
nnUNetv2_train 2 3d_fullres 0 \
    -tr nnUNetTrainerNoMirroring \
    -p nnUNetResEncUNetLPlans \
    --c

echo "Fold 0 training finished."