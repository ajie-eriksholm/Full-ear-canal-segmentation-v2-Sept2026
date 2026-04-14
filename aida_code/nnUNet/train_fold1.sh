#!/bin/bash
#$ -N nnunetv2_fold1
#$ -cwd
#$ -l nvgpu=1
#$ -l gputype=rtx*
#$ -l cores=8
#$ -l mem_free=64G
#$ -l h_rt=06:00:00
#$ -o fold1_out_new1.out
#$ -e fold1_out_new1.err

echo "Activating environment..."
source ~/code/nnunetv2_env/bin/activate

pip install nnunetv2
# nnU-Net environment variables
export nnUNet_raw=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_raw/
export nnUNet_preprocessed=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_preprocessed/
export nnUNet_results=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results/

echo "Training fold 1..." # do the same for the other folders
nnUNetv2_train 2 3d_fullres 1 \
    -tr nnUNetTrainerNoMirroring \
    -p nnUNetResEncUNetLPlans

echo "Fold 1 training finished."