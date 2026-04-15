#!/bin/bash
#$ -N nnunetv2_PRED
#$ -cwd
#$ -l nvgpu=1
#$ -l gputype=rtx*
#$ -l cores=8
#$ -l mem_free=64G
#$ -l h_rt=06:00:00
#$ -o nnunet_tcia.out
#$ -e nnunet_tcia.err

echo "Activating environment..."
source ~/Full-ear-canal-segmentation/landmark_env/bin/activate

# ---- PATHS ----
export nnUNet_raw=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_raw
export nnUNet_preprocessed=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_preprocessed
export nnUNet_results=/projects/oticon/erhdata/Processed-Data/AJIE/nnunet/nnUNet_results

# ---- VALIDATION ----
# Esto hace la predicción automática de VAL y guarda las métricas.
# Usa el checkpoint BEST del entrenamiento de fold 0.

#nnUNetv2_predict -i $nnUNet_raw/Dataset002_Ear/imagesTs -o $nnUNet_results/Dataset002_Ear/predictions_test -d 1 -c 3d_fullres -f 0 1 2 3 4 -chk checkpoint_best.pth

nnUNetv2_predict -i /projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Output/Preprocessing/P4_Normalized_Ears_nnUNet/ -o /projects/oticon/erhdata/Processed-Data/SBEO/test_scan/Output/Inference/nnUNet/ -d 1 -c 3d_fullres -f 0 1 2 3 4 -chk checkpoint_best.pth -device cpu
# nnUNetv2_predict \
#   -i $nnUNet_raw/Dataset002_Ear/imagesTs \
#   -o $nnUNet_results/Dataset002_Ear/predictions_test \
#   -d 1 \
#   -c 3d_fullres \
#   -f 0 1 2 3 4 \
#   -tr nnUNetTrainerNoMirroring \
#   -p nnUNetResEncUNetLPlans \
#   -chk best




echo "Validation completed for fold 0."