#!/bin/bash
#$ -N P2_Final_Preprocessing_date.pth # Job name
#$ -cwd                    # Run in current working directory
#$ -l nvgpu=1             # Request 1 NVIDIA GPU
#$ -l gputype=rtx*         # Request RTX series GPU
#$ -l cores=16           # Request 16 CPU cores
#$ -l mem_free=64G       # Request 64 GB of RAM
#$ -o my_gpu_output_P2_Final_Preprocessing_date.pth.log    # Standard output log
#$ -e my_gpu_error_P2_Final_Preprocessing_date.pth.log     # Standard error log


cd ~/Documents
source seg_env/bin/activate
python Final_Preprocessing/P2_Final_Preprocessing.py
#python Final_Preprocessing/P3_Final_Preprocessing.py
kill $$