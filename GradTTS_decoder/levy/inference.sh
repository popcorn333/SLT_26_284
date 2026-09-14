#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --job-name=levy
#SBATCH --time=0-05:00:00
export SLURM_EXPORT_ENV=ALL
module purge
module load Anaconda3/2025.06-1
module load GCC/12.3.0
module load CUDA/12.4.0

source activate grad-tts-masking
HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp +data=data_swp  model.masking.alpha=1.9
~                                                                                                                                                                   
~                                                                                                                    
