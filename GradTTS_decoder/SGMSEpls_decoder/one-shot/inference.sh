#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --job-name=one-shot
#SBATCH --time=0-3:00:00
export SLURM_EXPORT_ENV=ALL
module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0

source activate grad-tts-masking

HYDRA_FULL_ERROR=1 python inference_3.py -m --config-name=config_eval_swp +data=data_swp  model.masking.a=0.2
