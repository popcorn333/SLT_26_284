#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --job-name=fm
#SBATCH --array=0-4
#SBATCH --time=1-00:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
module purge
module load Anaconda3/2025.06-1
module load GCC/12.3.0
module load CUDA/12.4.0

source activate grad-tts-masking

values=(0.2 0.4 0.6 0.8 1)
value="${values[$SLURM_ARRAY_TASK_ID]}"

# metrics-sweep-overrides: model.masking.a=0.2,0.4,0.6,0.8,1
HYDRA_FULL_ERROR=1 python train.py -m --config-name=config_swp +data=data_swp model.masking.a="$value"
HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp +data=data_swp model.masking.a="$value"
