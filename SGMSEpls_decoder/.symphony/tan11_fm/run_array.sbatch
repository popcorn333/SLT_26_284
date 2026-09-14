#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --job-name=tan11_fm_array
#SBATCH --array=0-4
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/run-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/run-%A_%a.err
#SBATCH --time=1-00:00:00
export SLURM_EXPORT_ENV=ALL
set -euo pipefail
values=(0.2 0.4 0.6 0.8 1)
value="${values[${SLURM_ARRAY_TASK_ID}]}"
cd /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/fm
module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0

source activate CUDA124_sgmsetts

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC=$(which gcc)
export CXX=$(which g++)

HYDRA_FULL_ERROR=1 python train.py -m --config-name=config_swp +data=data_swp  model.masking.a="$value"
HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp +data=data_swp  model.masking.a="$value"
