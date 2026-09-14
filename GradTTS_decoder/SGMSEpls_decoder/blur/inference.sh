#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --job-name=blur
#SBATCH --time=0-10:00:00
export SLURM_EXPORT_ENV=ALL
module purge
if module is-avail Anaconda3/2025.06-1; then
    module load Anaconda3/2025.06-1
fi
module load GCC/12.3.0
module load CUDA/12.4.0

SGMSE_ENV=/mnt/parscratch/users/acp23xt/private/conda/envs/CUDA124_sgmsetts
export PATH="$SGMSE_ENV/bin:$PATH"

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export CC=$(which gcc)
export CXX=$(which g++)

HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp +data=data_swp  model.masking.b=10,20,40

