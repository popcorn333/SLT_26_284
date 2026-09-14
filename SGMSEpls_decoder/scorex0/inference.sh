#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1         # Number of GPUs
#SBATCH --cpus-per-task=4
#SBATCH --mem=3G
#SBATCH --job-name=fm
#SBATCH --time=0-00:30:00
export SLURM_EXPORT_ENV=ALL
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

HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp +data=data_swp  model.masking.a=1
