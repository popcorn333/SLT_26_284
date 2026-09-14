#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --job-name=tan7_x0r_inf_a
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/inference-array-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/inference-array-%A_%a.err

set -euo pipefail

values=(1 0.8 0.6 0.4 0.2)
value="${values[$SLURM_ARRAY_TASK_ID]}"
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
cd "$root/scorex0"

module purge
module load Anaconda3/2022.10
module load GCC/12.3.0
module load CUDA/12.4.0
source activate CUDA124_sgmsetts

export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC
CC=$(which gcc)
export CXX
CXX=$(which g++)

# Allocation 10943687 completed training for every configuration. Each task
# resumes one independent inference configuration using those fresh checkpoints.
# Keep Hydra in multirun mode even though each array task selects only one
# value.  config_eval_swp deliberately uses a sweep directory two levels
# below scorex0, and its data/checkpoint/code paths are relative to that
# layout.  A single-run invocation instead chdirs into ../hydra_outputs and
# makes those established paths resolve outside scorex0.
HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp \
  +data=data_swp "model.masking.a=$value"
