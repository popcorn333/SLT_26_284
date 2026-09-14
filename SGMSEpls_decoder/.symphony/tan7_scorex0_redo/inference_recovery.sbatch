#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --job-name=tan7_x0r_inf
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/inference-recovery-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/inference-recovery-%j.err

set -euo pipefail

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

# Training for all five configurations completed in allocation 10943687.
# Resume only inference so completed fresh checkpoints and loss logs remain intact.
HYDRA_FULL_ERROR=1 python inference.py -m --config-name=config_eval_swp \
  +data=data_swp model.masking.a=1,0.8,0.6,0.4,0.2
