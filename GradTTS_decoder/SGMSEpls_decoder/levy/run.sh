#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a100:1    # Established experiment GPU class
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --job-name=levy
#SBATCH --time=1-00:00:00
#SBATCH --array=0-0
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/logs/levy-run-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/logs/levy-run-%A_%a.err
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
SGMSE_ENV=/mnt/parscratch/users/acp23xt/private/conda/envs/CUDA124_sgmsetts
TAN20_STATE=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20
export CUDA_HOME=/opt/apps/testapps/el7-znver3/software/staging/CUDA/12.4.0
export PATH="$SGMSE_ENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$SGMSE_ENV/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="$TAN20_STATE/cache"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
# Keep multiprocessing's generated AF_UNIX socket below Linux's 108-byte limit.
export TMPDIR="/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.t/t20l-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"

cd /mnt/parscratch/users/acp23xt/private/DTDM_sgmse/levy
configs=(1.8)
config="${configs[$SLURM_ARRAY_TASK_ID]}"
"$SGMSE_ENV/bin/python" -B -c 'import torch; assert torch.cuda.is_available(); print("cuda=" + torch.cuda.get_device_name(0), flush=True)'
HYDRA_FULL_ERROR=1 "$SGMSE_ENV/bin/python" -B train.py -m --config-name=config_swp +data=data_swp model.masking.alpha="$config"
HYDRA_FULL_ERROR=1 "$SGMSE_ENV/bin/python" -B inference.py -m --config-name=config_eval_swp +data=data_swp model.masking.alpha="$config"
