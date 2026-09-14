#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=fmdt_env_2025_probe
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/fmdt-env-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/fmdt-env-%j.err
set -euo pipefail

# A login shell initializes the module function for the allocated GPU node.
# This deliberately avoids inheriting Symphony's CPU-node Lmod installation,
# which is not mounted on every GPU node.
module purge
module load Anaconda3/2025.06-1
module load GCC/12.3.0
module load CUDA/12.4.0
source activate grad-tts-masking
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the grad-tts-masking environment")
print("fmDT environment probe passed", flush=True)
print("python={}".format(sys.executable), flush=True)
print("torch={} cuda={}".format(torch.__version__, torch.version.cuda), flush=True)
print("gpu={}".format(torch.cuda.get_device_name(0)), flush=True)
PY
