#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=tan10_utmos_cache
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/utmos-cache-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/utmos-cache-%j.err
set -euo pipefail
module purge
module load Anaconda3/2022.10
module load CUDA/12.4.0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache
export XDG_CONFIG_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.config
export XDG_DATA_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.local/share
export HF_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/huggingface
export TORCH_HOME=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/torch
export UTMOSV2_CHACHE=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.cache/utmosv2
export TMPDIR=/mnt/parscratch/users/acp23xt/private/codex_utmosv2/.tmp
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$UTMOSV2_CHACHE"
cd /mnt/parscratch/users/acp23xt/private/codex_utmosv2
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python - <<'PY'
import utmosv2
utmosv2.create_model(pretrained=True, device="cuda:0")
print("UTMOSv2 model and dependencies cached successfully", flush=True)
PY
