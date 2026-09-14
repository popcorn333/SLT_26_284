#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --job-name=tan18_fmdt_run_r3
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/fmdt-run-%A_%a.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/fmdt-run-%A_%a.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT
set -euo pipefail

# Preserve fmDT/run.sh as the authoritative five-configuration payload.  A
# fresh login shell initializes Lmod on the allocated GPU node without relying
# on Symphony's CPU-node exported shell functions.
exec /bin/bash -l /mnt/parscratch/users/acp23xt/private/DTDM_Unet/fmDT/run.sh
