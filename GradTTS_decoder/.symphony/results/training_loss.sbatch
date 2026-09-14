#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan10_training_loss
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/training-loss-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/training-loss-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/aggregate_training_loss.py
