#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=tan6_final_audit30
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/final-audit-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/final-audit-%j.err
set -euo pipefail
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/final_audit.py
