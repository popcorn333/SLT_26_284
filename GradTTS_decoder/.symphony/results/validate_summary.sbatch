#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=tan10_summary
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary-%j.err
set -euo pipefail
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/pipeline.py summarize
