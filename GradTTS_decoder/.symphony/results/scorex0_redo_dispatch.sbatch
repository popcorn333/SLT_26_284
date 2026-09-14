#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=tan6_scorex0_redo_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scorex0-redo-dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scorex0-redo-dispatch-%j.err
set -euo pipefail
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scorex0_redo_dispatch.py
