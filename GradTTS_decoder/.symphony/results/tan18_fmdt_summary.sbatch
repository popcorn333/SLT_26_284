#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --job-name=tan18_fmdt_summary
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-summary-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-summary-%j.err
set -euo pipefail
ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
/usr/bin/python3 "$RES/tan18_fmdt_summarize.py"


