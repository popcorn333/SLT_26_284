#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:45:00
#SBATCH --job-name=tan10_summary_fix
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary-fix-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary-fix-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/pipeline.py summarize
sed -n '1p' /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/report.md | /usr/bin/grep -Fx '# TAN-10 speech metric summary'
/usr/bin/test "$(/usr/bin/wc -l < /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary.csv)" -eq 35

