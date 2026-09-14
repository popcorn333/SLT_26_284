#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=tan6_scope_audit
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scope-audit-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/scope-audit-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/completion_audit.py
