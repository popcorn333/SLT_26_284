#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=tan7_x0r_inf_ok
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/inference-barrier-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/inference-barrier-%j.err

set -euo pipefail
echo "All scorex0 inference-array tasks completed successfully."
