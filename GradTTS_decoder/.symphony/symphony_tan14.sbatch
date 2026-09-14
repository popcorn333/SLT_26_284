#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=symphony-tan14
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/symphony-tan14-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/symphony-tan14-%j.err
set -euo pipefail
. /opt/apps/testapps/common/software/staging/Anaconda3/2022.10/etc/profile.d/conda.sh
conda activate /mnt/parscratch/users/acp23xt/private/conda/envs/symphony
export LC_ALL=en_GB.utf8 LANG=en_GB.utf8
export MIX_REBAR3=/mnt/parscratch/users/acp23xt/private/conda/envs/symphony/bin/rebar3
: "${LINEAR_API_KEY:?LINEAR_API_KEY is required}"
cd /mnt/parscratch/users/acp23xt/private/symphony/elixir
exec mise exec -- ./bin/symphony --i-understand-that-this-will-be-running-without-the-usual-guardrails /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/WORKFLOW_02_aug.md --logs-root /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/log-tan14
