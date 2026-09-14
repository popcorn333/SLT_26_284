#!/bin/bash -l
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=symphony-fmdt-0807
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/symphony-fmdt-0807-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/symphony-fmdt-0807-%j.err
set -euo pipefail
export PATH="/mnt/parscratch/users/acp23xt/private/conda/envs/symphony/bin:/users/acp23xt/.local/bin:$PATH"
export LC_ALL=en_GB.utf8 LANG=en_GB.utf8
export MIX_REBAR3=/mnt/parscratch/users/acp23xt/private/conda/envs/symphony/bin/rebar3
: "${LINEAR_API_KEY:?LINEAR_API_KEY is required}"
cd /mnt/parscratch/users/acp23xt/private/symphony/elixir
exec mise exec -- ./bin/symphony \
  --i-understand-that-this-will-be-running-without-the-usual-guardrails \
  /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/WORKFLOW_02_aug.md \
  --logs-root /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/log-fmdt-20260807-restart2 \
  --port 4014
