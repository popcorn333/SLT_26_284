#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan20_finalize
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/finalize-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan20/logs/finalize-%j.err

set -euo pipefail
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan20"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$state/cache"
export TMPDIR="$root/.t/t20z-${SLURM_JOB_ID}"
export TAN7_SCOPE=tan12
export TAN7_STATE="$state"
export TAN7_TRAINING_LOSS="$state/training_loss_tan20.log"
export TAN7_RESULT_PREFIX=tan20
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd "$root"
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python \
  .symphony/tan20/merge_training_losses.py
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python \
  .symphony/tan7/finalize.py
