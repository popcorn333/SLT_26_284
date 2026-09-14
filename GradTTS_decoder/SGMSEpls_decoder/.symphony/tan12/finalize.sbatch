#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan12_finalize
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12/logs/finalize-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12/logs/finalize-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12/cache
export TMPDIR=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12/tmp/finalize-${SLURM_JOB_ID}
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /mnt/parscratch/users/acp23xt/private/DTDM_sgmse
export TAN7_SCOPE=tan12
export TAN7_STATE=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12
export TAN7_TRAINING_LOSS=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12/training_loss_tan12.log
export TAN7_RESULT_PREFIX=tan12
if ! /mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python .symphony/tan7/finalize.py; then
  "$TAN7_STATE/linear_update.sh" failure "final coverage audit" || true
  exit 1
fi
"$TAN7_STATE/linear_update.sh" complete
