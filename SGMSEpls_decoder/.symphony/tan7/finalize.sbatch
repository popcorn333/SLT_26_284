#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=tan7_finalize
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/finalize-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/finalize-%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/cache
export TMPDIR=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/tmp/finalize-${SLURM_JOB_ID}
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd /mnt/parscratch/users/acp23xt/private/DTDM_sgmse
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python .symphony/tan7/finalize.py
