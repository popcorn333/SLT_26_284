#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --job-name=tan7_metric_recover
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/metric-recover-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7/logs/metric-recover-%j.err

set -euo pipefail
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$root/.t/d-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
cd "$root"
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python \
  .symphony/tan7/retry_metrics_dispatch.py
