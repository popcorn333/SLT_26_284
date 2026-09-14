#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --job-name=tan7_x0r_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan7_scorex0_redo/logs/dispatch-%j.err

set -euo pipefail
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan7_scorex0_redo"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$state/cache"
export TMPDIR="$state/tmp/dispatch-${SLURM_JOB_ID}"
export TAN7_SCOPE=scorex0_redo
export TAN7_STATE="$state"
export TAN7_RUN_JOBS="$state/run_jobs.tsv"
export TAN7_TRAINING_LOSS="$state/training_loss_redo.log"
export TAN7_EXTRACT_LOSSES="$root/.symphony/tan7/extract_losses.py"
export TAN7_FINALIZER_BARRIER_MANIFEST="$root/.symphony/tan7/finalizer_job.tsv"
export TAN7_JOB_PREFIX=tan7_x0r
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$state/logs"
cd "$root"
/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python \
  .symphony/tan7/dispatch.py
