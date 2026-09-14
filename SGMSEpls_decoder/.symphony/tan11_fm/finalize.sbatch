#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:15:00
#SBATCH --job-name=tan11_fm_finalize
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/finalize-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan11_fm/logs/finalize-%j.err

set -euo pipefail
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan11_fm"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$state/cache"
export TMPDIR="$state/tmp/finalize-${SLURM_JOB_ID}"
export TAN7_SCOPE=tan11_fm
export TAN7_STATE="$state"
export TAN7_TRAINING_LOSS="$state/training_loss_fm.log"
export TAN7_RESULT_PREFIX=tan11_fm
export TAN7_FINALIZER_BARRIERS="$state/finalizer_barriers.tsv"
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
cd "$root"
python_bin=/mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python
"$python_bin" .symphony/tan7/finalize.py
"$python_bin" "$state/merge_training_losses.py"
"$python_bin" "$state/merge_results.py"
"$python_bin" "$state/final_audit.py"
