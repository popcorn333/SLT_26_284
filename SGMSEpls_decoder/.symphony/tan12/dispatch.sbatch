#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --job-name=tan12_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12/logs/dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse/.symphony/tan12/logs/dispatch-%j.err

set -euo pipefail
root=/mnt/parscratch/users/acp23xt/private/DTDM_sgmse
state="$root/.symphony/tan12"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$state/cache"
export TMPDIR="$state/tmp/dispatch-${SLURM_JOB_ID}"
export TAN7_SCOPE=tan12
export TAN7_STATE="$state"
export TAN7_RUN_JOBS="$state/run_jobs.tsv"
export TAN7_TRAINING_LOSS="$state/training_loss_tan12.log"
export TAN7_EXTRACT_LOSSES="$state/extract_losses.py"
export TAN7_JOB_PREFIX=tan12
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$state/logs"
cd "$root"
for task in 11108873_0 11108874_0 11108874_1 11108874_2 11108874_3 11108874_4; do
  status=$(sacct -X -j "$task" --format=State,ExitCode -n -P | head -1)
  if [[ "$status" != "COMPLETED|0:0" ]]; then
    "$state/linear_update.sh" failure "run-array task $task=$status" || true
    exit 1
  fi
done
if ! /mnt/parscratch/users/acp23xt/private/conda/envs/codex_utmosv2/bin/python \
  .symphony/tan7/dispatch.py; then
  "$state/linear_update.sh" failure "downstream dispatch" || true
  exit 1
fi
"$state/linear_update.sh" dispatch
