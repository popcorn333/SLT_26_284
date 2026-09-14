#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --job-name=tan18_all_bridge
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-all-bridge-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-all-bridge-%j.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
set -euo pipefail

RESULTS=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results
SUMMARY_ID=$(tr -d '[:space:]' < "$RESULTS/tan18_fmdt_summary_job_id.txt")
if [[ ! "$SUMMARY_ID" =~ ^[0-9]+$ ]]; then
    echo "invalid fmDT summary job ID: $SUMMARY_ID" >&2
    exit 1
fi

FINALIZER_ID=$(sbatch --parsable \
    --dependency="afterok:$SUMMARY_ID:$SLURM_JOB_ID" \
    "$RESULTS/tan18_all_finalize.sbatch")
printf '%s\n' "$FINALIZER_ID" > "$RESULTS/tan18_all_finalize_job_id.txt"
printf 'fmDT_summary=%s all_folder_finalizer=%s\n' \
    "$SUMMARY_ID" "$FINALIZER_ID"
