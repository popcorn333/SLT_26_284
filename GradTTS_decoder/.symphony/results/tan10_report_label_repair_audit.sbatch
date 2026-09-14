#!/bin/bash
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=tan10_audit_fix
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/completion-audit-fix-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/completion-audit-fix-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tmp/tan10-audit-cache-${SLURM_JOB_ID}
/usr/bin/mkdir -p "$PYTHONPYCACHEPREFIX"
LEDGER=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/pipeline_job_ids.tsv
LEDGER_TMP=${LEDGER}.audit-${SLURM_JOB_ID}.tmp
/usr/bin/awk -v audit_id="$SLURM_JOB_ID" 'BEGIN { FS=OFS="\t" } $1 == "summary_job_id" { $2="11104956" } $1 == "final_audit_job_id" { $2=audit_id } { print }' "$LEDGER" > "$LEDGER_TMP"
/usr/bin/grep -Fx "summary_job_id	11104956" "$LEDGER_TMP"
/usr/bin/grep -Fx "final_audit_job_id	${SLURM_JOB_ID}" "$LEDGER_TMP"
/usr/bin/mv "$LEDGER_TMP" "$LEDGER"
/usr/bin/python3 /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/completion_audit.py
sed -n '1p' /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/report.md | /usr/bin/grep -Fx '# TAN-10 speech metric summary'
/usr/bin/test "$(/usr/bin/wc -l < /mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/summary.csv)" -eq 35
