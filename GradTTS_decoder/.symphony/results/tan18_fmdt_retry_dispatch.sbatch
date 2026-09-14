#!/bin/bash -l
#SBATCH --partition=sheffield
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --job-name=tan18_fmdt_retry_dispatch
#SBATCH --output=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry-dispatch-%j.out
#SBATCH --error=/mnt/parscratch/users/acp23xt/private/DTDM_Unet/.symphony/results/tan18-fmdt-retry-dispatch-%j.err
#SBATCH --chdir=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
set -euo pipefail

ROOT=/mnt/parscratch/users/acp23xt/private/DTDM_Unet
RES="$ROOT/.symphony/results"
MAP="$RES/tan18_fmdt_mappings.tsv"
RETRY_JOBS="$RES/tan18_fmdt_retry_jobs.tsv"
OLD_SUMMARY=11149442
OLD_FINAL=11149445
WER_ARRAY=11149440
UTMOS_ARRAY=11149441
LOSS_JOB=11142111
BRIDGE_JOB=11143257

# These jobs can never satisfy after the first five metrics failed before work.
scancel "$OLD_SUMMARY" "$OLD_FINAL" || true

METRIC_ARRAY=$(sbatch --parsable "$RES/tan18_fmdt_metrics_retry.sbatch")
if [[ ! "$METRIC_ARRAY" =~ ^[0-9]+$ ]]; then
    echo "invalid retry metric array ID: $METRIC_ARRAY" >&2
    exit 1
fi

/usr/bin/python3 - "$MAP" "$METRIC_ARRAY" <<'PY'
import csv
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
array_id = sys.argv[2]
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fields = list(reader.fieldnames or [])
    rows = list(reader)
if fields != ["folder", "sweep", "gen", "out", "metric_job_id", "wer_job_id", "utmos_job_id"]:
    raise RuntimeError("unexpected mappings header: {}".format(fields))
expected = ["model.masking.a:" + value for value in ("0.2", "0.4", "0.6", "0.8", "1")]
if len(rows) != 5 or [row["sweep"] for row in rows] != expected:
    raise RuntimeError("unexpected retry mapping coverage")
for index, row in enumerate(rows):
    row["metric_job_id"] = "{}_{}".format(array_id, index)
temporary = path.with_name(path.name + ".retry-{}.tmp".format(array_id))
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
os.replace(str(temporary), str(path))
PY

SUMMARY=$(sbatch --parsable --export=NIL \
    --dependency="afterok:$METRIC_ARRAY:$WER_ARRAY:$UTMOS_ARRAY:$LOSS_JOB" \
    "$RES/tan18_fmdt_summary.sbatch")
FINAL=$(sbatch --parsable \
    --dependency="afterok:$SUMMARY:$BRIDGE_JOB" \
    "$RES/tan18_all_finalize.sbatch")

printf '%s\n' "$SUMMARY" > "$RES/tan18_fmdt_summary_job_id.txt"
printf '%s\n' "$FINAL" > "$RES/tan18_all_finalize_job_id.txt"
printf 'stage\tjob_id\tdetail\n' > "$RETRY_JOBS"
printf 'failed_metric_jobs\t11149435-11149439\tmodule Anaconda3/2025.06-1 unavailable; no output produced\n' >> "$RETRY_JOBS"
printf 'cancelled_unsatisfiable_summary\t%s\tafterok failed metrics\n' "$OLD_SUMMARY" >> "$RETRY_JOBS"
printf 'cancelled_unsatisfiable_final\t%s\tafterok old summary\n' "$OLD_FINAL" >> "$RETRY_JOBS"
printf 'metric_retry_array\t%s\tfive tasks using established Anaconda3/2022.10 Grad-TTS-EVAL environment\n' "$METRIC_ARRAY" >> "$RETRY_JOBS"
printf 'retained_wer_array\t%s\tfive original pending tasks\n' "$WER_ARRAY" >> "$RETRY_JOBS"
printf 'retained_utmos_array\t%s\tfive original pending tasks\n' "$UTMOS_ARRAY" >> "$RETRY_JOBS"
printf 'summary_retry\t%s\tafter retry metrics, retained WER/UTMOSv2, and completed loss\n' "$SUMMARY" >> "$RETRY_JOBS"
printf 'final_audit_retry\t%s\tafter replacement summary and completed bridge\n' "$FINAL" >> "$RETRY_JOBS"
printf 'metric_retry=%s summary_retry=%s final_retry=%s retained_wer=%s retained_utmos=%s\n' \
    "$METRIC_ARRAY" "$SUMMARY" "$FINAL" "$WER_ARRAY" "$UTMOS_ARRAY"
